"""AnthropicBackend with an injected fake client: request shape, history, stop reasons, errors.

No network: the fake client records every ``create`` call and returns scripted
responses shaped like SDK ``Message`` objects.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from marketdata_agent import AuditLog, Copilot, read_records, verify_chain
from marketdata_agent.backends import (
    DEFAULT_MODEL,
    FALLBACK_BETA,
    AnthropicBackend,
    AnthropicConfig,
    BackendError,
)
from marketdata_agent.backends.anthropic_backend import NONSTREAMING_MAX_TOKENS

from conftest import AS_OF


FORBIDDEN_REQUEST_KEYS = {"temperature", "top_p", "top_k", "tool_choice", "budget_tokens"}


def text(value: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=value)


def thinking() -> SimpleNamespace:
    return SimpleNamespace(type="thinking", thinking="", signature="sig-abc")


def tool_use(block_id: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=block_id, name=name, input=args)


def response(content, stop_reason="end_turn", model=DEFAULT_MODEL, request_id="req_1") -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        model=model,
        usage=SimpleNamespace(input_tokens=100, output_tokens=20, cache_read_input_tokens=0),
        stop_details=None,
        _request_id=request_id,
    )


class Refusal:
    stop_reason = "refusal"
    model = DEFAULT_MODEL
    usage = SimpleNamespace(input_tokens=10, output_tokens=0)
    stop_details = SimpleNamespace(type="refusal", category="cyber", explanation="declined")
    _request_id = "req_refusal"

    @property
    def content(self):  # the backend must check stop_reason before reading content
        raise AssertionError("content was read on a refusal")


class FakeStream:
    def __init__(self, message):
        self.message = message
        self.request_id = "req_stream"

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.streamed: list[dict] = []

    def _next(self, kwargs, log):
        snapshot = dict(kwargs)
        snapshot["messages"] = list(kwargs["messages"])  # the loop keeps appending to its own list
        log.append(snapshot)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def create(self, **kwargs):
        return self._next(kwargs, self.calls)

    def stream(self, **kwargs):
        return FakeStream(self._next(kwargs, self.streamed))


class FakeClient:
    def __init__(self, responses=(), beta_responses=()):
        self.messages = FakeMessages(responses)
        self.beta = SimpleNamespace(messages=FakeMessages(beta_responses))


def copilot(client, source, *, audit=None, config=None, max_steps=8) -> Copilot:
    backend = AnthropicBackend(config or AnthropicConfig(), client=client)
    return Copilot(backend, source=source, as_of=AS_OF, audit=audit, max_steps=max_steps)


def test_request_shape_follows_the_current_api(source):
    client = FakeClient([response([text("ANSWER: INSUFFICIENT_DATA")])])
    copilot(client, source).run("What will SYN01 close at next week?")
    (call,) = client.messages.calls
    assert call["model"] == "claude-opus-5" == DEFAULT_MODEL
    assert call["max_tokens"] == 16000
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": "high"}
    assert not (FORBIDDEN_REQUEST_KEYS & set(call)) and "betas" not in call and "fallbacks" not in call
    assert all(tool["strict"] is True and tool["input_schema"]["additionalProperties"] is False for tool in call["tools"])
    assert isinstance(call["system"], str) and "2023-06-30" in call["system"] and "INSUFFICIENT_DATA" in call["system"]
    assert call["messages"] == [{"role": "user", "content": "What will SYN01 close at next week?"}]
    assert client.beta.messages.calls == []


def test_full_content_is_appended_and_tool_results_share_one_user_message(source, tmp_path):
    first = [
        thinking(),
        text("Two calls."),
        tool_use("toolu_1", "period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "latest"}),
        tool_use("toolu_2", "period_return", {"symbol": "SYN02", "start": "2023-01-02", "end": "2023-07-31"}),
    ]
    client = FakeClient([response(first, "tool_use"), response([text("ANSWER: done")])])
    audit = AuditLog(tmp_path / "ep.audit.jsonl", fsync=False)
    episode = copilot(client, source, audit=audit).run("Compare SYN01 and SYN02 in 2023.")
    second_call = client.messages.calls[1]["messages"]
    assert second_call[1]["role"] == "assistant"
    assert len(second_call[1]["content"]) == 4
    assert all(a is b for a, b in zip(second_call[1]["content"], first))  # thinking block passed back unchanged
    results = second_call[2]
    assert results["role"] == "user" and [b["tool_use_id"] for b in results["content"]] == ["toolu_1", "toolu_2"]
    assert [b["is_error"] for b in results["content"]] == [False, True]
    assert results["content"][1]["content"].startswith("[error:lookahead_violation]")
    assert episode.lookahead_attempts == 1 and episode.status == "completed"
    assert episode.usage["input_tokens"] == 200 and episode.served_models == ("claude-opus-5",)
    assert verify_chain(audit.path, expected_head=episode.audit_head).ok


def test_fallback_is_an_explicit_flag_and_the_served_model_is_recorded(source, tmp_path):
    served = "claude-opus-4-8"
    client = FakeClient(beta_responses=[response([text("ANSWER: INSUFFICIENT_DATA")], model=served)])
    audit = AuditLog(tmp_path / "ep.audit.jsonl", fsync=False)
    episode = copilot(client, source, audit=audit, config=AnthropicConfig(fallback=True)).run("Next week's close?")
    (call,) = client.beta.messages.calls
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default" and call["model"] == DEFAULT_MODEL
    assert client.messages.calls == []
    assert episode.served_models == (served,) and episode.requested_model == DEFAULT_MODEL
    model_call = next(r for r in read_records(audit.path) if r["kind"] == "model_call")["payload"]
    assert model_call["served_model"] == served and model_call["served_model_differs"] is True
    assert model_call["fallback_enabled"] is True and model_call["usage"]["input_tokens"] == 100
    assert AnthropicConfig().fallback is False  # benchmark default


def test_refusal_ends_the_episode_without_reading_content(source):
    client = FakeClient([Refusal()])
    episode = copilot(client, source).run("Question")
    assert episode.status == "refusal" and episode.stop_reason == "refusal" and episode.answer == ""
    assert episode.transcript == ({"role": "user", "content": "Question"},)


def test_pause_turn_is_resent_with_the_paused_content(source):
    paused = [text("Working on it.")]
    client = FakeClient([response(paused, "pause_turn"), response([text("ANSWER: INSUFFICIENT_DATA")])])
    backend = AnthropicBackend(client=client)
    turn = backend.step("sys", [{"role": "user", "content": "q"}], [])
    assert len(client.messages.calls) == 2 and turn.continuations == 1
    resent = client.messages.calls[1]["messages"]
    assert resent[-1] == {"role": "assistant", "content": paused}
    assert turn.text == "Working on it.\nANSWER: INSUFFICIENT_DATA" and turn.stop_reason == "end_turn"
    assert turn.usage["input_tokens"] == 200 and turn.request_ids == ("req_1", "req_1")


def test_max_tokens_is_recoverable_and_truncated_tool_calls_are_not_run(source, tmp_path):
    cut = [tool_use("toolu_cut", "period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "latest"})]
    client = FakeClient([response(cut, "max_tokens"), response([text("ANSWER: INSUFFICIENT_DATA")])])
    audit = AuditLog(tmp_path / "ep.audit.jsonl", fsync=False)
    episode = copilot(client, source, audit=audit).run("Question")
    followup = client.messages.calls[1]["messages"][2]["content"]
    assert followup[0]["tool_use_id"] == "toolu_cut" and followup[0]["is_error"] is True
    assert followup[0]["content"].startswith("[error:output_truncated]")
    assert followup[1]["type"] == "text"
    assert episode.tool_calls == () and episode.max_tokens_events == 1 and episode.status == "completed"
    kinds = [r["kind"] for r in read_records(audit.path)]
    assert "tool_call_skipped" in kinds and "policy_decision" not in kinds


def test_max_steps_bounds_a_model_that_never_stops(source):
    loop = [response([tool_use(f"t{i}", "list_symbols", {})], "tool_use") for i in range(3)]
    episode = copilot(FakeClient(loop), source, max_steps=3).run("Question")
    assert episode.status == "max_steps" and episode.steps == 3 and len(episode.tool_calls) == 3


def _sdk():
    """The SDK error classes are exercised only when the optional ``anthropic`` extra is installed."""

    return pytest.importorskip("anthropic"), pytest.importorskip("httpx2")


def _status_error(name: str, status: int):
    anthropic, httpx = _sdk()
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    return getattr(anthropic, name)("simulated", response=httpx.Response(status, request=request), body=None)


@pytest.mark.parametrize(
    ("name", "status", "kind", "retryable"),
    [
        ("RateLimitError", 429, "rate_limit", True),
        ("AuthenticationError", 401, "authentication", False),
        ("PermissionDeniedError", 403, "authentication", False),
        ("NotFoundError", 404, "not_found", False),
        ("BadRequestError", 400, "bad_request", False),
        ("InternalServerError", 500, "server_error", True),
        ("APIConnectionError", None, "connection", True),
    ],
)
def test_sdk_errors_are_classified_most_specific_first(name, status, kind, retryable):
    anthropic, httpx = _sdk()
    if status is None:
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"))
    else:
        error = _status_error(name, status)
    backend = AnthropicBackend(client=FakeClient([error]))
    with pytest.raises(BackendError) as caught:
        backend.step("sys", [{"role": "user", "content": "q"}], [])
    assert caught.value.kind == kind and caught.value.retryable is retryable


def test_backend_error_ends_the_episode_and_is_audited(source, tmp_path):
    audit = AuditLog(tmp_path / "ep.audit.jsonl", fsync=False)
    episode = copilot(FakeClient([_status_error("RateLimitError", 429)]), source, audit=audit).run("Question")
    assert episode.status == "backend_error" and episode.error["kind"] == "rate_limit"
    kinds = [r["kind"] for r in read_records(audit.path)]
    assert kinds == ["episode_start", "model_error", "final_answer"]


def test_missing_credentials_become_a_backend_error_not_a_crash(source):
    _sdk()
    message = "Could not resolve authentication method. Expected one of api_key, auth_token, or credentials to be set."
    episode = copilot(FakeClient([TypeError(message)]), source).run("Question")
    assert episode.status == "backend_error" and episode.error["kind"] == "missing_credentials"
    with pytest.raises(TypeError):  # unrelated TypeErrors are programming errors and still raise
        AnthropicBackend(client=FakeClient([TypeError("unexpected keyword")])).step("s", [{"role": "user", "content": "q"}], [])


def test_config_validation_and_description():
    with pytest.raises(ValueError):
        AnthropicConfig(effort="extreme")
    with pytest.raises(ValueError):
        AnthropicConfig(model=" ")
    described = AnthropicConfig(model="claude-opus-5", effort="max").to_dict()
    assert described["thinking"] == {"type": "adaptive"} and described["fallback"] is False
    assert copy.deepcopy(described) == described


def test_large_output_budgets_are_streamed_instead_of_crashing(source):
    """Regression: the SDK refuses non-streaming requests above about 21,333 max_tokens."""

    assert not AnthropicConfig().streaming and AnthropicConfig().to_dict()["transport"] == "create"
    config = AnthropicConfig(max_tokens=64000)
    assert config.streaming and config.to_dict()["transport"] == "stream" and NONSTREAMING_MAX_TOKENS == 21_333
    client = FakeClient([response([text("ANSWER: INSUFFICIENT_DATA")], request_id=None)])
    episode = copilot(client, source, config=config).run("Question")
    assert episode.status == "completed" and client.messages.calls == [] and len(client.messages.streamed) == 1
    assert client.messages.streamed[0]["max_tokens"] == 64000 and "tool_choice" not in client.messages.streamed[0]
    beta = FakeClient(beta_responses=[response([text("ANSWER: INSUFFICIENT_DATA")])])
    copilot(beta, source, config=AnthropicConfig(max_tokens=64000, fallback=True)).run("Question")
    assert beta.beta.messages.streamed[0]["betas"] == [FALLBACK_BETA]


def test_the_sdk_streaming_refusal_becomes_a_backend_error_not_a_crash(source):
    _sdk()
    error = ValueError("Streaming is required for operations that may take longer than 10 minutes.")
    episode = copilot(FakeClient([error]), source).run("Question")
    assert episode.status == "backend_error" and episode.error["kind"] == "bad_request"


def test_every_segment_model_and_the_refusal_details_are_logged(source, tmp_path):
    paused = response([text("Working.")], "pause_turn", model="claude-substitute")
    client = FakeClient([paused, response([text("ANSWER: INSUFFICIENT_DATA")])])
    audit = AuditLog(tmp_path / "ep.audit.jsonl", fsync=False)
    episode = copilot(client, source, audit=audit).run("Question")
    assert episode.served_models == ("claude-substitute", DEFAULT_MODEL)
    call = next(r for r in read_records(audit.path) if r["kind"] == "model_call")["payload"]
    assert call["served_models"] == ["claude-substitute", DEFAULT_MODEL] and call["served_model_differs"] is True
    assert call["continuations"] == 1

    refusal_audit = AuditLog(tmp_path / "refusal.audit.jsonl", fsync=False)
    refused = copilot(FakeClient([Refusal()]), source, audit=refusal_audit).run("Question")
    assert refused.stop_details == ({"turn": 1, "type": "refusal", "category": "cyber", "explanation": "declined"},)
    logged = next(r for r in read_records(refusal_audit.path) if r["kind"] == "model_call")["payload"]
    assert logged["stop_details"]["category"] == "cyber"
