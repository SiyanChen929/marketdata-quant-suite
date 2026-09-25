"""LLM backends and proposer, exercised offline with fake clients (no network)."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest

from llm_factor_mining.proposers import (
    AnthropicBackend,
    CachedBackend,
    EvaluatedFeedback,
    FakeBackend,
    LLMProposer,
    PROPOSAL_SCHEMA,
    ProposalContext,
    ProposerError,
    PromptLeakError,
    RejectedFeedback,
    ReplayBackend,
    ReplayMissError,
    find_anonymization_violations,
    load_prompt,
)
from llm_factor_mining.proposers.llm import (
    FALLBACK_BETA,
    MAX_NONSTREAMING_TOKENS,
    REDACTED_TEXT,
    LLMAPIError,
    LLMConfigurationError,
    LLMFormatError,
    LLMRefusalError,
    LLMTransientError,
    LLMTruncatedError,
    request_fingerprint,
)
from llm_factor_mining.search import grammar_card
from llm_factor_mining.dsl.validate import DSLLimits


PAYLOAD = {
    "proposals": [
        {"expression": "-ts_mean(returns, 5)", "rationale": "short-term reversal", "economic_mechanism": "overreaction"},
        {"expression": "volume / ts_mean(volume, 20)", "rationale": "attention", "economic_mechanism": "attention"},
        {"expression": "ts_mean(close,", "rationale": "broken", "economic_mechanism": "none"},
    ]
}


class Block:
    def __init__(self, type_: str, text: str | None = None) -> None:
        self.type = type_
        if text is not None:
            self.text = text


class Response:
    def __init__(self, stop_reason: str, text: str | None = None, *, model: str = "claude-opus-5", category=None):
        self.stop_reason = stop_reason
        self.model = model
        self.usage = SimpleNamespace(input_tokens=120, output_tokens=45)
        self.stop_details = SimpleNamespace(category=category, explanation="declined") if stop_reason == "refusal" else None
        self._content = [Block("thinking"), Block("text", text)] if text is not None else [Block("thinking")]

    @property
    def content(self):
        if self.stop_reason == "refusal":
            raise AssertionError("content must not be read before checking stop_reason")
        return self._content


class Messages:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Client:
    def __init__(self, responses=(), beta_responses=()) -> None:
        self.messages = Messages(responses)
        self.beta = SimpleNamespace(messages=Messages(beta_responses))


def ok(text: str = json.dumps(PAYLOAD), **kwargs) -> Response:
    return Response("end_turn", text, **kwargs)


# --------------------------------------------------------------------------
# AnthropicBackend request construction and response handling
# --------------------------------------------------------------------------


def test_request_shape_has_no_sampling_parameters_and_records_served_model() -> None:
    client = Client([ok(model="claude-opus-5-served")])
    backend = AnthropicBackend(client=client)
    response = backend.complete_json("SYSTEM", "USER", PROPOSAL_SCHEMA)
    (call,) = client.messages.calls
    assert call["model"] == "claude-opus-5"
    assert call["max_tokens"] == 16000
    assert call["system"] == "SYSTEM"
    assert call["messages"] == [{"role": "user", "content": "USER"}]  # no assistant prefill
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"]["effort"] == "high"
    assert call["output_config"]["format"] == {"type": "json_schema", "schema": PROPOSAL_SCHEMA}
    for forbidden in ("temperature", "top_p", "top_k", "budget_tokens", "tool_choice", "betas", "fallbacks"):
        assert forbidden not in call
    assert "budget_tokens" not in json.dumps(call["thinking"])
    assert client.beta.messages.calls == []  # fallback is off by default
    assert response.data == PAYLOAD
    assert response.metadata["served_model"] == "claude-opus-5-served"
    assert response.metadata["requested_model"] == "claude-opus-5"
    assert response.metadata["stop_reason"] == "end_turn"
    assert response.metadata["usage"] == {"input_tokens": 120, "output_tokens": 45}
    assert response.metadata["server_side_fallback"] is False
    assert backend.identity()["model"] == "claude-opus-5"


def test_schema_objects_forbid_additional_properties() -> None:
    assert PROPOSAL_SCHEMA["additionalProperties"] is False
    item = PROPOSAL_SCHEMA["properties"]["proposals"]["items"]
    assert item["additionalProperties"] is False
    assert set(item["required"]) == {"expression", "rationale", "economic_mechanism"}


def test_configurable_model_effort_and_explicit_fallback() -> None:
    client = Client(beta_responses=[ok(model="claude-opus-4-8")])
    backend = AnthropicBackend(client=client, model="claude-sonnet-5", effort="medium", max_tokens=8000, use_fallback=True)
    response = backend.complete_json("s", "u", PROPOSAL_SCHEMA)
    assert client.messages.calls == []
    (call,) = client.beta.messages.calls
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["model"] == "claude-sonnet-5" and call["max_tokens"] == 8000
    assert call["output_config"]["effort"] == "medium"
    assert response.metadata["served_model"] == "claude-opus-4-8"  # the substitution is visible
    assert response.metadata["server_side_fallback"] is True
    with pytest.raises(ValueError):
        AnthropicBackend(effort="extreme")


def test_refusal_is_checked_before_content() -> None:
    backend = AnthropicBackend(client=Client([Response("refusal", category="cyber")]))
    with pytest.raises(LLMRefusalError) as info:
        backend.complete_json("s", "u", PROPOSAL_SCHEMA)
    assert isinstance(info.value, ProposerError)  # recoverable for the search loop
    assert info.value.metadata["stop_details"]["category"] == "cyber"
    assert info.value.metadata["served_model"] == "claude-opus-5"


def test_max_tokens_and_bad_json_are_recoverable_errors() -> None:
    with pytest.raises(LLMTruncatedError):
        AnthropicBackend(client=Client([Response("max_tokens", "{\"propo")])).complete_json("s", "u", PROPOSAL_SCHEMA)
    with pytest.raises(LLMFormatError):
        AnthropicBackend(client=Client([ok("not json")])).complete_json("s", "u", PROPOSAL_SCHEMA)
    with pytest.raises(LLMFormatError):
        AnthropicBackend(client=Client([ok("[1, 2]")])).complete_json("s", "u", PROPOSAL_SCHEMA)
    with pytest.raises(LLMFormatError):
        AnthropicBackend(client=Client([Response("end_turn", None)])).complete_json("s", "u", PROPOSAL_SCHEMA)


def test_pause_turn_is_resumed_with_the_full_assistant_content() -> None:
    paused = Response("pause_turn", None)
    client = Client([paused, ok()])
    response = AnthropicBackend(client=client).complete_json("s", "u", PROPOSAL_SCHEMA)
    first, second = client.messages.calls
    assert len(first["messages"]) == 1
    assert second["messages"][1] == {"role": "assistant", "content": paused.content}
    assert response.metadata["pause_continuations"] == 1


def _sdk_error(name: str, status: int):
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    return getattr(anthropic, name)("error", response=httpx.Response(status, request=request), body=None)


def test_transient_sdk_errors_are_retried_with_backoff() -> None:
    anthropic = pytest.importorskip("anthropic")
    httpx = pytest.importorskip("httpx2")
    request = httpx.Request("POST", "https://example.invalid/v1/messages")
    sleeps: list[float] = []
    transient = [
        _sdk_error("RateLimitError", 429),
        _sdk_error("OverloadedError", 529),
        _sdk_error("InternalServerError", 500),
        anthropic.APIConnectionError(request=request),
    ]
    client = Client([*transient, ok()])
    backend = AnthropicBackend(client=client, sleep=sleeps.append)
    response = backend.complete_json("s", "u", PROPOSAL_SCHEMA)
    assert response.data == PAYLOAD and response.metadata["transient_retries"] == 4
    assert sleeps == [2.0, 4.0, 8.0, 16.0]  # bounded exponential backoff
    with pytest.raises(LLMTransientError) as info:
        AnthropicBackend(client=Client([_sdk_error("RateLimitError", 429)] * 2), max_retries=1, sleep=sleeps.append).complete_json(
            "s", "u", PROPOSAL_SCHEMA
        )
    assert info.value.metadata["attempts"] == 2 and info.value.metadata["status_code"] == 429


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("BadRequestError", 400),
        ("AuthenticationError", 401),
        ("PermissionDeniedError", 403),
        ("NotFoundError", 404),
        ("UnprocessableEntityError", 422),
    ],
)
def test_configuration_errors_are_fatal_not_recoverable(name: str, status: int) -> None:
    client = Client([_sdk_error(name, status)])
    with pytest.raises(LLMConfigurationError) as info:
        AnthropicBackend(client=client, sleep=lambda _: None).complete_json("s", "u", PROPOSAL_SCHEMA)
    assert not isinstance(info.value, ProposerError)  # the search loop must not swallow it
    assert len(client.messages.calls) == 1  # never retried
    assert info.value.metadata["status_code"] == status


def test_other_status_errors_stay_recoverable() -> None:
    with pytest.raises(LLMAPIError):
        AnthropicBackend(client=Client([_sdk_error("APIStatusError", 418)])).complete_json("s", "u", PROPOSAL_SCHEMA)


def test_max_tokens_is_bounded_by_the_non_streaming_limit() -> None:
    AnthropicBackend(max_tokens=MAX_NONSTREAMING_TOKENS)
    with pytest.raises(ValueError, match="stream"):
        AnthropicBackend(max_tokens=MAX_NONSTREAMING_TOKENS + 1)


def test_backend_construction_is_lazy() -> None:
    backend = AnthropicBackend()
    assert backend._client is None  # no client (and no credentials) needed until a call is made
    assert backend.identity() == {
        "backend": "anthropic",
        "model": "claude-opus-5",
        "effort": "high",
        "max_tokens": 16000,
        "thinking": "adaptive",
        "server_side_fallback": False,
    }


# --------------------------------------------------------------------------
# cache and replay
# --------------------------------------------------------------------------


def test_cached_backend_round_trip_and_exact_replay(tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    fake = FakeBackend([PAYLOAD])
    cached = CachedBackend(fake, path, reuse=True)
    first = cached.complete_json("sys", "user", PROPOSAL_SCHEMA, request_tag="replicate=0")
    assert first.metadata["cache"] == "miss" and len(fake.calls) == 1
    again = cached.complete_json("sys", "user", PROPOSAL_SCHEMA, request_tag="replicate=0")
    assert again.metadata["cache"] == "hit" and len(fake.calls) == 1
    assert again.data == first.data

    record = json.loads(path.read_text().splitlines()[0])
    assert record["request"]["user"] == "user" and record["identity"] == fake.identity()
    assert record["key"] == request_fingerprint("sys", "user", PROPOSAL_SCHEMA, fake.identity(), "replicate=0").key

    replay = ReplayBackend(path)
    assert replay.identity() == fake.identity()
    replayed = replay.complete_json("sys", "user", PROPOSAL_SCHEMA, request_tag="replicate=0")
    assert replayed.data == first.data and replayed.metadata["cache"] == "replay"
    assert replayed.metadata["served_model"] == "fake-model"
    with pytest.raises(ReplayMissError):
        replay.complete_json("sys", "user!", PROPOSAL_SCHEMA, request_tag="replicate=0")
    with pytest.raises(ReplayMissError):
        replay.complete_json("sys", "user", PROPOSAL_SCHEMA, request_tag="replicate=1")
    assert not issubclass(ReplayMissError, ProposerError)  # a miss is fatal, never skipped


def test_cached_refusal_replays_as_refusal(tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    backend = AnthropicBackend(client=Client([Response("refusal", category="bio")]))
    cached = CachedBackend(backend, path)
    with pytest.raises(LLMRefusalError):
        cached.complete_json("s", "u", PROPOSAL_SCHEMA)
    with pytest.raises(LLMRefusalError):
        ReplayBackend(path).complete_json("s", "u", PROPOSAL_SCHEMA)


def test_record_files_are_write_only_by_default(tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    live = CachedBackend(FakeBackend([PAYLOAD, PAYLOAD]), path)
    live.complete_json("s", "u", PROPOSAL_SCHEMA)
    again = live.complete_json("s", "u", PROPOSAL_SCHEMA)  # a live run never answers from its own file
    assert again.metadata["cache"] == "miss" and again.metadata["occurrence"] == 1
    with pytest.raises(FileExistsError, match="write-only"):
        CachedBackend(FakeBackend([PAYLOAD]), path)


def test_replay_rejects_mixed_identities_and_missing_file(tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    CachedBackend(FakeBackend([PAYLOAD], model="a"), path).complete_json("s", "u", PROPOSAL_SCHEMA)
    CachedBackend(FakeBackend([PAYLOAD], model="b"), path, reuse=True).complete_json("s", "u", PROPOSAL_SCHEMA)
    with pytest.raises(ValueError):
        ReplayBackend(path)
    assert ReplayBackend(path, identity={"backend": "fake", "model": "b"}).complete_json("s", "u", PROPOSAL_SCHEMA)
    with pytest.raises(FileNotFoundError):
        ReplayBackend(tmp_path / "absent.jsonl")


# --------------------------------------------------------------------------
# prompts, anonymization and the proposer
# --------------------------------------------------------------------------


def _context(**kwargs) -> ProposalContext:
    top = (EvaluatedFeedback("neg(ts_mean(returns,5))", 0.031, 0.22, 4.1, 0.41, 4.5),)
    return ProposalContext(
        round_index=2, n_requested=2, grammar=grammar_card(DSLLimits()), top=top, n_trials_so_far=20,
        budget_remaining=180, **kwargs,
    )


def test_prompt_files_are_versioned_and_hashed() -> None:
    template = load_prompt("propose_v1")
    assert template.sha256 == hashlib.sha256(template.text.encode("utf-8")).hexdigest()
    assert "$grammar" in template.text and "$top_block" in template.text
    with pytest.raises(ValueError):
        load_prompt("../secrets")


def test_anonymization_checker() -> None:
    assert find_anonymization_violations("ts_mean(close, 20) and ts_std(returns, 252)") == []
    assert find_anonymization_violations("data from 2019-03-04")
    assert find_anonymization_violations("the 2008 crisis")
    assert find_anonymization_violations("in September") != []
    assert find_anonymization_violations("buy AAPL now", forbidden_tokens=["AAPL"]) == ["forbidden identifier 'AAPL'"]
    assert find_anonymization_violations("x * 2020", check_years=False) == []
    # sentence-final years, capitalized March/May and abbreviations
    assert find_anonymization_violations("the crisis of 2008.")
    assert find_anonymization_violations("data from 2015.")
    assert find_anonymization_violations("the March 2020 crash", check_years=False)
    assert find_anonymization_violations("since May", check_years=False)
    assert find_anonymization_violations("in Sept.", check_years=False)
    assert find_anonymization_violations("a scale of 2008.5 units") == []  # a decimal, not a year
    assert find_anonymization_violations("signals may march on") == []  # ordinary English words


def test_rendered_prompts_are_anonymized_and_complete() -> None:
    proposer = LLMProposer(FakeBackend([PAYLOAD]), forbidden_tokens=["SYN001"])
    last = (EvaluatedFeedback("volume", 0.01, 0.05, 0.9, 0.2, 1.0),)
    system, user = proposer.render(_context(last_round=last))
    assert "neg(ts_mean(returns,5))" in user and "IC=+0.0310" in user
    assert "Propose exactly 2" in user and "round 2" in user
    assert "Evaluated in the previous round:\n- volume | IC=+0.0100" in user  # v2 shows the last round
    assert "investor attention" not in user and "overreaction" not in user  # no steering examples
    assert find_anonymization_violations(system + user, forbidden_tokens=["SYN001"]) == []
    # model-written rejected text that looks like an identifier or a date is redacted, not fatal
    leaky = _context(
        rejected=(
            RejectedFeedback("SYN001 + close", ("UNKNOWN_TERMINAL",), "unknown data field"),
            RejectedFeedback("ts_mean(returns, 5) in January", ("PARSE",), "unexpected token"),
        )
    )
    _, user = proposer.render(leaky)
    assert "SYN001" not in user and "January" not in user and user.count(REDACTED_TEXT) == 4
    assert "UNKNOWN_TERMINAL" in user and "PARSE" in user
    # harness-written text must be clean: a leak there aborts the run
    with pytest.raises(PromptLeakError):
        proposer.render(ProposalContext(round_index=0, n_requested=2, grammar="data start 2031-01-02"))
    with pytest.raises(PromptLeakError):
        proposer.render(ProposalContext(round_index=0, n_requested=2, grammar="terminal SYN001"))
    # large integer counters are not mistaken for years
    busy = ProposalContext(round_index=1999, n_requested=2, grammar="g", n_trials_so_far=2019, budget_remaining=2100)
    assert "Trials used so far: 2019" in proposer.render(busy)[1]


def test_redacted_rejections_do_not_crash_a_search_or_its_replay(tmp_path) -> None:
    from lfm_helpers import small_market, small_splits, tiny_search_config
    from llm_factor_mining.search import run_search

    market = small_market()
    splits = small_splits(market.panel)
    reply = {"proposals": [{"expression": "ts_mean(returns, 5) in January", "rationale": "r", "economic_mechanism": "m"}]}
    good = {"proposals": [{"expression": "-ts_mean(returns, 5)", "rationale": "r", "economic_mechanism": "m"}]}
    path = tmp_path / "responses.jsonl"
    live = LLMProposer(CachedBackend(FakeBackend([reply, good]), path), forbidden_tokens=market.panel.symbols)
    result = run_search(live, market.panel, splits, tiny_search_config(budget=2, batch_size=1))
    assert result.n_trials == 2 and result.trials[0].status == "invalid" and not result.aborted
    assert live.provenance()["n_redacted_rejections"] == 1
    replay = LLMProposer(ReplayBackend(path), forbidden_tokens=market.panel.symbols)
    again = run_search(replay, market.panel, splits, tiny_search_config(budget=2, batch_size=1))
    assert [t.expression for t in again.trials] == [t.expression for t in result.trials]


def test_llm_proposer_parses_truncates_and_records_calls() -> None:
    fake = FakeBackend([PAYLOAD])
    proposer = LLMProposer(fake, replicate=3, run_tag="snr=0.05,market_seed=1")
    proposals = proposer.propose(_context())
    assert [p.expression for p in proposals] == ["-ts_mean(returns, 5)", "volume / ts_mean(volume, 20)"]
    assert proposals[0].metadata["economic_mechanism"] == "overreaction"
    assert proposals[0].metadata["served_model"] == "fake-model"
    assert fake.calls[0]["tag"] == "replicate=3;snr=0.05,market_seed=1"
    assert fake.calls[0]["schema"] == PROPOSAL_SCHEMA
    call = proposer.calls[0]
    assert call["status"] == "ok" and call["n_returned"] == 3 and call["n_used"] == 2
    assert call["user_template_sha256"] == load_prompt("propose_v2").sha256
    assert proposer.prompt_version == "v2" and LLMProposer(fake, prompt_version="v1").user_template.name == "propose_v1"
    with pytest.raises(ValueError):
        LLMProposer(fake, run_tag="bad;tag")
    provenance = proposer.provenance()
    assert provenance["served_models"] == ["fake-model"] and provenance["n_calls"] == 1
    description = proposer.describe()
    assert description["backend"] == {"backend": "fake", "model": "fake-model"}
    assert description["system_template_sha256"] == load_prompt("system_v2").sha256


def test_llm_proposer_surfaces_format_and_refusal_errors() -> None:
    with pytest.raises(LLMFormatError):
        LLMProposer(FakeBackend([{"ideas": []}])).propose(_context())
    refusing = LLMProposer(AnthropicBackend(client=Client([Response("refusal")])))
    with pytest.raises(LLMRefusalError):
        refusing.propose(_context())
    assert refusing.calls[0]["status"] == "refusal"
    assert refusing.provenance()["status_counts"] == {"refusal": 1}


class FlakyBackend(FakeBackend):
    """Fails its ``n``-th call with a transient error that survived the backend's retries."""

    def __init__(self, responses, *, fail_on: int, error=LLMTransientError) -> None:
        super().__init__(responses)
        self.fail_on = fail_on
        self.error = error
        self.attempts = 0

    def complete_json(self, system, user, schema, *, request_tag=""):
        self.attempts += 1
        if self.attempts == self.fail_on:
            raise self.error("rate limited: 429", {"status_code": 429})
        return super().complete_json(system, user, schema, request_tag=request_tag)


def test_a_run_with_a_transient_failure_replays_exactly(tmp_path) -> None:
    from lfm_helpers import small_market, small_splits, tiny_search_config
    from llm_factor_mining.search import run_search

    market = small_market()
    splits = small_splits(market.panel)

    def respond(system, user, schema):
        round_index = int(user.split("(round ")[1].split(")")[0])
        expressions = ["-ts_mean(returns, 5)", "volume / ts_mean(volume, 20)", "high", "low", "cs_rank(close)", "open"]
        return {"proposals": [{"expression": expressions[round_index % len(expressions)], "rationale": "r", "economic_mechanism": "m"}]}

    path = tmp_path / "llm_responses.jsonl"
    config = tiny_search_config(budget=4, batch_size=1)
    live = LLMProposer(CachedBackend(FlakyBackend(respond, fail_on=2), path))
    first = run_search(live, market.panel, splits, config)
    assert first.n_proposer_errors == 1 and first.n_trials == 4 and first.complete
    replayed = LLMProposer(ReplayBackend(path))
    second = run_search(replayed, market.panel, splits, config)
    assert second.n_proposer_errors == 1
    assert [t.expression for t in second.trials] == [t.expression for t in first.trials]
    assert second.frozen == first.frozen and second.commitment == first.commitment
    assert [r.kind for r in second.ledger.records] == [r.kind for r in first.ledger.records]


def test_fatal_backend_errors_stop_the_search(tmp_path) -> None:
    from lfm_helpers import small_market, small_splits, tiny_search_config
    from llm_factor_mining.search import run_search

    market = small_market()
    splits = small_splits(market.panel)
    proposer = LLMProposer(FlakyBackend([PAYLOAD], fail_on=1, error=LLMConfigurationError))
    with pytest.raises(LLMConfigurationError):
        run_search(proposer, market.panel, splits, tiny_search_config(budget=4), run_dir=tmp_path / "run")
    assert not (tmp_path / "run" / "selected.json").exists()


def test_replay_follows_the_recorded_order_of_outcomes(tmp_path) -> None:
    path = tmp_path / "responses.jsonl"
    cached = CachedBackend(FlakyBackend([PAYLOAD], fail_on=1), path)
    with pytest.raises(LLMTransientError):
        cached.complete_json("s", "u", PROPOSAL_SCHEMA)
    cached.complete_json("s", "u", PROPOSAL_SCHEMA)
    replay = ReplayBackend(path)
    with pytest.raises(LLMTransientError):
        replay.complete_json("s", "u", PROPOSAL_SCHEMA)  # first occurrence: the recorded failure
    assert replay.complete_json("s", "u", PROPOSAL_SCHEMA).data == PAYLOAD  # second: the recorded response
    assert replay.complete_json("s", "u", PROPOSAL_SCHEMA).data == PAYLOAD  # deterministic repeat
    only_failure = tmp_path / "failure.jsonl"
    with pytest.raises(LLMTransientError):
        CachedBackend(FlakyBackend([PAYLOAD], fail_on=1), only_failure).complete_json("s", "u", PROPOSAL_SCHEMA)
    replay = ReplayBackend(only_failure)
    with pytest.raises(LLMTransientError):
        replay.complete_json("s", "u", PROPOSAL_SCHEMA)
    with pytest.raises(ReplayMissError):
        replay.complete_json("s", "u", PROPOSAL_SCHEMA)
