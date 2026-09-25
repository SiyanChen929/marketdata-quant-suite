"""Backend-neutral types, the scripted backend, and record/replay."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from marketdata_agent import Copilot
from marketdata_agent.backends import (
    BackendTurn,
    LLMBackend,
    RecordingBackend,
    ReplayBackend,
    ReplayMissError,
    ScriptExhausted,
    ScriptedBackend,
    ScriptedTurn,
    normalize_blocks,
    parse_tool_result,
    request_fingerprint,
    to_plain,
)
from marketdata_agent.backends.base import accumulate_usage

from conftest import AS_OF


def test_to_plain_drops_none_and_handles_models_namespaces_and_dataclasses():
    class Model:
        def model_dump(self, mode="python", exclude_none=False):
            assert mode == "json" and exclude_none
            return {"type": "text", "text": "hi"}

    value = {"a": None, "b": [SimpleNamespace(type="tool_use", id="t", input={"x": 1}, _private=2)], "c": Model()}
    assert to_plain(value) == {"b": [{"type": "tool_use", "id": "t", "input": {"x": 1}}], "c": {"type": "text", "text": "hi"}}
    assert to_plain(to_plain(value)) == to_plain(value)  # idempotent: SDK objects and recorded dicts hash alike


def test_normalize_blocks_keeps_text_and_tool_use_only():
    content = [
        SimpleNamespace(type="thinking", thinking="", signature="sig"),
        {"type": "text", "text": "Checking."},
        SimpleNamespace(type="tool_use", id="toolu_1", name="list_symbols", input={}),
        {"type": "fallback", "from": {"model": "a"}, "to": {"model": "b"}},
    ]
    parts = normalize_blocks(content)
    assert [p.type for p in parts] == ["text", "tool_use"]
    turn = BackendTurn.from_content(content, stop_reason="tool_use", served_model="m")
    assert turn.text == "Checking." and turn.tool_calls[0].name == "list_symbols"
    assert turn.assistant_message()["content"][0] is content[0]  # raw content, unchanged


def test_backend_turn_round_trips_through_json():
    turn = BackendTurn.from_content(
        [{"type": "text", "text": "ANSWER: 1"}],
        stop_reason="end_turn",
        served_model="claude-opus-5",
        requested_model="claude-opus-5",
        usage={"input_tokens": 3, "output_tokens": 4},
        request_ids=("req_1",),
    )
    again = BackendTurn.from_dict(turn.to_dict())
    assert again.to_dict() == turn.to_dict() and again.text == "ANSWER: 1"


def test_usage_accumulation_sums_integers_recursively():
    total: dict = {}
    accumulate_usage(total, {"input_tokens": 3, "cache_creation": {"ephemeral_5m_input_tokens": 1}, "service_tier": "standard"})
    accumulate_usage(total, {"input_tokens": 4, "cache_creation": {"ephemeral_5m_input_tokens": 2}, "service_tier": "priority"})
    assert total == {"input_tokens": 7, "cache_creation": {"ephemeral_5m_input_tokens": 3}, "service_tier": "priority"}


def test_parse_tool_result_reads_ids_payloads_and_errors(runtime):
    ok = runtime.call_tool("list_symbols", {}, tool_use_id="t1").to_tool_result_block()
    parsed = parse_tool_result(ok)
    assert parsed.result_id == runtime.outcomes[0].result.result_id and parsed.tool == "list_symbols"
    assert parsed.payload["count"] == 6 and not parsed.is_error  # SYN06 has listed by AS_OF
    bad = runtime.call_tool("period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "2023-12-29"}, tool_use_id="t2")
    parsed_bad = parse_tool_result(bad.to_tool_result_block())
    assert parsed_bad.is_error and parsed_bad.error_code == "lookahead_violation" and parsed_bad.result_id is None


def test_scripted_backend_is_an_llm_backend_with_deterministic_ids():
    backend = ScriptedBackend([ScriptedTurn(tool_calls=(("list_symbols", {}), ("list_symbols", {}))), ScriptedTurn(text="done")])
    assert isinstance(backend, LLMBackend)
    first = backend.step("sys", [{"role": "user", "content": "q"}], [])
    assert [c.id for c in first.tool_calls] == ["toolu_s00_00", "toolu_s00_01"] and first.stop_reason == "tool_use"
    second = backend.step("sys", [], [])
    assert second.text == "done" and second.stop_reason == "end_turn" and second.served_model == "scripted:scripted"
    with pytest.raises(ScriptExhausted):
        backend.step("sys", [], [])


def test_request_fingerprint_is_sensitive_to_every_part():
    base = ("sys", [{"role": "user", "content": "q"}], [{"name": "t"}], {"model": "m"})
    fp = request_fingerprint(*base)
    assert fp == request_fingerprint(*base)
    assert fp != request_fingerprint("sys2", *base[1:])
    assert fp != request_fingerprint(base[0], [{"role": "user", "content": "q2"}], *base[2:])
    assert fp != request_fingerprint(*base[:3], {"model": "m2"})


def _script() -> ScriptedBackend:
    def policy(state):
        if state.step == 0:
            return ScriptedTurn(tool_calls=(("period_return", {"symbol": "SYN01", "start": "2023-01-02", "end": "latest"}),))
        result = state.last_results[0]
        return ScriptedTurn(text=f"ANSWER: {result.payload['simple_return'] * 100:.2f}% [r:{result.result_id}]")

    return ScriptedBackend(policy, name="rec")


def test_record_then_replay_reproduces_the_episode_offline(source, tmp_path):
    path = tmp_path / "turns.jsonl"
    recorded = Copilot(RecordingBackend(_script(), path), source=source, as_of=AS_OF).run("Return of SYN01 in 2023?")
    replay = ReplayBackend(path)
    assert len(replay) == 2 and replay.describe()["name"] == "rec"
    replayed = Copilot(replay, source=source, as_of=AS_OF).run("Return of SYN01 in 2023?")
    assert replayed.answer == recorded.answer and replayed.status == "completed"
    assert [c.result_id for c in replayed.tool_calls] == [c.result_id for c in recorded.tool_calls]
    assert replayed.grounding.grounding_rate == 1.0

    missing = Copilot(ReplayBackend(path), source=source, as_of=AS_OF).run("A different question?")
    assert missing.status == "backend_error" and missing.error["kind"] == "replay_miss"
    with pytest.raises(ReplayMissError):
        ReplayBackend(path).step("other system prompt", [{"role": "user", "content": "q"}], [])


def test_a_recording_is_loaded_not_truncated_and_recorded_requests_are_reused(source, tmp_path):
    path = tmp_path / "turns.jsonl"
    first = RecordingBackend(_script(), path)
    Copilot(first, source=source, as_of=AS_OF).run("Return of SYN01 in 2023?")
    assert first.live_calls == 2 and first.reused == 0

    class Exploding:
        name = "rec"

        def describe(self):
            return _script().describe()

        def step(self, system, messages, tools):
            raise AssertionError("a recorded request must not reach the model again")

    resumed = RecordingBackend(Exploding(), path)
    episode = Copilot(resumed, source=source, as_of=AS_OF).run("Return of SYN01 in 2023?")
    assert episode.status == "completed" and resumed.reused == 2 and resumed.live_calls == 0
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(ValueError):  # a recording from another configuration is never mixed in
        RecordingBackend(ScriptedBackend([], name="other"), path)


def test_a_missing_replay_file_is_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ReplayBackend(tmp_path / "absent.jsonl")


def test_served_models_cover_every_segment():
    turn = BackendTurn.from_content([], stop_reason="end_turn", served_model="b", segment_models=("a", "b", "b"))
    assert turn.served_models == ("a", "b")
    assert BackendTurn.from_dict(turn.to_dict()).segment_models == ("a", "b", "b")
    assert BackendTurn.from_content([], stop_reason="end_turn", served_model="m").served_models == ("m",)
