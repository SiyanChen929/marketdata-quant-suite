"""The copilot loop driven by the scripted backend: tools, denials, limits, audit, ablation."""

from __future__ import annotations

from datetime import date

import pytest

from marketdata_agent import (
    ABSTAIN_TOKEN,
    Copilot,
    Policy,
    ScriptedBackend,
    ScriptedTurn,
    load_system_prompt,
    read_records,
    verify_chain,
)
from marketdata_agent.agent import PROMPT_VERSION, default_episode_id

from conftest import AS_OF, LATE_SYMBOL


RETURN_ARGS = {"symbol": "SYN01", "start": "2023-01-02", "end": "latest"}


def cite_return(state) -> ScriptedTurn:
    result = state.last_results[0]
    return ScriptedTurn(text=f"ANSWER: {result.payload['simple_return'] * 100:.2f}% [r:{result.result_id}]")


def test_tool_calls_run_and_the_cited_answer_is_grounded(source, audit):
    backend = ScriptedBackend([ScriptedTurn(tool_calls=(("period_return", RETURN_ARGS),)), cite_return])
    episode = Copilot(backend, source=source, as_of=AS_OF, audit=audit).run("SYN01 return in H1 2023?", episode_id="ep-1")
    assert episode.status == "completed" and episode.steps == 2
    (call,) = episode.tool_calls
    assert call.allowed and call.result_id in episode.results and call.last_date == "2023-06-30"
    assert episode.grounding.n_claims == 1 and episode.grounding.grounding_rate == 1.0
    assert episode.grounding.cited_ids == (call.result_id,)
    assert episode.leaked_results == () and episode.lookahead_attempts == 0
    assert episode.transcript[1]["role"] == "assistant" and episode.transcript[2]["content"][0]["type"] == "tool_result"


def test_policy_denial_is_returned_as_an_error_result_the_model_can_read(source):
    seen = {}

    def react(state) -> ScriptedTurn:
        seen["results"] = state.last_results
        return ScriptedTurn(text=f"ANSWER: {ABSTAIN_TOKEN}\nThe period ends after the as-of date.")

    future = {"symbol": "SYN01", "start": "2023-07-03", "end": "2023-09-29"}
    backend = ScriptedBackend([ScriptedTurn(tool_calls=(("period_return", future),)), react])
    episode = Copilot(backend, source=source, as_of=AS_OF).run("SYN01 return in Q3 2023?")
    (result,) = seen["results"]
    assert result.is_error and result.error_code == "lookahead_violation"
    assert episode.lookahead_attempts == 1 and episode.violations == {"lookahead_violation": 2}  # start and end
    assert not episode.tool_calls[0].allowed and episode.tool_calls[0].decision_code == "lookahead_violation"
    assert episode.results == {} and episode.answer.startswith("ANSWER: INSUFFICIENT_DATA")


def test_execution_tools_are_refused_and_counted(source):
    backend = ScriptedBackend(
        [
            ScriptedTurn(tool_calls=(("execute_order", {"symbol": "SYN01", "side": "buy", "quantity": 10}),)),
            ScriptedTurn(text="ANSWER: EXECUTION_REFUSED"),
        ]
    )
    episode = Copilot(backend, source=source, as_of=AS_OF).run("Buy 10 SYN01.")
    assert episode.order_execution_attempts == 1 and episode.violations == {"order_execution_forbidden": 1}
    assert episode.proposals == ()


def test_max_steps_is_enforced(source):
    backend = ScriptedBackend(lambda state: ScriptedTurn(tool_calls=(("list_symbols", {}),)))
    episode = Copilot(backend, source=source, as_of=AS_OF, max_steps=3).run("Loop forever.")
    assert episode.status == "max_steps" and episode.steps == 3 and episode.answer == ""
    assert len(episode.tool_calls) == 3 and backend.calls == 3


def test_tool_budget_denials_reach_the_model(source):
    backend = ScriptedBackend(lambda state: ScriptedTurn(tool_calls=(("list_symbols", {}),) * 3))
    episode = Copilot(backend, source=source, as_of=AS_OF, policy=Policy(max_tool_calls=4), max_steps=2).run("Q")
    codes = [call.decision_code for call in episode.tool_calls]
    assert codes == ["allowed"] * 4 + ["tool_budget_exhausted"] * 2


def test_script_exhaustion_is_a_backend_error(source):
    episode = Copilot(ScriptedBackend([]), source=source, as_of=AS_OF).run("Q")
    assert episode.status == "backend_error" and episode.error["kind"] == "script_exhausted"


def test_audit_log_records_the_whole_episode_and_verifies_against_its_head(source, audit):
    backend = ScriptedBackend([ScriptedTurn(tool_calls=(("period_return", RETURN_ARGS),)), cite_return])
    episode = Copilot(backend, source=source, as_of=AS_OF, audit=audit).run("SYN01 return?", episode_id="ep-audit")
    records = read_records(audit.path)
    assert [r["kind"] for r in records] == [
        "episode_start",
        "model_call",
        "policy_decision",
        "tool_call",
        "model_call",
        "final_answer",
    ]
    start = records[0]["payload"]
    prompt = load_system_prompt(AS_OF, Policy())
    assert start["prompt"] == prompt.meta() and start["prompt"]["version"] == PROMPT_VERSION
    assert start["clock_enforced"] is True and start["data_cutoff"] == "2023-06-30"
    assert start["backend"]["backend"] == "scripted" and len(start["tools_sha256"]) == 64
    final = records[-1]["payload"]
    assert final["cited_result_ids"] == [episode.tool_calls[0].result_id]
    assert final["grounding"]["grounding_rate"] == 1.0 and final["status"] == "completed"
    assert records[1]["payload"]["served_model"] == "scripted:scripted"
    assert episode.audit_head.records == len(records)
    assert verify_chain(audit.path, expected_head=episode.audit_head).ok


def test_system_prompt_states_the_protocol_and_is_versioned():
    prompt = load_system_prompt(date(2022, 3, 31), Policy(max_tool_calls=12))
    for phrase in ("2022-03-31", "INSUFFICIENT_DATA", "EXECUTION_REFUSED", "[r:", "ANSWER:", "12 tool calls", "propose_order"):
        assert phrase in prompt.text
    assert "$" + "as_of" not in prompt.text and len(prompt.sha256) == 64
    assert load_system_prompt(date(2022, 3, 31), Policy(max_tool_calls=12)).sha256 == prompt.sha256
    assert load_system_prompt(date(2022, 4, 1), Policy()).template_sha256 == prompt.template_sha256


def test_clock_ablation_serves_only_explicitly_named_later_dates_and_counts_them(source, audit):
    future = {"symbol": "SYN01", "start": "2023-07-03", "end": "2023-09-29"}
    backend = ScriptedBackend([ScriptedTurn(tool_calls=(("period_return", future), ("list_symbols", {}))), ScriptedTurn(text="x")])
    episode = Copilot(backend, source=source, as_of=date(2023, 2, 15), audit=audit, enforce_clock=False).run("Q")
    assert all(call.allowed and call.error_code is None for call in episode.tool_calls)
    future_call, listing_call = episode.tool_calls
    # The explicitly named later dates are served, and still counted as a look-ahead attempt.
    assert future_call.unenforced == ("lookahead_violation",) and episode.lookahead_attempts == 1
    assert episode.metrics["lookahead_unenforced"] == 1
    assert episode.leaked_results == (future_call.result_id,)
    # Everything derived implicitly still refers to the nominal as-of date.
    listing = episode.results[listing_call.result_id]
    assert LATE_SYMBOL not in [row["symbol"] for row in listing.payload["symbols"]]
    assert listing.payload["as_of"] == "2023-02-15" and listing.payload["latest_session"] == "2023-02-15"
    assert all("as_of=2023-02-15" in result.text for result in episode.results.values())
    start = read_records(audit.path)[0]["payload"]
    assert start["clock_enforced"] is False and start["data_cutoff"] == "9999-12-31" and start["as_of"] == "2023-02-15"


LATEST_CALLS = (
    ("list_symbols", {}),
    ("get_daily_bars", {"symbol": "SYN01", "start": "2022-06-24", "end": "latest"}),
    ("period_return", {"symbol": "SYN02", "start": "2021-01-04", "end": "latest"}),
    ("realized_volatility", {"symbol": "SYN03", "window": 20, "end": "latest"}),
    ("rolling_correlation", {"a": "SYN01", "b": "SYN02", "window": 60, "end": "latest"}),
    ("max_drawdown", {"symbol": "SYN04", "start": "2021-12-31", "end": "latest"}),
    ("compare_returns", {"symbols": ["SYN01", "SYN02", "SYN03"], "start": "2022-03-31", "end": "latest"}),
    ("propose_order", {"symbol": "SYN05", "side": "buy", "quantity": 10, "rationale": "test"}),
)


def test_a_policy_that_never_names_a_later_date_is_unaffected_by_the_ablation(source):
    """Regression: the ablation once resolved 'latest' to 9999-12-31, denying spans and leaking the future."""

    def run(enforce: bool):
        backend = ScriptedBackend([ScriptedTurn(tool_calls=LATEST_CALLS), ScriptedTurn(text="ANSWER: done")])
        return Copilot(backend, source=source, as_of=date(2022, 6, 30), enforce_clock=enforce).run("Q", episode_id="ep-a1")

    enforced, ablated = run(True), run(False)
    assert all(call.allowed and call.error_code is None for call in ablated.tool_calls)
    assert ablated.leaked_results == () and ablated.lookahead_attempts == 0
    assert [c.result_id for c in ablated.tool_calls] == [c.result_id for c in enforced.tool_calls]
    assert [r.text for r in ablated.results.values()] == [r.text for r in enforced.results.values()]


def test_a_verifier_failure_is_audited_and_marks_the_episode_ungrounded(source, audit, monkeypatch):
    import marketdata_agent.agent as agent_module

    def broken(*args, **kwargs):
        raise RuntimeError("verifier defect")

    monkeypatch.setattr(agent_module, "verify_grounding", broken)
    episode = Copilot(ScriptedBackend([ScriptedTurn(text="ANSWER: 5.23%")]), source=source, as_of=AS_OF, audit=audit).run("Q")
    assert episode.status == "completed" and episode.grounding.error == "RuntimeError: verifier defect"
    assert episode.grounding.fully_grounded is False
    kinds = [r["kind"] for r in read_records(audit.path)]
    assert kinds[-2:] == ["grounding_error", "final_answer"]


def test_a_degenerate_digit_run_in_the_answer_does_not_crash_the_episode(source, audit):
    episode = Copilot(ScriptedBackend([ScriptedTurn(text="ANSWER: " + "9" * 5000)]), source=source, as_of=AS_OF, audit=audit).run("Q")
    assert episode.status == "completed" and episode.grounding.error is None
    assert [check.status for check in episode.grounding.checks] == ["unparsed"]
    assert read_records(audit.path)[-1]["kind"] == "final_answer"


def test_copilot_accepts_a_clock_object(source):
    from marketdata_agent import AsOfClock

    copilot = Copilot(ScriptedBackend([ScriptedTurn(text="ANSWER: INSUFFICIENT_DATA")]), source=source, as_of=AsOfClock(AS_OF))
    assert copilot.clock.as_of == AS_OF and copilot.run("Q").as_of == "2023-06-30"


def test_episode_ids_are_deterministic_and_inputs_are_validated(source):
    with pytest.raises(ValueError):
        Copilot(ScriptedBackend([]), source=source, as_of=AS_OF, prompt_version="copilot_system_v9")
    assert default_episode_id("q", AS_OF, {"model": "m"}) == default_episode_id("q", AS_OF, {"model": "m"})
    assert default_episode_id("q", AS_OF, {"model": "m"}) != default_episode_id("q2", AS_OF, {"model": "m"})
    with pytest.raises(ValueError):
        Copilot(ScriptedBackend([]), source=source, as_of=AS_OF).run("   ")
    with pytest.raises(ValueError):
        Copilot(ScriptedBackend([]), source=source, as_of=AS_OF, max_steps=0)


def test_episode_serializes_with_and_without_transcript(source):
    backend = ScriptedBackend([ScriptedTurn(tool_calls=(("period_return", RETURN_ARGS),)), cite_return])
    episode = Copilot(backend, source=source, as_of=AS_OF).run("Q")
    compact = episode.to_dict()
    assert "transcript" not in compact and compact["grounding"]["n_claims"] == 1
    full = episode.to_dict(include_transcript=True)
    assert len(full["transcript"]) == 4 and full["transcript"][0] == {"role": "user", "content": "Q"}
