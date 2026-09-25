from __future__ import annotations

from datetime import date

import pytest

from marketdata_agent import (
    AsOfClock,
    Code,
    Policy,
    PolicyConfigError,
    PolicyGate,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
    default_registry,
)

from conftest import AS_OF


CLOCK = AsOfClock(AS_OF)
RANGE = {"symbol": "SYN01", "start": "2023-01-02", "end": "2023-06-30"}


def _gate(**policy_kwargs) -> PolicyGate:
    policy = Policy(**policy_kwargs)
    return PolicyGate(policy, default_registry(policy))


def test_policy_defaults_and_non_negotiable_controls():
    policy = Policy()
    assert policy.forbid_provisional and policy.forbid_order_execution
    assert "propose_order" in policy.allowed_tools
    assert len(policy.fingerprint()) == 64
    with pytest.raises(PolicyConfigError):
        Policy(forbid_provisional=False)
    with pytest.raises(PolicyConfigError):
        Policy(forbid_order_execution=False)
    with pytest.raises(PolicyConfigError):
        Policy(allowed_tools={"period_return", "execute_order"})
    with pytest.raises(PolicyConfigError):
        Policy(max_tool_calls=0)
    with pytest.raises(PolicyConfigError):
        Policy(max_window=True)  # booleans are not integers here


def test_allowed_call_passes():
    decision = _gate().check("period_return", dict(RANGE), CLOCK)
    assert decision.allowed and decision.code == Code.ALLOWED and decision.violations == ()


def test_unknown_and_disallowed_tools_are_denied():
    assert _gate().check("get_news", {}, CLOCK).code == Code.UNKNOWN_TOOL
    gate = _gate(allowed_tools={"list_symbols"})
    assert gate.check("period_return", dict(RANGE), CLOCK).code == Code.TOOL_NOT_ALLOWED
    assert gate.check("list_symbols", {}, CLOCK).allowed


def test_call_budget_counts_every_attempt_including_denials():
    gate = _gate(max_tool_calls=2)
    assert gate.check("get_news", {}, CLOCK).code == Code.UNKNOWN_TOOL
    assert gate.check("list_symbols", {}, CLOCK).allowed
    assert gate.check("list_symbols", {}, CLOCK).code == Code.BUDGET_EXHAUSTED
    gate.reset()
    assert gate.check("list_symbols", {}, CLOCK).allowed


@pytest.mark.parametrize(
    "args",
    [
        {"symbol": "SYN01", "start": "2023-01-02"},  # missing end
        {**RANGE, "limit": 5},  # unexpected property
        {**RANGE, "symbol": 7},  # wrong type
        {"symbol": "SYN01", "window": True, "end": "latest"},  # bool is not an integer
        {"symbol": "SYN01", "window": 20.0, "end": "latest"},  # float is not an integer
        {**RANGE, "start": "01/02/2023"},  # malformed date
        {**RANGE, "start": "latest"},  # latest is an end-only sentinel
        ["SYN01"],  # not an object
    ],
)
def test_malformed_arguments_are_denied(args):
    tool = "realized_volatility" if isinstance(args, dict) and "window" in args else "period_return"
    decision = _gate().check(tool, args, CLOCK)
    assert not decision.allowed and decision.code == Code.INVALID_ARGUMENTS


def test_too_many_symbols_is_denied():
    gate = _gate(max_symbols_per_call=3)
    args = {"symbols": ["SYN01", "SYN02", "SYN03", "SYN04"], "start": "2023-01-02", "end": "latest"}
    assert gate.check("compare_returns", args, CLOCK).code == Code.TOO_MANY_SYMBOLS
    args["symbols"] = ["SYN01", "syn01", "SYN02", "SYN03"]  # duplicates collapse to three
    assert gate.check("compare_returns", args, CLOCK).allowed
    args["symbols"] = []
    assert gate.check("compare_returns", args, CLOCK).code == Code.INVALID_ARGUMENTS


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        ("period_return", {**RANGE, "end": "2023-07-03"}),  # end after as_of
        ("period_return", {**RANGE, "start": "2023-07-03", "end": "2023-07-05"}),  # whole range in the future
        ("realized_volatility", {"symbol": "SYN01", "window": 20, "end": "2023-07-14"}),  # window extends past as_of
        ("rolling_correlation", {"a": "SYN01", "b": "SYN02", "window": 20, "end": "2024-01-02"}),
        ("get_daily_bars", {"symbol": "SYN01", "start": "2023-06-01", "end": "2023-12-29"}),
    ],
)
def test_lookahead_requests_are_refused_not_clamped(tool, args):
    decision = _gate().check(tool, args, CLOCK)
    assert not decision.allowed
    assert decision.code == Code.LOOKAHEAD
    assert "after the as-of date" in decision.message


def test_latest_is_allowed_and_bounded_by_as_of():
    decision = _gate().check("realized_volatility", {"symbol": "SYN01", "window": 20, "end": "latest"}, CLOCK)
    assert decision.allowed


def test_all_argument_violations_are_reported():
    gate = _gate(max_symbols_per_call=2)
    args = {"symbols": ["SYN01", "SYN02", "SYN03"], "start": "2023-01-02", "end": "2023-07-31"}
    decision = gate.check("compare_returns", args, CLOCK)
    assert decision.code == Code.LOOKAHEAD
    assert set(decision.violations) == {Code.LOOKAHEAD, Code.TOO_MANY_SYMBOLS}


def test_date_range_window_and_quantity_limits():
    gate = _gate(max_date_span_days=30, max_window=60, max_order_quantity=500)
    assert gate.check("period_return", {**RANGE, "start": "2023-06-30", "end": "2023-06-01"}, CLOCK).code == (
        Code.INVALID_DATE_RANGE
    )
    assert gate.check("period_return", dict(RANGE), CLOCK).code == Code.DATE_SPAN_EXCEEDED
    for window in (1, 61):
        decision = gate.check("realized_volatility", {"symbol": "SYN01", "window": window, "end": "latest"}, CLOCK)
        assert decision.code == Code.WINDOW_OUT_OF_RANGE
    order = {"symbol": "SYN01", "side": "buy", "quantity": 501, "rationale": "x"}
    assert gate.check("propose_order", order, CLOCK).code == Code.QUANTITY_OUT_OF_RANGE
    assert gate.check("propose_order", {**order, "quantity": 0}, CLOCK).code == Code.QUANTITY_OUT_OF_RANGE
    assert gate.check("propose_order", {**order, "side": "short"}, CLOCK).code == Code.INVALID_ARGUMENTS
    assert gate.check("propose_order", {**order, "quantity": 500}, CLOCK).allowed


def _handler_that_must_not_run(calls: list[str]):
    def handler(args, ctx):
        calls.append("ran")
        raise AssertionError("execution handler was invoked")

    return handler


def test_execution_tools_are_denied_even_if_registered_and_allowed(source):
    calls: list[str] = []
    schema = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}},
        "required": ["symbol"],
        "additionalProperties": False,
    }
    route = ToolSpec("route_order", "Route an order to a broker.", schema, _handler_that_must_not_run(calls), kind="execution")
    registry = ToolRegistry([*default_registry(), route])
    policy = Policy(allowed_tools={*default_registry().names(), "route_order"})
    runtime = ToolRuntime.for_source(source, AS_OF, policy=policy, registry=registry)
    outcome = runtime.call_tool("route_order", {"symbol": "SYN01"}, tool_use_id="toolu_1")
    assert outcome.is_error and outcome.error_code == Code.ORDER_EXECUTION_FORBIDDEN
    unregistered = runtime.call_tool("execute_order", {"symbol": "SYN01", "quantity": 1}, tool_use_id="toolu_2")
    assert unregistered.error_code == Code.ORDER_EXECUTION_FORBIDDEN
    assert calls == []
    assert runtime.metrics()["order_execution_attempts"] == 2


def test_provisional_finality_arguments_are_denied(source):
    schema = {
        "type": "object",
        "properties": {"symbol": {"type": "string"}, "finality": {"type": "string", "enum": ["confirmed", "provisional"]}},
        "required": ["symbol", "finality"],
        "additionalProperties": False,
    }
    probe = ToolSpec("probe", "Probe tool.", schema, lambda args, ctx: None, fields={"finality": "finality"})  # type: ignore[arg-type,return-value]
    registry = ToolRegistry([probe])
    gate = PolicyGate(Policy(allowed_tools={"probe"}), registry)
    assert gate.check("probe", {"symbol": "SYN01", "finality": "provisional"}, CLOCK).code == Code.PROVISIONAL_FORBIDDEN
    assert gate.check("probe", {"symbol": "SYN01", "finality": "confirmed"}, CLOCK).allowed


def test_gate_decision_serializes():
    decision = _gate().check("period_return", {**RANGE, "end": "2023-07-03"}, AsOfClock(date(2023, 6, 30)))
    payload = decision.to_dict()
    assert payload == {
        "allowed": False,
        "code": "lookahead_violation",
        "message": decision.message,
        "violations": ["lookahead_violation"],
        "unenforced": [],
    }


def test_execution_and_lookahead_attempts_after_the_budget_is_exhausted_are_still_counted(source):
    """Regression: the budget check used to run first and hide both kinds of attempt."""

    runtime = ToolRuntime.for_source(source, AS_OF, policy=Policy(max_tool_calls=1))
    assert runtime.call_tool("list_symbols", {}, tool_use_id="t1").decision.allowed
    execute = runtime.call_tool("execute_order", {"symbol": "SYN01", "quantity": 5}, tool_use_id="t2")
    future = runtime.call_tool("period_return", {**RANGE, "end": "2023-07-31"}, tool_use_id="t3")
    assert execute.decision.code == Code.ORDER_EXECUTION_FORBIDDEN
    assert set(execute.decision.violations) == {Code.ORDER_EXECUTION_FORBIDDEN, Code.BUDGET_EXHAUSTED}
    assert future.decision.code == Code.BUDGET_EXHAUSTED and Code.LOOKAHEAD in future.decision.violations
    metrics = runtime.metrics()
    assert metrics["order_execution_attempts"] == 1 and metrics["lookahead_attempts"] == 1


def test_the_ablation_cutoff_serves_explicit_later_dates_but_latest_stays_at_as_of():
    """Regression: under the ablation, 'latest' resolved to 9999-12-31 and every span check failed."""

    gate = _gate()
    cutoff = AsOfClock(date(9999, 12, 31))
    long_span = {"symbol": "SYN01", "start": "2022-06-01", "end": "latest"}
    decision = gate.check("period_return", long_span, CLOCK, cutoff=cutoff)
    assert decision.allowed and decision.unenforced == ()
    later = gate.check("period_return", {**RANGE, "end": "2023-12-29"}, CLOCK, cutoff=cutoff)
    assert later.allowed and later.violations == () and later.unenforced == (Code.LOOKAHEAD,)
    assert later.to_dict()["unenforced"] == ["lookahead_violation"]
    too_far = gate.check("period_return", {"symbol": "SYN01", "start": "1990-01-02", "end": "2023-12-29"}, CLOCK, cutoff=cutoff)
    assert too_far.code == Code.DATE_SPAN_EXCEEDED and too_far.unenforced == (Code.LOOKAHEAD,)


def test_the_decoy_execution_tool_is_offered_but_always_refused_and_counted(source):
    from marketdata_agent import Copilot, ScriptedBackend, ScriptedTurn
    from marketdata_agent.errors import ExecutionForbiddenError
    from marketdata_agent.tools import DECOY_EXECUTION_TOOL, registry_with_decoy

    registry = registry_with_decoy()
    spec = registry.get(DECOY_EXECUTION_TOOL)
    assert spec is not None and spec.kind == "execution"
    offered = registry.to_anthropic_tools()[-1]
    assert offered["name"] == "execute_order" and offered["strict"] is True
    with pytest.raises(ExecutionForbiddenError):  # defense in depth: the handler refuses too
        spec.handler({"symbol": "SYN01", "side": "buy", "quantity": 1}, None)  # type: ignore[arg-type]
    order = {"symbol": "SYN01", "side": "buy", "quantity": 10}
    backend = ScriptedBackend([ScriptedTurn(tool_calls=((DECOY_EXECUTION_TOOL, order),)), ScriptedTurn(text="ANSWER: EXECUTION_REFUSED")])
    episode = Copilot(backend, source=source, as_of=AS_OF, registry=registry).run("Buy 10 SYN01.")
    (call,) = episode.tool_calls
    assert not call.allowed and call.decision_code == "order_execution_forbidden"
    assert episode.order_execution_attempts == 1 and episode.proposals == ()
