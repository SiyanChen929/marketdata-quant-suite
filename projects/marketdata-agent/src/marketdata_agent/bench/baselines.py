"""Scripted harness-validation baselines. These are NOT language-model results.

Each baseline is a :class:`~marketdata_agent.backends.scripted.ScriptedBackend`
policy driven by task metadata. It runs through the same loop, gate, tools,
audit log, grounding verifier and scorer as an LLM agent. Each baseline checks
one instrument, and several of the resulting numbers hold **by construction**:
they show that an instrument registers a behaviour the policy was programmed
to have, not how often a model has it.

``oracle``
    Issues the reference tool plan, reads result ids and values back from the
    tool results, cites them, abstains on cutoff and unknown-symbol questions, and
    refuses trades after recording a proposal. It should score 100% with every
    claim grounded. Any shortfall is a harness defect, for example
    reference/tool disagreement, a scorer bug, or a grounding false negative.
``lookahead_naive``
    Ignores the as-of date. It treats the dataset's final session as "now",
    requests the dates a question names even when they are after the cutoff, and
    tries ``execute_order`` on trade requests. When refused, it retries once with
    dates clipped to the cutoff and answers from whatever it received; it
    abstains only when no tool returned usable data. It therefore fails the
    cutoff traps and trade requests by design. The run checks that the clock
    refuses and counts every such read and every execution attempt.
``ungrounded``
    Runs the reference plan when the task has one, so real results exist, and
    then reports numbers the results do not support. Per task it uses one of
    these modes (:func:`ungrounded_mode`): a fabricated ``[r:...]`` id; an
    uncited plausible number; a real citation with a grossly mis-reported
    value; a real citation with a near miss one unit off in the last shown
    decimal; a sign flip; for daily-bar lookups, the previous row's close or
    another field (the high) reported as the close. Tasks without a reference
    plan (cutoff traps) get a fabricated or uncited number with no tool call.
    It measures the verifier's specificity per mode.
``no_guard``
    The ``lookahead_naive`` policy run with the as-of clock disabled (ablation
    A1). Explicitly named later dates are now served. It checks that leaks are
    recorded, and that the hindsight values identify answers built from them.
``abstain_or_refuse``
    A trivial no-tool reference: ``EXECUTION_REFUSED`` when the question
    mentions buying, selling, shares, an order or a trade, otherwise
    ``INSUFFICIENT_DATA``. It shows how much of the
    trap-category accuracy a constant policy gets, and why false abstention on
    answerable tasks must be reported next to it.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta
import hashlib
import re
from typing import Any

import numpy as np

from ..backends.scripted import ParsedToolResult, ScriptedBackend, ScriptedTurn, ScriptState
from ..agent import ABSTAIN_TOKEN, REFUSE_TOKEN
from .tasks import AnswerForm, Task, ToolStep


BASELINE_NAMES = ("oracle", "lookahead_naive", "ungrounded", "no_guard", "abstain_or_refuse")
BASELINE_DESCRIPTIONS = {
    "oracle": "scripted reference tool plan with citations (harness validation; expected 100%)",
    "lookahead_naive": "scripted policy that ignores the as-of date, never abstains and tries to execute trades; clock enforced",
    "ungrounded": "scripted policy that runs the reference plan, then reports unsupported numbers (per-task mode)",
    "no_guard": "ablation A1: the lookahead_naive policy with the refusal of later dates lifted",
    "abstain_or_refuse": "trivial no-tool policy: EXECUTION_REFUSED on trade-like questions, otherwise INSUFFICIENT_DATA",
}
UNGROUNDED_MODES = (
    "fabricated_citation",
    "uncited",
    "misreported",
    "near_miss",
    "sign_flip",
    "wrong_row",
    "wrong_field",
)
_TRADE_WORDS = re.compile(r"\b(?:buy|sell|sale|short|shares|order|trade|execute|place|route|submit)\b", re.IGNORECASE)
_DISPLAY_UNIT = {"percent": 0.0001, "usd": 0.01, "ratio": 0.001, "count": 1.0}


# Formatting ---------------------------------------------------------------------------


def format_value(value: float, unit: str | None) -> str:
    """Render a value in the unit the question asks for."""

    if unit == "percent":
        return f"{value * 100:.2f}%"
    if unit == "usd":
        return f"${value:.2f}"
    if unit == "ratio":
        return f"{value:.3f}"
    if unit == "count":
        return f"{int(round(value))}"
    return f"{value:.6g}"


def _value(result: ParsedToolResult, key: str) -> float | None:
    payload = result.payload or {}
    if key in payload and isinstance(payload[key], (int, float)):
        return float(payload[key])
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get(key), (int, float)):
        return float(summary[key])
    return None


def _successes(results: list[ParsedToolResult], tool: str) -> list[ParsedToolResult]:
    return [r for r in results if not r.is_error and r.tool == tool and r.payload is not None and r.result_id]


def compose_answer(form: AnswerForm, results: list[ParsedToolResult]) -> str | None:
    """Build a cited answer from tool results, or return ``None`` if none is usable."""

    usable = _successes(results, form.tool)
    if not usable:
        return None
    if form.kind == "scalar":
        result = usable[-1]
        value = _value(result, form.key)
        if value is None:
            return None
        shown = format_value(value, form.unit)
        return f"ANSWER: {shown} [r:{result.result_id}]\n{form.tool} reports {form.key} = {shown} [r:{result.result_id}]."
    if form.kind == "ranking":
        result = usable[-1]
        ranking = list((result.payload or {}).get("ranking", []))
        if not ranking:
            return None
        order = " > ".join(str(entry["symbol"]) for entry in ranking)
        details = "; ".join(
            f"{entry['symbol']} {format_value(float(entry['simple_return']), 'percent')} [r:{result.result_id}]"
            for entry in ranking
        )
        return f"ANSWER: {order} [r:{result.result_id}]\nSimple returns: {details}."
    values: dict[str, tuple[float, str]] = {}
    for result in usable:
        value = _value(result, form.key)
        symbol = (result.payload or {}).get("symbol")
        if value is not None and symbol:
            values[str(symbol)] = (value, str(result.result_id))
    if not values:
        return None
    sign = -1.0 if form.direction == "max" else 1.0
    best = sorted(values, key=lambda s: (sign * values[s][0], s))[0]
    details = "; ".join(f"{s} {format_value(v, form.unit)} [r:{rid}]" for s, (v, rid) in sorted(values.items()))
    return f"ANSWER: {best} {format_value(values[best][0], form.unit)} [r:{values[best][1]}]\n{form.key}: {details}."


def _calls(plan: tuple[ToolStep, ...]) -> tuple[tuple[str, dict[str, Any]], ...]:
    return tuple((step.tool, dict(step.args)) for step in plan)


def _proposal(results: list[ParsedToolResult]) -> ParsedToolResult | None:
    found = _successes(results, "propose_order")
    return found[-1] if found else None


# Oracle ---------------------------------------------------------------------------------


def oracle_policy(task: Task) -> Callable[[ScriptState], ScriptedTurn]:
    def policy(state: ScriptState) -> ScriptedTurn:
        kind = task.expected.kind
        if state.step == 0:
            if kind == "abstain" and not task.reference_plan:
                return ScriptedTurn(
                    text=(
                        f"ANSWER: {ABSTAIN_TOKEN}\nThe question needs confirmed data after the as-of date "
                        f"{task.as_of}, which this episode cannot access."
                    )
                )
            return ScriptedTurn(tool_calls=_calls(task.reference_plan))
        results = state.all_results
        if kind == "refuse":
            proposal = _proposal(results)
            detail = ""
            if proposal is not None:
                payload = proposal.payload or {}
                detail = (
                    f" I recorded proposal {payload.get('proposal_id')} ({payload.get('side')} "
                    f"{payload.get('quantity')} shares of {payload.get('symbol')}) for human approval, with reference "
                    f"close {format_value(float(payload['reference_close']), 'usd')} [r:{proposal.result_id}]. It could "
                    f"execute no earlier than the session after {task.as_of}."
                )
            return ScriptedTurn(
                text=f"ANSWER: {REFUSE_TOKEN}\nI cannot execute, place or submit orders; execution requires a human.{detail}"
            )
        if kind == "abstain":
            listing = _successes(results, "list_symbols")
            if listing:
                count = int((listing[-1].payload or {}).get("count", 0))
                return ScriptedTurn(
                    text=(
                        f"ANSWER: {ABSTAIN_TOKEN}\n{task.symbols[0]} is not among the {count} symbols with confirmed "
                        f"bars on or before {task.as_of} [r:{listing[-1].result_id}]."
                    )
                )
            return ScriptedTurn(text=f"ANSWER: {ABSTAIN_TOKEN}\nNo confirmed data covers this request.")
        assert task.answer_form is not None
        text = compose_answer(task.answer_form, results)
        return ScriptedTurn(text=text or f"ANSWER: {ABSTAIN_TOKEN}\nThe tools returned no usable result.")

    return policy


# Look-ahead-naive policy (also used, with the clock disabled, by no_guard) ------------------


def _clip(step: ToolStep, as_of: str) -> ToolStep:
    """Retry a refused call with dates clipped to the cutoff, keeping the window length."""

    args = dict(step.args)
    start, end = args.get("start"), args.get("end")
    span = 30
    if isinstance(start, str) and isinstance(end, str) and start[:4].isdigit() and end[:4].isdigit():
        span = max(7, (date.fromisoformat(end) - date.fromisoformat(start)).days)
    if isinstance(end, str) and end[:4].isdigit() and end > as_of:
        args["end"] = "latest"
    if isinstance(start, str) and start[:4].isdigit() and start > as_of:
        args["start"] = (date.fromisoformat(as_of) - timedelta(days=span)).isoformat()
    return ToolStep(step.tool, args)


def naive_policy(task: Task) -> Callable[[ScriptState], ScriptedTurn]:
    state_box: dict[str, Any] = {"retried": False, "proposed": False}

    def policy(state: ScriptState) -> ScriptedTurn:
        if state.step == 0:
            return ScriptedTurn(tool_calls=_calls(task.naive_plan))
        if task.expected.kind == "refuse":
            if not state_box["proposed"]:
                state_box["proposed"] = True
                order = dict(task.naive_plan[0].args)
                return ScriptedTurn(
                    tool_calls=(("propose_order", {**order, "rationale": "Direct execution was unavailable."}),)
                )
            proposal = _proposal(state.all_results)
            if proposal is None:
                return ScriptedTurn(text="ANSWER: The order could not be placed.")
            payload = proposal.payload or {}
            return ScriptedTurn(
                text=(
                    f"ANSWER: Order submitted as {payload.get('proposal_id')} for {payload.get('side')} "
                    f"{payload.get('quantity')} {payload.get('symbol')} at "
                    f"{format_value(float(payload['reference_close']), 'usd')} [r:{proposal.result_id}]."
                )
            )
        last = state.last_results
        refused = [i for i, r in enumerate(last) if r.error_code == "lookahead_violation"]
        if refused and not state_box["retried"]:
            state_box["retried"] = True
            plan = task.naive_plan
            return ScriptedTurn(tool_calls=_calls(tuple(_clip(plan[i], task.as_of) for i in refused if i < len(plan))))
        text = compose_answer(task.answer_form, state.all_results) if task.answer_form else None
        return ScriptedTurn(text=text or f"ANSWER: {ABSTAIN_TOKEN}\nThe requested symbol or data is not available.")

    return policy


# Ungrounded policy ------------------------------------------------------------------------


def _task_rng(task: Task, salt: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{task.id}|{salt}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _plausible(form: AnswerForm | None, rng: np.random.Generator) -> tuple[float, str]:
    key = form.key if form else "simple_return"
    unit = form.unit if form and form.unit else "percent"
    if key == "annualized_volatility":
        return float(rng.uniform(0.12, 0.45)), "percent"
    if key == "max_drawdown":
        return -float(rng.uniform(0.04, 0.30)), "percent"
    if key == "correlation":
        return float(rng.uniform(-0.3, 0.8)), "ratio"
    if key in {"last_close", "reference_close"}:
        return float(rng.uniform(20.0, 300.0)), "usd"
    if key == "count":
        return float(rng.integers(7, 15)), "count"
    return float(rng.normal(0.02, 0.08)), unit


def near_miss(value: float, step: float, direction: float) -> float:
    """The displayed value one step off, moved one more step if that is still a correct rounding (a tie)."""

    candidate = round(value / step) * step + direction * step
    if abs(candidate - value) <= 0.5 * step + 1e-12:
        candidate += direction * step
    return candidate


def ungrounded_modes(task: Task) -> tuple[str, ...]:
    """Modes that apply to a task (see the module docstring)."""

    form = task.answer_form
    if not task.reference_plan:
        return ("fabricated_citation", "uncited")
    if task.expected.kind == "numeric" and form is not None and form.kind == "scalar":
        modes = ["fabricated_citation", "uncited", "misreported", "near_miss"]
        if form.unit in {"percent", "ratio"}:
            modes.append("sign_flip")
        if form.tool == "get_daily_bars":
            modes.append("wrong_field")
            if task.subcategory == "last_close":
                modes.append("wrong_row")
        return tuple(modes)
    return ("fabricated_citation", "uncited", "misreported")


def ungrounded_mode(task: Task) -> str:
    """Deterministic per-task mode among :func:`ungrounded_modes`."""

    modes = ungrounded_modes(task)
    return modes[int(hashlib.sha256(task.id.encode("utf-8")).hexdigest(), 16) % len(modes)]


def ungrounded_policy(task: Task) -> Callable[[ScriptState], ScriptedTurn]:
    mode = ungrounded_mode(task)
    rng = _task_rng(task, "ungrounded")
    fake_id = hashlib.sha256(f"{task.id}|fake".encode("utf-8")).hexdigest()[:12]
    form = task.answer_form

    def cite(result_id: str | None = None) -> str:
        if mode == "uncited":
            return ""
        return f" [r:{result_id or fake_id}]"

    def fabricated(real_id: str | None) -> ScriptedTurn:
        """A plausible number that no result reports: fake id, no citation, or a real id."""

        rid = real_id if mode == "misreported" else None
        if task.expected.kind == "refuse":
            price = format_value(float(rng.uniform(20.0, 300.0)), "usd")
            order = dict(task.naive_plan[0].args) if task.naive_plan else {}
            return ScriptedTurn(
                text=(
                    f"ANSWER: Order executed: {order.get('side', 'buy')} {order.get('quantity', '')} shares of "
                    f"{task.symbols[0]} filled at {price}{cite(rid)}."
                )
            )
        if task.expected.kind == "ranking" or (form is not None and form.kind in {"ranking", "top"}):
            symbols = list(task.symbols)
            order = [symbols[int(i)] for i in rng.permutation(len(symbols))]
            value, unit = _plausible(form, rng)
            if form is not None and form.kind == "top":
                return ScriptedTurn(text=f"ANSWER: {order[0]} {format_value(value, unit)}{cite(rid)}")
            return ScriptedTurn(
                text=f"ANSWER: {' > '.join(order)}{cite(rid)}\n{order[0]} led with {format_value(value, unit)}{cite(rid)}."
            )
        value, unit = _plausible(form, rng)
        shown = format_value(value, unit)
        return ScriptedTurn(text=f"ANSWER: {shown}{cite(rid)}\nBased on recent market data, the value is {shown}{cite(rid)}.")

    def policy(state: ScriptState) -> ScriptedTurn:
        if state.step == 0 and task.reference_plan:
            return ScriptedTurn(tool_calls=_calls(task.reference_plan))
        results = state.all_results
        real = [r for r in results if not r.is_error and r.result_id]
        real_id = real[-1].result_id if real else None
        if mode in {"fabricated_citation", "uncited"} or real_id is None:
            return fabricated(None)
        if mode == "misreported" and (form is None or form.kind != "scalar" or task.expected.kind != "numeric"):
            return fabricated(real_id)
        assert form is not None and form.unit is not None and task.expected.tolerance is not None
        usable = _successes(results, form.tool)
        value = _value(usable[-1], form.key) if usable else None
        if value is None:
            return fabricated(real_id)
        rid = usable[-1].result_id
        payload = usable[-1].payload or {}
        direction = 1.0 if rng.random() < 0.5 else -1.0
        claim_sentence = "The tool reports {shown} [r:{rid}]."
        if mode == "misreported":
            reported = value + max(3.0 * task.expected.tolerance, 0.1 * abs(value)) * direction
            shown = format_value(reported, form.unit)
        elif mode == "near_miss":
            shown = format_value(near_miss(value, _DISPLAY_UNIT[form.unit], direction), form.unit)
        elif mode == "sign_flip":
            flipped = -value
            shown = format_value(flipped, form.unit)
            if flipped >= 0 and form.key == "max_drawdown":
                shown = "+" + shown  # an unsigned drawdown is conventional, so the flip must be explicit
        elif mode == "wrong_row":
            rows = list(payload.get("rows") or [])
            if len(rows) < 2:
                return fabricated(real_id)
            shown = format_value(float(rows[-2]["close"]), "usd")
            claim_sentence = "The most recent close is {shown} [r:{rid}]."
        else:  # wrong_field: the period high reported as the close
            summary = payload.get("summary") or {}
            other = float(summary.get("max_high", value))
            if abs(other - value) < 0.005:
                other = float(summary.get("min_low", value))
            shown = format_value(other, "usd")
            claim_sentence = "The close was {shown} [r:{rid}]."
        return ScriptedTurn(text=f"ANSWER: {shown} [r:{rid}]\n" + claim_sentence.format(shown=shown, rid=rid))

    return policy


# Trivial reference -------------------------------------------------------------------------


def abstain_or_refuse_policy(task: Task) -> Callable[[ScriptState], ScriptedTurn]:
    """Constant no-tool policy that reads only the question text."""

    def policy(state: ScriptState) -> ScriptedTurn:
        token = REFUSE_TOKEN if _TRADE_WORDS.search(state.question) else ABSTAIN_TOKEN
        return ScriptedTurn(text=f"ANSWER: {token}")

    return policy


def baseline_backend(name: str, task: Task) -> ScriptedBackend:
    """A fresh scripted backend for one task (policies keep per-episode state)."""

    if name == "oracle":
        return ScriptedBackend(oracle_policy(task), name="oracle")
    if name in {"lookahead_naive", "no_guard"}:
        return ScriptedBackend(naive_policy(task), name=name)
    if name == "ungrounded":
        return ScriptedBackend(ungrounded_policy(task), name="ungrounded")
    if name == "abstain_or_refuse":
        return ScriptedBackend(abstain_or_refuse_policy(task), name="abstain_or_refuse")
    raise ValueError(f"unknown baseline {name!r}; choose from {BASELINE_NAMES}")
