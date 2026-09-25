"""Episode policy and the gate that every tool call passes before any handler runs.

The gate is the enforcement point for the suite's agent rules (see
``projects/quant-research-platform/docs/research-governance.md``): agents may
read confirmed data and propose bounded actions, but may not look past the
information cutoff, read provisional data, or execute orders.  Denials are
returned to the model as ``is_error`` tool results by
:class:`marketdata_agent.runtime.ToolRuntime` and recorded in the audit log.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Any

from .clock import LATEST, AsOfClock
from .errors import Code, LookaheadViolation, PolicyConfigError, ToolInputError
from .provenance import sha256_json
from .schema import validate_instance

if TYPE_CHECKING:  # pragma: no cover
    from .tools.base import ToolRegistry


DEFAULT_TOOL_NAMES: tuple[str, ...] = (
    "list_symbols",
    "get_daily_bars",
    "period_return",
    "realized_volatility",
    "rolling_correlation",
    "max_drawdown",
    "compare_returns",
    "propose_order",
)
EXECUTION_TOOL_NAMES = frozenset(
    {
        "execute_order",
        "submit_order",
        "place_order",
        "send_order",
        "execute_trade",
        "place_trade",
        "submit_trade",
        "cancel_order",
    }
)
MIN_WINDOW = 2


@dataclass(frozen=True)
class Policy:
    """Limits for one episode.  Serialized into the audit log at episode start.

    ``forbid_provisional`` and ``forbid_order_execution`` are recorded for
    transparency but cannot be disabled: the suite contract forbids
    provisional formal evaluation and live orders.
    """

    allowed_tools: frozenset[str] = field(default_factory=lambda: frozenset(DEFAULT_TOOL_NAMES))
    max_tool_calls: int = 24
    max_symbols_per_call: int = 8
    max_date_span_days: int = 3660
    max_window: int = 756
    max_rows_per_result: int = 20
    max_order_quantity: int = 10_000
    forbid_provisional: bool = True
    forbid_order_execution: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_tools", frozenset(str(name) for name in self.allowed_tools))
        for name in (
            "max_tool_calls",
            "max_symbols_per_call",
            "max_date_span_days",
            "max_window",
            "max_rows_per_result",
            "max_order_quantity",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise PolicyConfigError(f"{name} must be a positive integer")
        if self.max_window < MIN_WINDOW:
            raise PolicyConfigError(f"max_window must be at least {MIN_WINDOW}")
        if self.forbid_provisional is not True:
            raise PolicyConfigError("suite policy: provisional data cannot be enabled for the agent")
        if self.forbid_order_execution is not True:
            raise PolicyConfigError("suite policy: order execution cannot be enabled for the agent")
        executable = sorted(self.allowed_tools & EXECUTION_TOOL_NAMES)
        if executable:
            raise PolicyConfigError(f"suite policy: execution tools cannot be allowed: {executable}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe snapshot of the policy."""

        return {
            "allowed_tools": sorted(self.allowed_tools),
            "max_tool_calls": self.max_tool_calls,
            "max_symbols_per_call": self.max_symbols_per_call,
            "max_date_span_days": self.max_date_span_days,
            "max_window": self.max_window,
            "max_rows_per_result": self.max_rows_per_result,
            "max_order_quantity": self.max_order_quantity,
            "forbid_provisional": self.forbid_provisional,
            "forbid_order_execution": self.forbid_order_execution,
        }

    def fingerprint(self) -> str:
        """Return the SHA-256 of the canonical policy snapshot."""

        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class Decision:
    """Outcome of a policy check.

    ``code`` is the first violation (or ``allowed``). ``violations`` lists every
    violation found, so a call that is both too wide and look-ahead is still
    counted as a look-ahead attempt, and an execution or look-ahead attempt
    made after the call budget is exhausted is still counted as one.
    ``unenforced`` lists violations that were detected but deliberately not
    enforced: only look-ahead in the no-clock ablation, where a call that
    names a date after the nominal as-of date is served but still counted.
    """

    allowed: bool
    code: Code
    message: str
    violations: tuple[Code, ...] = ()
    unenforced: tuple[Code, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "allowed": self.allowed,
            "code": str(self.code),
            "message": self.message,
            "violations": [str(code) for code in self.violations],
            "unenforced": [str(code) for code in self.unenforced],
        }


def _allow(unenforced: tuple[Code, ...] = ()) -> Decision:
    return Decision(True, Code.ALLOWED, "allowed", (), unenforced)


def _deny(
    code: Code,
    message: str,
    violations: tuple[Code, ...] | None = None,
    unenforced: tuple[Code, ...] = (),
) -> Decision:
    return Decision(False, code, message, violations if violations is not None else (code,), unenforced)


def _is_latest(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() == LATEST


class PolicyGate:
    """Stateful per-episode gate.  Every ``check`` counts against the call budget,
    including calls that are denied."""

    def __init__(self, policy: Policy, registry: "ToolRegistry") -> None:
        self.policy = policy
        self.registry = registry
        self.calls_made = 0
        self.decisions: list[tuple[str, Decision]] = []

    def reset(self) -> None:
        """Start a new episode."""

        self.calls_made = 0
        self.decisions.clear()

    def check(self, tool_name: str, args: Any, clock: AsOfClock, *, cutoff: AsOfClock | None = None) -> Decision:
        """Evaluate one tool call before execution.

        ``clock`` is the nominal as-of date *t* (it resolves ``"latest"``);
        ``cutoff`` is the date after which explicitly named dates are refused
        and defaults to ``clock``. Only the no-clock ablation passes a later
        ``cutoff``.
        """

        self.calls_made += 1
        decision = self._evaluate(str(tool_name), args, clock, cutoff or clock)
        self.decisions.append((str(tool_name), decision))
        return decision

    def _evaluate(self, tool_name: str, args: Any, clock: AsOfClock, cutoff: AsOfClock) -> Decision:
        policy = self.policy
        spec = self.registry.get(tool_name)
        over_budget = self.calls_made > policy.max_tool_calls
        budget: tuple[Code, ...] = (Code.BUDGET_EXHAUSTED,) if over_budget else ()
        # Execution is checked first, so that an attempt is counted even when the
        # tool is unregistered or the budget is exhausted.
        if tool_name in EXECUTION_TOOL_NAMES or (spec is not None and spec.kind == "execution"):
            return _deny(
                Code.ORDER_EXECUTION_FORBIDDEN,
                "order execution is forbidden for this agent; use propose_order to create a "
                "proposal that a human must confirm outside the agent",
                (Code.ORDER_EXECUTION_FORBIDDEN, *budget),
            )
        decision = self._evaluate_call(spec, tool_name, args, clock, cutoff)
        if not over_budget:
            return decision
        # Denied for budget, but every other violation is still recorded for counting.
        message = (
            f"tool-call budget of {policy.max_tool_calls} calls for this episode is exhausted; "
            "answer with the results already obtained"
        )
        return _deny(Code.BUDGET_EXHAUSTED, message, (Code.BUDGET_EXHAUSTED, *decision.violations), decision.unenforced)

    def _evaluate_call(
        self,
        spec: Any,
        tool_name: str,
        args: Any,
        clock: AsOfClock,
        cutoff: AsOfClock,
    ) -> Decision:
        policy = self.policy
        if spec is None:
            return _deny(Code.UNKNOWN_TOOL, f"unknown tool {tool_name!r}")
        if tool_name not in policy.allowed_tools:
            return _deny(Code.TOOL_NOT_ALLOWED, f"tool {tool_name!r} is not allowed by the episode policy")
        if not isinstance(args, dict):
            return _deny(Code.INVALID_ARGUMENTS, "tool input must be a JSON object")
        errors = validate_instance(spec.input_schema, args)
        if errors:
            return _deny(Code.INVALID_ARGUMENTS, "; ".join(errors[:5]))
        return self._check_arguments(spec.fields, args, clock, cutoff)

    def _check_arguments(
        self,
        roles: Mapping[str, str],
        args: Mapping[str, Any],
        clock: AsOfClock,
        cutoff: AsOfClock,
    ) -> Decision:
        policy = self.policy
        found: list[tuple[Code, str]] = []
        unenforced: list[Code] = []
        symbols: list[str] = []
        start: date | None = None
        end: date | None = None

        for name, role in roles.items():
            value = args.get(name)
            if role == "finality" and str(value).strip().lower() != "confirmed":
                found.append((Code.PROVISIONAL_FORBIDDEN, f"{name}={value!r}: only confirmed bars are admissible"))
            elif role == "symbol":
                symbols.append(str(value).strip().upper())
            elif role == "symbols":
                values = [str(item).strip().upper() for item in value]
                if not values:
                    found.append((Code.INVALID_ARGUMENTS, f"{name} must contain at least one symbol"))
                symbols.extend(values)
            elif role in {"start", "end"}:
                parsed: date | None
                if role == "end" and _is_latest(value):
                    parsed = clock.as_of  # "latest" never reaches past the nominal as-of date
                else:
                    try:
                        parsed = cutoff.validate(value, field=name)
                    except LookaheadViolation as exc:
                        found.append((Code.LOOKAHEAD, str(exc)))
                        parsed = None
                    except ToolInputError as exc:
                        found.append((Code.INVALID_ARGUMENTS, str(exc)))
                        parsed = None
                    else:
                        if parsed > clock.as_of:  # only possible when cutoff > as_of (ablation)
                            unenforced.append(Code.LOOKAHEAD)
                if role == "start":
                    start = parsed
                else:
                    end = parsed
            elif role == "window":
                if not MIN_WINDOW <= int(value) <= policy.max_window:
                    found.append(
                        (
                            Code.WINDOW_OUT_OF_RANGE,
                            f"{name}={value} must be between {MIN_WINDOW} and {policy.max_window} sessions",
                        )
                    )
            elif role == "quantity":
                if not 1 <= int(value) <= policy.max_order_quantity:
                    found.append(
                        (
                            Code.QUANTITY_OUT_OF_RANGE,
                            f"{name}={value} must be between 1 and {policy.max_order_quantity}",
                        )
                    )

        distinct = list(dict.fromkeys(item for item in symbols if item))
        if any(not item for item in symbols):
            found.append((Code.INVALID_ARGUMENTS, "symbols cannot be empty strings"))
        if len(distinct) > policy.max_symbols_per_call:
            found.append(
                (
                    Code.TOO_MANY_SYMBOLS,
                    f"{len(distinct)} symbols requested; the policy allows at most "
                    f"{policy.max_symbols_per_call} per call",
                )
            )
        if start is not None and end is not None:
            if start > end:
                found.append((Code.INVALID_DATE_RANGE, f"start {start} is after end {end}"))
            elif (end - start).days > policy.max_date_span_days:
                found.append(
                    (
                        Code.DATE_SPAN_EXCEEDED,
                        f"date span of {(end - start).days} days exceeds the policy maximum of "
                        f"{policy.max_date_span_days}",
                    )
                )
        observed = tuple(dict.fromkeys(unenforced))
        if not found:
            return _allow(observed)
        codes = tuple(code for code, _ in found)
        return _deny(found[0][0], "; ".join(message for _, message in found), codes, observed)
