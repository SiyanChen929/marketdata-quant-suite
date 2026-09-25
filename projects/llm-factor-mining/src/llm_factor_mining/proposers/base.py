"""Proposer protocol and the formation-only context a proposer may see.

Information barrier.  A proposer receives a :class:`ProposalContext` and
nothing else.  The context types are closed (frozen, slotted dataclasses that
reject foreign element types) and carry only:

* the grammar card and the number of expressions requested;
* :class:`EvaluatedFeedback` - canonical expression text plus *formation
  window* statistics (fields are literally named ``formation_*``); the only
  factory from a metrics report refuses any window not named ``formation``;
* :class:`RejectedFeedback` - raw text plus static validation error codes.

There is no field that could hold validation or test metrics, dates, or asset
identifiers, and the search computes feedback on a panel truncated at the end
of the formation window.  ``tests/test_proposer_isolation.py`` checks both
the types and the end-to-end invariance of every context to post-formation
data.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import numbers
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from ..dsl.validate import ERROR_CODES
from ..evaluate.metrics import FactorReport


FORMATION_WINDOW = "formation"
# EVAL_ERROR: numerical failure during evaluation; DEGENERATE: too few formation
# dates with a defined IC for a trustworthy p-value (see SearchConfig.min_ic_dates).
FEEDBACK_ERROR_CODES: frozenset[str] = frozenset(ERROR_CODES) | {"EVAL_ERROR", "DEGENERATE"}
MAX_FEEDBACK_TEXT = 400


class LeakageError(ValueError):
    """Something other than formation-window information was offered to a proposer."""


class ProposerError(RuntimeError):
    """A recoverable proposer failure (e.g. an LLM refusal); the search logs it and continues."""


def _finite_or_nan(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    number = float(value)
    return number if math.isfinite(number) else float("nan")


@dataclass(frozen=True, slots=True)
class EvaluatedFeedback:
    """Formation-window statistics of one evaluated expression."""

    expression: str
    formation_ic: float
    formation_icir: float
    formation_tstat: float
    formation_turnover: float
    complexity: float

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str) or not self.expression:
            raise TypeError("expression must be a non-empty string")
        if len(self.expression) > MAX_FEEDBACK_TEXT * 5:
            raise ValueError("expression text too long for feedback")
        for name in ("formation_ic", "formation_icir", "formation_tstat", "formation_turnover", "complexity"):
            object.__setattr__(self, name, _finite_or_nan(getattr(self, name), name))

    @classmethod
    def from_report(cls, report: FactorReport, *, complexity: float) -> "EvaluatedFeedback":
        """Build feedback from a formation-window :class:`FactorReport` only."""

        if not isinstance(report, FactorReport):
            raise TypeError("from_report expects a FactorReport")
        if report.window != FORMATION_WINDOW:
            raise LeakageError(
                f"proposers may only see formation-window metrics, got window {report.window!r}"
            )
        if report.expression is None:
            raise ValueError("report has no expression")
        return cls(
            expression=report.expression,
            formation_ic=report.ic_mean,
            formation_icir=report.icir,
            formation_tstat=report.ic_tstat_nw,
            formation_turnover=report.top_quantile_turnover,
            complexity=complexity,
        )


@dataclass(frozen=True, slots=True)
class RejectedFeedback:
    """A proposal that failed static validation, evaluation or the sample-size rule, and why."""

    expression: str
    codes: tuple[str, ...]
    message: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str):
            raise TypeError("expression must be a string")
        codes = tuple(self.codes)
        unknown = sorted(set(codes) - FEEDBACK_ERROR_CODES)
        if unknown:
            raise ValueError(f"unknown rejection codes {unknown}")
        if not isinstance(self.message, str):
            raise TypeError("message must be a string")
        object.__setattr__(self, "expression", self.expression[:MAX_FEEDBACK_TEXT])
        object.__setattr__(self, "codes", codes)
        object.__setattr__(self, "message", self.message[:MAX_FEEDBACK_TEXT])


def _typed_tuple(items: Sequence[Any], kind: type, name: str) -> tuple[Any, ...]:
    if isinstance(items, (str, bytes, Mapping)):
        raise TypeError(f"{name} must be a sequence of {kind.__name__}")
    values = tuple(items)
    for item in values:
        if type(item) is not kind:  # exact type: no subclasses carrying extra fields
            raise LeakageError(f"{name} accepts only {kind.__name__}, got {type(item).__name__}")
    return values


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """Everything a proposer is allowed to know when proposing a batch."""

    round_index: int
    n_requested: int
    grammar: str
    top: tuple[EvaluatedFeedback, ...] = ()
    bottom: tuple[EvaluatedFeedback, ...] = ()
    last_round: tuple[EvaluatedFeedback, ...] = ()
    rejected: tuple[RejectedFeedback, ...] = ()
    n_trials_so_far: int = 0
    budget_remaining: int = 0

    def __post_init__(self) -> None:
        for name in ("round_index", "n_requested", "n_trials_so_far", "budget_remaining"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(f"{name} must be a non-negative integer")
        if not isinstance(self.grammar, str) or not self.grammar:
            raise TypeError("grammar must be a non-empty string")
        object.__setattr__(self, "top", _typed_tuple(self.top, EvaluatedFeedback, "top"))
        object.__setattr__(self, "bottom", _typed_tuple(self.bottom, EvaluatedFeedback, "bottom"))
        object.__setattr__(
            self, "last_round", _typed_tuple(self.last_round, EvaluatedFeedback, "last_round")
        )
        object.__setattr__(self, "rejected", _typed_tuple(self.rejected, RejectedFeedback, "rejected"))

    @property
    def evaluated(self) -> tuple[EvaluatedFeedback, ...]:
        """Distinct feedback items across ``top``, ``bottom`` and ``last_round``."""

        seen: dict[str, EvaluatedFeedback] = {}
        for item in (*self.top, *self.bottom, *self.last_round):
            seen.setdefault(item.expression, item)
        return tuple(seen.values())


@dataclass(frozen=True)
class Proposal:
    """One candidate expression with a free-text rationale and generator metadata."""

    expression: str
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.expression, str):
            raise TypeError("Proposal.expression must be a string")
        if not isinstance(self.rationale, str):
            raise TypeError("Proposal.rationale must be a string")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@runtime_checkable
class Proposer(Protocol):
    """Generates candidate expressions from a :class:`ProposalContext`."""

    name: str

    def propose(self, context: ProposalContext) -> list[Proposal]: ...

    def describe(self) -> dict[str, Any]: ...


__all__ = [
    "EvaluatedFeedback",
    "FEEDBACK_ERROR_CODES",
    "FORMATION_WINDOW",
    "LeakageError",
    "Proposal",
    "ProposalContext",
    "Proposer",
    "ProposerError",
    "RejectedFeedback",
]
