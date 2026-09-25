"""Operator and terminal registry for the factor language.

The registry is pure metadata: names, argument kinds, window bounds and the
lookback each operator adds.  Numerical implementations live in
:mod:`llm_factor_mining.evaluate.engine`.

Lookahead is impossible by construction.  Every time-series operator reads the
current row and a bounded number of *earlier* rows of its inputs; no operator
accepts a negative or zero shift (``delay``/``delta`` require ``d >= 1``, rolling
operators ``d >= 2``), and cross-sectional operators read one date at a time.
Forward returns are not a terminal of the language; they exist only inside the
evaluation metrics, after an explicit execution lag.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Literal


class ArgKind(str, Enum):
    """Kind of value an operator slot accepts."""

    SERIES = "series"  # any expression; numeric literals broadcast to a constant panel
    WINDOW = "window"  # integer literal number of trading sessions
    CONST = "const"  # floating-point literal parameter


Category = Literal["elementwise", "time_series", "cross_sectional"]


@dataclass(frozen=True)
class OperatorSpec:
    """Static description of one operator.

    ``window_lookback_offset`` is ``None`` for operators without a window.
    Otherwise the operator needs ``window + offset`` rows before the current
    row: ``-1`` for rolling windows (a ``d``-row window reaches back ``d - 1``
    rows) and ``0`` for shifts (``delay(x, d)`` reaches back ``d`` rows).
    """

    name: str
    arg_kinds: tuple[ArgKind, ...]
    category: Category
    description: str
    commutative: bool = False
    min_window: int = 2
    window_lookback_offset: int | None = None
    const_bounds: tuple[float, float] | None = None

    @property
    def arity(self) -> int:
        return len(self.arg_kinds)

    @property
    def window_slot(self) -> int | None:
        for index, kind in enumerate(self.arg_kinds):
            if kind is ArgKind.WINDOW:
                return index
        return None

    def lookback(self, window: int | None) -> int:
        """Rows of history this operator adds on top of its inputs' lookback."""

        if self.window_lookback_offset is None or window is None:
            return 0
        return max(int(window) + self.window_lookback_offset, 0)


@dataclass(frozen=True)
class TerminalSpec:
    """A data field exposed to expressions."""

    name: str
    description: str
    lookback: int = 0


S, W, C = ArgKind.SERIES, ArgKind.WINDOW, ArgKind.CONST


def _elementwise(
    name: str, arity: int, description: str, *, commutative: bool = False
) -> OperatorSpec:
    return OperatorSpec(
        name, (S,) * arity, "elementwise", description, commutative=commutative
    )


def _rolling(
    name: str, n_series: int, description: str, *, commutative: bool = False
) -> OperatorSpec:
    return OperatorSpec(
        name,
        (S,) * n_series + (W,),
        "time_series",
        description,
        commutative=commutative,
        window_lookback_offset=-1,
    )


def _shift(name: str, description: str) -> OperatorSpec:
    return OperatorSpec(
        name,
        (S, W),
        "time_series",
        description,
        min_window=1,
        window_lookback_offset=0,
    )


_OPERATOR_LIST: tuple[OperatorSpec, ...] = (
    _elementwise("add", 2, "x + y", commutative=True),
    _elementwise("sub", 2, "x - y"),
    _elementwise("mul", 2, "x * y", commutative=True),
    _elementwise("div", 2, "x / y; NaN where |y| < 1e-12"),
    _elementwise("neg", 1, "-x"),
    _elementwise("abs", 1, "|x|"),
    _elementwise("sign", 1, "sign(x) in {-1, 0, 1}"),
    _elementwise("log", 1, "signed log: sign(x) * log(1 + |x|)"),
    _elementwise("sqrt", 1, "signed square root: sign(x) * sqrt(|x|)"),
    OperatorSpec(
        "power",
        (S, C),
        "elementwise",
        "signed power: sign(x) * |x| ** c",
        const_bounds=(-4.0, 4.0),
    ),
    _shift("delay", "x lagged by d sessions"),
    _shift("delta", "x - delay(x, d)"),
    _rolling("ts_mean", 1, "rolling mean over the last d sessions"),
    _rolling("ts_std", 1, "rolling sample standard deviation (ddof=1)"),
    _rolling("ts_sum", 1, "rolling sum over the last d sessions"),
    _rolling("ts_min", 1, "rolling minimum"),
    _rolling("ts_max", 1, "rolling maximum"),
    _rolling("ts_rank", 1, "percentile rank of today's value within the last d values, in (0, 1]"),
    _rolling("ts_zscore", 1, "(x - ts_mean(x, d)) / ts_std(x, d)"),
    _rolling("ts_argmax", 1, "sessions since the window maximum, in [0, d-1]; ties -> most recent"),
    _rolling("ts_argmin", 1, "sessions since the window minimum, in [0, d-1]; ties -> most recent"),
    _rolling("decay_linear", 1, "linearly decaying weighted mean, weight d on today down to 1"),
    _rolling("ts_corr", 2, "rolling Pearson correlation of x and y", commutative=True),
    _rolling("ts_cov", 2, "rolling sample covariance of x and y (ddof=1)", commutative=True),
    OperatorSpec("cs_rank", (S,), "cross_sectional", "per-date percentile rank in (0, 1]"),
    OperatorSpec("cs_zscore", (S,), "cross_sectional", "per-date z-score (ddof=1)"),
    OperatorSpec("cs_demean", (S,), "cross_sectional", "per-date deviation from the mean"),
)

OPERATORS: Mapping[str, OperatorSpec] = MappingProxyType(
    {spec.name: spec for spec in _OPERATOR_LIST}
)

_TERMINAL_LIST: tuple[TerminalSpec, ...] = (
    TerminalSpec("open", "session open price"),
    TerminalSpec("high", "session high price"),
    TerminalSpec("low", "session low price"),
    TerminalSpec("close", "session close price"),
    TerminalSpec("volume", "session share volume"),
    TerminalSpec("returns", "close-to-close return: close / delay(close, 1) - 1", lookback=1),
    TerminalSpec("vwap_proxy", "typical price (high + low + close) / 3; daily bars carry no true VWAP"),
    TerminalSpec("dollar_volume", "close * volume"),
)

TERMINALS: Mapping[str, TerminalSpec] = MappingProxyType(
    {spec.name: spec for spec in _TERMINAL_LIST}
)

INFIX_OPERATORS: Mapping[str, str] = MappingProxyType(
    {"+": "add", "-": "sub", "*": "mul", "/": "div"}
)


def describe_language() -> str:
    """Return a compact plain-text grammar card (used later in LLM prompts)."""

    lines = ["Terminals:"]
    lines.extend(f"  {spec.name}: {spec.description}" for spec in _TERMINAL_LIST)
    lines.append("Operators (d = integer window literal, c = numeric literal):")
    for spec in _OPERATOR_LIST:
        names = []
        series_index = 0
        for kind in spec.arg_kinds:
            if kind is ArgKind.SERIES:
                names.append("xyz"[series_index])
                series_index += 1
            elif kind is ArgKind.WINDOW:
                names.append("d")
            else:
                names.append("c")
        lines.append(f"  {spec.name}({', '.join(names)}): {spec.description}")
    lines.append("Infix sugar: + - * / and unary minus; numeric literals broadcast.")
    return "\n".join(lines)
