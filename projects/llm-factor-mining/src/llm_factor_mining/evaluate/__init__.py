"""Causal factor evaluation and leakage-aware metrics."""

from __future__ import annotations

from collections.abc import Iterable

from ..data import Panel
from ..dsl.canonical import canonical_string, structural_hash
from ..dsl.nodes import Node
from ..dsl.parser import parse
from .engine import CacheInfo, Evaluator, ExpressionError, KERNELS, evaluate_expression, terminal_frame
from .metrics import (
    EvaluationWindow,
    FactorEvaluation,
    FactorReport,
    ICSummary,
    MetricConfig,
    default_nw_lags,
    embargoed_signal_dates,
    evaluate_signal,
    evaluate_windows,
    forward_returns,
    holding_weights,
    ic_summary,
    long_short_backtest,
    long_short_weights,
    newey_west_se,
    quantile_buckets,
    quantile_returns,
    rank_ic,
    restrict,
    top_quantile_turnover,
)


def evaluate_factor(
    panel: Panel,
    expression: str | Node,
    *,
    windows: Iterable[EvaluationWindow] | None = None,
    config: MetricConfig | None = None,
    evaluator: Evaluator | None = None,
) -> dict[str, FactorEvaluation]:
    """Compute a factor once on the full panel, then score each embargoed window."""

    engine = evaluator or Evaluator(panel)
    if engine.panel is not panel:
        raise ValueError("evaluator is bound to a different panel")
    signal = engine.evaluate(expression)  # raises ExpressionError before any metric work
    node = parse(expression) if isinstance(expression, str) else expression
    return evaluate_windows(
        signal,
        panel.close,
        list(windows) if windows is not None else [EvaluationWindow()],
        config=config,
        expression=canonical_string(node),
        expression_hash=structural_hash(node),
        panel_sha256=panel.content_sha256,
    )


__all__ = [
    "CacheInfo",
    "EvaluationWindow",
    "Evaluator",
    "ExpressionError",
    "FactorEvaluation",
    "FactorReport",
    "ICSummary",
    "KERNELS",
    "MetricConfig",
    "default_nw_lags",
    "embargoed_signal_dates",
    "evaluate_expression",
    "evaluate_factor",
    "evaluate_signal",
    "evaluate_windows",
    "forward_returns",
    "holding_weights",
    "ic_summary",
    "long_short_backtest",
    "long_short_weights",
    "newey_west_se",
    "quantile_buckets",
    "quantile_returns",
    "rank_ic",
    "restrict",
    "terminal_frame",
    "top_quantile_turnover",
]
