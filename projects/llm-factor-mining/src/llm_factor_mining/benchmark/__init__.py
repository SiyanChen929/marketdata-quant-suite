"""Synthetic planted-alpha benchmark (synthetic data only)."""

from .report import render_markdown, write_summary
from .runner import (
    BANNER,
    BenchmarkConfig,
    aggregate,
    make_proposer,
    oracle_scores,
    parse_arm,
    run_benchmark,
    score_search,
    true_alpha,
)
from .synthetic import (
    DEFAULT_PLANTED,
    DRAWN_PLANTED,
    FIXED_PLANTED,
    PLANTED_FAMILIES,
    PlantedFactor,
    SyntheticMarket,
    SyntheticMarketConfig,
    best_matches,
    factor_value_correlation,
    recovery_matrix,
    simulate_market,
    structural_match,
)

__all__ = [
    "BANNER",
    "BenchmarkConfig",
    "DEFAULT_PLANTED",
    "DRAWN_PLANTED",
    "FIXED_PLANTED",
    "PLANTED_FAMILIES",
    "PlantedFactor",
    "SyntheticMarket",
    "SyntheticMarketConfig",
    "aggregate",
    "best_matches",
    "factor_value_correlation",
    "make_proposer",
    "oracle_scores",
    "parse_arm",
    "recovery_matrix",
    "render_markdown",
    "run_benchmark",
    "score_search",
    "simulate_market",
    "structural_match",
    "true_alpha",
    "write_summary",
]
