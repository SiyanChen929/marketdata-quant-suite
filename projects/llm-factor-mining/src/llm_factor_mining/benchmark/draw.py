"""Pre-registered draw of the benchmark's "drawn" planted signal.

The textbook planted signals favour any proposer that knows the factor
literature, and the volume-return signal belongs to a documented family
(Kakushadze's Alpha#2 idea; the reference library holds that alpha).  To have
one planted signal that no proposer can know in advance, the benchmark also
plants an expression *drawn at random from the typed grammar* by the
procedure below, fixed before the draw was run:

* candidates ``k = 0, 1, 2, ...`` are ``random_valid_tree`` samples from
  ``numpy.random.default_rng([DRAW_SEED, k])`` with the default
  :class:`~llm_factor_mining.proposers.trees.GrammarConfig` and
  :class:`~llm_factor_mining.dsl.validate.DSLLimits`;
* the first candidate meeting every criterion is accepted:

  - C1 structure: depth >= 3, complexity <= 8, at least one time-series
    operator;
  - C2 well-defined: on the reference null market (60 symbols x 750
    sessions, ``snr = 0``, seed 424242), at least 90% of cells after the
    expression's lookback are finite, and at least 95% of those dates have
    >= 10 distinct values;
  - C3 unfamiliar: its mean per-date rank correlation with every reference
    library entry and with every other planted signal is below 0.3 in
    absolute value on that market (a value-based check, not a syntactic one);
  - C4 well-posed: the benchmark market with all planted signals converges
    (fixed point) at SNR 0.05 and 0.15 for market seed 0.

Every rejected candidate and its reasons are recorded
(``scripts/draw_planted_signal.py`` writes ``results/planted_signal_draw.json``).
The drawn signal comes from the random-grammar baseline's own sampling
distribution, which, if anything, favours that baseline.  "Drawn at random"
does not prove the expression is absent from the literature; C3 only shows it
is far, in values, from the reference library.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
import pandas as pd

from ..dsl.canonical import canonical_string, complexity
from ..dsl.library import LIBRARY
from ..dsl.nodes import Call, Node, to_expression, walk
from ..dsl.operators import OPERATORS
from ..dsl.validate import DSLLimits, depth, lookback
from ..evaluate.engine import Evaluator
from ..evaluate.metrics import rank_ic
from ..jsonutil import json_safe
from ..proposers.random_grammar import random_valid_tree
from ..proposers.trees import GrammarConfig
from .synthetic import PlantedFactor, SyntheticMarketConfig, simulate_market


DRAW_SEED = 20260925
REFERENCE_SEED = 424242
MIN_DEPTH = 3
MAX_COMPLEXITY = 8.0
MIN_COVERAGE = 0.9
MIN_DISTINCT_VALUES = 10
MIN_DISTINCT_SHARE = 0.95
MAX_ABS_CORRELATION = 0.3
CONVERGENCE_SNRS = (0.05, 0.15)


def candidate_tree(k: int) -> Node:
    """Candidate ``k`` of the pre-registered draw."""

    rng = np.random.default_rng([DRAW_SEED, int(k)])
    tree, _ = random_valid_tree(rng, GrammarConfig(), DSLLimits())
    return tree


def _mean_abs_rank_corr(left: pd.DataFrame, right: pd.DataFrame, dates: pd.DatetimeIndex) -> float:
    series = rank_ic(left.loc[dates], right.loc[dates], min_names=10).dropna()
    return abs(float(series.mean())) if len(series) else float("nan")


def assess_candidate(
    node: Node,
    evaluator: Evaluator,
    references: Mapping[str, pd.DataFrame],
    *,
    fixed_planted: tuple[PlantedFactor, ...],
    check_convergence: bool = True,
) -> dict[str, Any]:
    """Criteria C1-C4 for one candidate; ``reasons`` is empty when it is accepted."""

    reasons: list[str] = []
    info: dict[str, Any] = {
        "expression": to_expression(node),
        "canonical": canonical_string(node),
        "depth": depth(node),
        "complexity": complexity(node),
    }
    has_ts = any(
        isinstance(sub, Call) and OPERATORS[sub.op].category == "time_series" for _, sub in walk(node)
    )
    if info["depth"] < MIN_DEPTH:
        reasons.append(f"C1: depth {info['depth']} < {MIN_DEPTH}")
    if info["complexity"] > MAX_COMPLEXITY:
        reasons.append(f"C1: complexity {info['complexity']:g} > {MAX_COMPLEXITY:g}")
    if not has_ts:
        reasons.append("C1: no time-series operator")
    if reasons:
        return {**info, "reasons": reasons}

    values = evaluator.evaluate(node)
    warm = min(lookback(node) + 1, len(values) - 1)
    body = values.iloc[warm:]
    coverage = float(np.isfinite(body.to_numpy(dtype="float64")).mean())
    distinct = body.apply(lambda row: row.dropna().nunique(), axis=1)
    distinct_share = float((distinct >= MIN_DISTINCT_VALUES).mean())
    info.update({"coverage": coverage, "distinct_share": distinct_share})
    if coverage < MIN_COVERAGE:
        reasons.append(f"C2: coverage {coverage:.3f} < {MIN_COVERAGE}")
    if distinct_share < MIN_DISTINCT_SHARE:
        reasons.append(f"C2: only {distinct_share:.3f} of dates have >= {MIN_DISTINCT_VALUES} distinct values")
    if reasons:
        return {**info, "reasons": reasons}

    dates = pd.DatetimeIndex(body.index)
    correlations = {name: _mean_abs_rank_corr(values, ref, dates) for name, ref in references.items()}
    finite = {name: value for name, value in correlations.items() if math.isfinite(value)}
    nearest = max(sorted(finite), key=lambda name: finite[name]) if finite else None
    info.update({"max_abs_corr": None if nearest is None else finite[nearest], "nearest_reference": nearest})
    if nearest is None:
        reasons.append("C3: no defined correlation with the references")
    elif finite[nearest] >= MAX_ABS_CORRELATION:
        reasons.append(f"C3: |corr| {finite[nearest]:.3f} >= {MAX_ABS_CORRELATION} with {nearest}")
    if reasons or not check_convergence:
        return {**info, "reasons": reasons}

    planted = (*fixed_planted, PlantedFactor("drawn", to_expression(node), 1.0, "drawn"))
    iterations: dict[str, int] = {}
    for snr in CONVERGENCE_SNRS:
        try:
            market = simulate_market(SyntheticMarketConfig(snr=snr, seed=0, planted=planted))
        except RuntimeError as exc:
            reasons.append(f"C4: no fixed point at SNR {snr:g}: {exc}")
            break
        iterations[f"{snr:g}"] = market.iterations
    info["iterations"] = iterations
    return {**info, "reasons": reasons}


def draw_planted_signal(
    fixed_planted: tuple[PlantedFactor, ...],
    *,
    max_draws: int = 1000,
    check_convergence: bool = True,
) -> dict[str, Any]:
    """Run the pre-registered draw; returns the accepted candidate and every rejection."""

    reference = simulate_market(
        SyntheticMarketConfig(snr=0.0, seed=REFERENCE_SEED, planted=fixed_planted)
    )
    evaluator = Evaluator(reference.panel, max_cache_entries=64)
    references: dict[str, pd.DataFrame] = {factor.name: evaluator.evaluate(factor.node) for factor in LIBRARY}
    references.update({f"planted:{name}": signal for name, signal in reference.planted_signals.items()})
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for k in range(max_draws):
        node = candidate_tree(k)
        key = canonical_string(node)
        if key in seen:
            rejected.append({"k": k, "expression": to_expression(node), "reasons": ["repeat of an earlier candidate"]})
            continue
        seen.add(key)
        result = assess_candidate(
            node, evaluator, references, fixed_planted=fixed_planted, check_convergence=check_convergence
        )
        if result["reasons"]:
            rejected.append({"k": k, **result})
            continue
        return json_safe(
            {
                "procedure": {
                    "draw_seed": DRAW_SEED,
                    "reference_seed": REFERENCE_SEED,
                    "min_depth": MIN_DEPTH,
                    "max_complexity": MAX_COMPLEXITY,
                    "min_coverage": MIN_COVERAGE,
                    "min_distinct_values": MIN_DISTINCT_VALUES,
                    "min_distinct_share": MIN_DISTINCT_SHARE,
                    "max_abs_correlation": MAX_ABS_CORRELATION,
                    "convergence_snrs": list(CONVERGENCE_SNRS),
                    "grammar": GrammarConfig().__dict__,
                },
                "accepted": {"k": k, **result},
                "rejected": rejected,
            }
        )
    raise RuntimeError(f"no candidate met the criteria in {max_draws} draws")


__all__ = ["DRAW_SEED", "assess_candidate", "candidate_tree", "draw_planted_signal"]
