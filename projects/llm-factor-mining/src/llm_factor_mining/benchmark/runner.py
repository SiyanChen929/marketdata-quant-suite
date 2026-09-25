"""Planted-alpha benchmark: equal-budget searches across seeds and SNR levels.

For every ``(snr, seed)`` a synthetic market is generated, split
chronologically, and each arm (a proposer, optionally with its own budget
``name@budget``) runs the full protocol
(:func:`llm_factor_mining.search.run_search`).  Markets with ``snr = 0``
(``null_seeds``) contain no planted return component at all: every factor
selected there is a false discovery, so they measure the protocol's
false-selection rate per arm.  Scores (see :func:`score_search`) are then
aggregated per arm and SNR over *complete* runs (the full budget used);
incomplete and aborted runs are listed separately, never averaged in.
All numbers describe synthetic data only.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from ..dsl.validate import DSLLimits
from ..evaluate.engine import Evaluator
from ..evaluate.metrics import MetricConfig, embargoed_signal_dates, evaluate_signal
from ..inference import benjamini_hochberg, t_to_p
from ..jsonutil import json_safe
from ..protocol.contamination import behavioural_novelty, library_novelty, library_signals
from ..protocol.seal import COMPOSITE_NAME, composite_signal
from ..protocol.splits import Splits, chronological_splits, truncate_panel
from ..proposers.base import Proposer
from ..proposers.evolutionary import EvolutionaryProposer
from ..proposers.random_grammar import RandomGrammarProposer
from ..search import SearchConfig, SearchResult, software_versions, run_search
from .synthetic import (
    DEFAULT_PLANTED,
    PlantedFactor,
    SyntheticMarket,
    SyntheticMarketConfig,
    best_matches,
    factor_value_correlation,
    recovery_matrix,
    simulate_market,
)


BANNER = "SYNTHETIC DATA - NOT EVIDENCE ABOUT REAL MARKETS"
BASELINE_PROPOSERS = ("random", "evolutionary")
KNOWN_PROPOSERS = ("random", "evolutionary", "llm")
PROPOSER_SEED_OFFSET = 10_000

LLMFactory = Callable[..., Proposer]
"""``factory(seed, *, run_tag, forbidden_tokens) -> Proposer`` for the LLM arm."""


def parse_arm(label: str, default_budget: int) -> tuple[str, int]:
    """``"evolutionary@2000"`` -> ``("evolutionary", 2000)``; a bare name uses the default budget."""

    name, _, budget = label.partition("@")
    if name not in KNOWN_PROPOSERS:
        raise ValueError(f"unknown proposer {name!r} in arm {label!r}")
    if not budget:
        return name, int(default_budget)
    if not budget.isdigit() or int(budget) < 1:
        raise ValueError(f"arm {label!r} needs a positive integer budget after '@'")
    return name, int(budget)


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark grid and protocol settings (identical for every arm except its budget)."""

    seeds: tuple[int, ...] = (0, 1, 2)
    snr_levels: tuple[float, ...] = (0.05, 0.15)
    null_seeds: tuple[int, ...] = tuple(range(10))
    n_symbols: int = 60
    n_dates: int = 750
    budget: int = 200
    batch_size: int = 20
    proposers: tuple[str, ...] = BASELINE_PROPOSERS
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    metric: MetricConfig = field(default_factory=MetricConfig)
    limits: DSLLimits = field(default_factory=DSLLimits)
    fdr_method: str = "bh"
    fdr_alpha: float = 0.10
    confirm_method: str = "bh"
    confirm_alpha: float = 0.10
    top_k: int = 5
    max_abs_corr: float = 0.7
    min_ic_dates: int = 100
    min_ic_fraction: float = 0.5
    recovery_threshold: float = 0.7
    true_discovery_threshold: float = 0.1
    test_alpha: float = 0.05
    recall_stride: int = 5
    planted: tuple[PlantedFactor, ...] = DEFAULT_PLANTED

    def __post_init__(self) -> None:
        if not self.seeds or not self.snr_levels:
            raise ValueError("need at least one seed and one SNR level")
        if any(float(level) <= 0 for level in self.snr_levels):
            raise ValueError("snr_levels must be positive; null markets are configured with null_seeds")
        for label in self.proposers:
            parse_arm(label, self.budget)
        if len(set(self.proposers)) != len(self.proposers):
            raise ValueError("arm labels must be unique")

    def arms(self) -> list[tuple[str, str, int]]:
        """``(label, proposer name, budget)`` for every configured arm."""

        return [(label, *parse_arm(label, self.budget)) for label in self.proposers]

    def search_config(self, budget: int | None = None) -> SearchConfig:
        return SearchConfig(
            budget=self.budget if budget is None else int(budget),
            batch_size=self.batch_size,
            metric=self.metric,
            limits=self.limits,
            fdr_method=self.fdr_method,
            fdr_alpha=self.fdr_alpha,
            confirm_method=self.confirm_method,
            confirm_alpha=self.confirm_alpha,
            top_k=self.top_k,
            max_abs_corr=self.max_abs_corr,
            min_ic_dates=self.min_ic_dates,
            min_ic_fraction=self.min_ic_fraction,
            record_timestamps=False,
        )

    def market_config(self, snr: float, seed: int) -> SyntheticMarketConfig:
        return SyntheticMarketConfig(
            n_symbols=self.n_symbols,
            n_dates=self.n_dates,
            snr=float(snr),
            seed=int(seed),
            planted=self.planted,
            lag=self.metric.lag,
        )

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def make_proposer(
    name: str,
    seed: int,
    *,
    limits: DSLLimits,
    llm_factory: LLMFactory | None = None,
    run_tag: str = "",
    forbidden_tokens: Sequence[str] = (),
) -> Proposer:
    """Instantiate a proposer by name with a proposer-specific seed."""

    if name == "random":
        return RandomGrammarProposer(seed, limits=limits)
    if name == "evolutionary":
        return EvolutionaryProposer(seed, limits=limits)
    if name == "llm":
        if llm_factory is None:
            raise ValueError("the llm proposer needs an llm_factory")
        return llm_factory(seed, run_tag=run_tag, forbidden_tokens=tuple(forbidden_tokens))
    raise ValueError(f"unknown proposer {name!r}")


def _window_dates(panel_dates: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return panel_dates[(panel_dates >= start) & (panel_dates <= end)]


def _report_numbers(report: Any) -> dict[str, float]:
    return {
        "ic_mean": report.ic_mean,
        "icir": report.icir,
        "t_nw": report.ic_tstat_nw,
        "ls_net_sharpe": report.ls_net_sharpe,
        "n_ic_dates": report.n_ic_dates,
    }


def true_alpha(market: SyntheticMarket) -> pd.DataFrame:
    """The planted composite: proportional to the true expected-return signal (equal weights)."""

    return composite_signal(list(market.planted_signals.values()))


def oracle_scores(market: SyntheticMarket, splits: Splits, metric: MetricConfig, *, test_alpha: float = 0.05) -> dict[str, Any]:
    """Test-window scores of the planted signals themselves (a reference ceiling).

    ``power`` is the share of planted signals whose one-sided test IC passes BH
    at ``test_alpha`` within the planted set: how often a *true* signal is
    detectable on this test window.
    """

    out: dict[str, Any] = {}
    close = market.panel.close
    t_values = []
    for name, signal in market.planted_signals.items():
        out[name] = _report_numbers(evaluate_signal(signal, close, config=metric, window=splits.test).report)
        t_values.append(out[name]["t_nw"])
    out[COMPOSITE_NAME] = _report_numbers(
        evaluate_signal(true_alpha(market), close, config=metric, window=splits.test).report
    )
    p_values = [t_to_p(value, alternative="greater") for value in t_values]
    out["power"] = benjamini_hochberg(p_values, test_alpha).n_rejected / len(p_values) if p_values else None
    return json_safe(out)


def score_search(
    result: SearchResult,
    market: SyntheticMarket,
    splits: Splits,
    config: BenchmarkConfig,
    *,
    library: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Benchmark metrics of one completed (revealed) search.

    * ``recovery``: planted signal ``k`` is recovered when some selected
      (oriented) factor's test-window values have mean cross-sectional rank
      correlation ``rho >= recovery_threshold`` with it - signed, because a
      factor anti-correlated with the planted signal bets against it;
    * ``recall``: ``|rho| >= recovery_threshold`` over *every* evaluated trial
      on the formation window (unoriented: did the proposer ever find it);
      ``first_found_trial`` is the first trial index that did;
    * ``fdp_true``: share of selected factors that are false discoveries by
      ground truth - oriented rank correlation with the planted composite (the
      true expected-return signal) on the test window below
      ``true_discovery_threshold``; on a null market (``snr = 0``) every
      selected factor is false;
    * ``test_nonsignificant_rate``: share of selected factors whose one-sided
      test IC fails BH at ``test_alpha`` (low test power, not falsity);
    * test IC/ICIR of the selected factors (mean) and of their composite;
    * behavioural novelty (primary) and structural novelty (secondary) of the
      selected factors against the reference library.
    """

    if result.reveal is None:
        raise ValueError("score_search needs a revealed search result")
    metric = config.metric
    threshold = config.recovery_threshold
    reveal = result.reveal
    selected = list(result.frozen.factors)
    panel_dates = market.panel.dates
    test_dates = _window_dates(panel_dates, splits.test.start, splits.test.end)
    min_names = metric.effective_min_names

    signals = {factor.expression_hash: reveal.signals[factor.expression_hash] for factor in selected}
    matches = best_matches(recovery_matrix(signals, market.planted_signals, test_dates, min_names=min_names), signed=True)
    recovered = {name: bool(match["rho"] is not None and match["rho"] >= threshold) for name, match in matches.items()}

    alpha = true_alpha(market)
    null_market = float(market.config.snr) == 0.0
    true_rho = {key: factor_value_correlation(values, alpha, test_dates, min_names=min_names) for key, values in signals.items()}
    false = [
        null_market or not (math.isfinite(true_rho[factor.expression_hash]) and true_rho[factor.expression_hash] >= config.true_discovery_threshold)
        for factor in selected
    ]
    fdp_true = float(np.mean(false)) if selected else None

    t_stats = [reveal.evaluations[factor.expression_hash].report.ic_tstat_nw for factor in selected]
    p_values = [t_to_p(value, alternative="greater") for value in t_stats]
    nonsignificant: float | None = None
    if selected:
        nonsignificant = 1.0 - benjamini_hochberg(p_values, config.test_alpha).n_rejected / len(selected)
    test_ic = [reveal.evaluations[factor.expression_hash].report.ic_mean for factor in selected]
    test_icir = [reveal.evaluations[factor.expression_hash].report.icir for factor in selected]
    composite = (
        _report_numbers(reveal.evaluations[COMPOSITE_NAME].report) if COMPOSITE_NAME in reveal.evaluations else None
    )

    formation_panel = truncate_panel(market.panel, splits.formation.end)
    formation_dates = _window_dates(formation_panel.dates, splits.formation.start, splits.formation.end)
    ic_dates = embargoed_signal_dates(formation_panel.dates, splits.formation, lag=metric.lag, horizon=metric.horizon)
    engine = Evaluator(formation_panel, limits=config.limits, max_cache_entries=8)
    evaluated = [trial for trial in result.trials if trial.status == "evaluated" and trial.canonical]
    recall_best = {name: 0.0 for name in market.planted_signals}
    recall_trial: dict[str, int | None] = {name: None for name in market.planted_signals}
    first_found: dict[str, int | None] = {name: None for name in market.planted_signals}
    planted_formation = {name: signal.loc[formation_panel.dates] for name, signal in market.planted_signals.items()}
    for trial in evaluated:
        values = engine.evaluate(trial.canonical)  # type: ignore[arg-type]
        row = recovery_matrix(
            {"candidate": values}, planted_formation, formation_dates, min_names=min_names, stride=config.recall_stride
        )
        for name, scores in row.items():
            value = abs(scores["candidate"])
            if not math.isfinite(value):
                continue
            if value > recall_best[name]:
                recall_best[name], recall_trial[name] = value, trial.index
            if value >= threshold and first_found[name] is None:
                first_found[name] = trial.index
    recalled = {name: bool(value >= threshold) for name, value in recall_best.items()}

    library_values = library if library is not None else library_signals(engine)
    behavioural = [
        behavioural_novelty(engine.evaluate(factor.expression) * float(factor.orientation), library_values, ic_dates, min_names=min_names)
        for factor in selected
    ]
    structural = [library_novelty(factor.expression) for factor in selected]
    n_planted = len(market.planted_signals)

    def mean_of(values: list[float]) -> float | None:
        finite = [value for value in values if value is not None and math.isfinite(value)]
        return float(np.mean(finite)) if finite else None

    return json_safe(
        {
            "n_trials": result.n_trials,
            "n_evaluated": result.n_evaluated,
            "n_degenerate": result.n_degenerate,
            "n_invalid": result.n_invalid,
            "n_errors": result.n_errors,
            "n_duplicates": result.n_duplicates,
            "validity_rate": result.validity_rate,
            **{key: int(value) for key, value in result.funnel.items() if key.startswith(("n_", "m_"))},
            "n_selected": len(selected),
            "any_selected": 1.0 if selected else 0.0,
            "recovery_rate": sum(recovered.values()) / n_planted,
            "recovered": recovered,
            "best_match": matches,
            "recall_rate": sum(recalled.values()) / n_planted,
            "recalled": recalled,
            "recall_best_abs_rho": recall_best,
            "recall_best_trial": recall_trial,
            "first_found_trial": first_found,
            "fdp_true": fdp_true,
            "true_alpha_rho": [true_rho[factor.expression_hash] for factor in selected],
            "test_nonsignificant_rate": nonsignificant,
            "selected_test_ic_mean": float(np.mean(test_ic)) if test_ic else None,
            "selected_test_icir_mean": float(np.mean(test_icir)) if test_icir else None,
            "selected_test_t": t_stats,
            "composite_test": composite,
            "behavioural_novelty_mean": mean_of([item.novelty for item in behavioural]),
            "structural_novelty_mean": mean_of([item.novelty for item in structural]),
            "selected": [
                {
                    **factor.to_dict(),
                    "behavioural_nearest": b.nearest,
                    "behavioural_novelty": b.novelty,
                    "structural_nearest": s.nearest,
                    "structural_novelty": s.novelty,
                    "true_alpha_rho": true_rho[factor.expression_hash],
                }
                for factor, b, s in zip(selected, behavioural, structural)
            ],
            "ledger_head": result.ledger_head,
            "commitment": result.commitment,
            "stop_reason": result.stop_reason,
        }
    )


AGGREGATED_METRICS: tuple[str, ...] = (
    "recovery_rate",
    "recall_rate",
    "fdp_true",
    "test_nonsignificant_rate",
    "any_selected",
    "n_selected",
    "n_screen_survivors",
    "n_confirmed",
    "n_behaviour_classes",
    "n_behaviour_duplicates",
    "n_degenerate",
    "selected_test_ic_mean",
    "selected_test_icir_mean",
    "composite_test_ic",
    "composite_test_icir",
    "composite_test_t",
    "composite_test_ls_net_sharpe",
    "n_trials",
    "validity_rate",
    "behavioural_novelty_mean",
    "structural_novelty_mean",
    "runtime_seconds",
)


def _flatten(row: Mapping[str, Any]) -> dict[str, Any]:
    flat = dict(row)
    composite = row.get("composite_test") or {}
    flat["composite_test_ic"] = composite.get("ic_mean")
    flat["composite_test_icir"] = composite.get("icir")
    flat["composite_test_t"] = composite.get("t_nw")
    flat["composite_test_ls_net_sharpe"] = composite.get("ls_net_sharpe")
    return flat


def _stats(values: Sequence[Any]) -> dict[str, Any]:
    numbers = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numbers:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    array = np.asarray(numbers)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if array.size > 1 else None,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def aggregate(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per ``(arm, snr)``: mean/std/min/max of each metric across *complete* runs.

    ``n_runs`` counts every run of the cell and ``n_complete`` the runs that
    used their full budget; only the latter enter the statistics.
    """

    groups: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for run in runs:
        groups.setdefault((run["arm"], float(run["snr"])), []).append(run)
    out = []
    for (arm, snr), members in sorted(groups.items(), key=lambda item: (item[0][1], item[0][0])):
        rows = [_flatten(run) for run in members if run.get("complete")]
        planted_names = sorted(rows[0].get("recovered", {})) if rows else []
        out.append(
            {
                "arm": arm,
                "proposer": members[0]["proposer"],
                "budget": members[0]["budget"],
                "snr": snr,
                "n_runs": len(members),
                "n_complete": len(rows),
                "n_seeds": len(rows),
                "seeds": [row["seed"] for row in rows],
                "metrics": {name: _stats([row.get(name) for row in rows]) for name in AGGREGATED_METRICS},
                "recovered_counts": {
                    name: int(sum(bool(row["recovered"].get(name)) for row in rows)) for name in planted_names
                },
                "recalled_counts": {
                    name: int(sum(bool(row["recalled"].get(name)) for row in rows)) for name in planted_names
                },
            }
        )
    return out


def _arm_status(runs: Sequence[Mapping[str, Any]], arm: str) -> str:
    mine = [run for run in runs if run["arm"] == arm]
    incomplete = [run for run in mine if not run.get("complete")]
    if not incomplete:
        return "run"
    reasons = sorted({str(run.get("stop_reason")) for run in incomplete})
    return f"run; {len(incomplete)} of {len(mine)} runs incomplete (stop reasons: {', '.join(reasons)})"


def run_benchmark(
    config: BenchmarkConfig | None = None,
    *,
    llm_factory: LLMFactory | None = None,
    llm_status: str = "not run: no credentials",
    runs_dir: str | Path | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run the full grid and return a JSON-safe summary (see ``write_summary``)."""

    cfg = config or BenchmarkConfig()
    say = progress or (lambda message: None)
    arms = [arm for arm in cfg.arms() if arm[1] != "llm"]
    llm_arms = [arm for arm in cfg.arms() if arm[1] == "llm"]
    status: dict[str, str] = {}
    if llm_factory is not None:
        arms += llm_arms or [("llm", "llm", cfg.budget)]
    elif llm_arms or "llm" in cfg.proposers:
        for label, _, _ in llm_arms:
            status[label] = llm_status
    started = time.perf_counter()
    markets: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    grid = [(float(snr), int(seed)) for snr in cfg.snr_levels for seed in cfg.seeds]
    grid += [(0.0, int(seed)) for seed in cfg.null_seeds]
    for snr, seed in grid:
        tick = time.perf_counter()
        market = simulate_market(cfg.market_config(snr, seed))
        splits = chronological_splits(
            market.panel.dates, fractions=cfg.fractions, lag=cfg.metric.lag, max_horizon=cfg.metric.horizon
        )
        formation_panel = truncate_panel(market.panel, splits.formation.end)
        library = library_signals(Evaluator(formation_panel, limits=cfg.limits))
        ic_dates = embargoed_signal_dates(
            formation_panel.dates, splits.formation, lag=cfg.metric.lag, horizon=cfg.metric.horizon
        )
        markets.append(
            {
                "snr": snr,
                "seed": seed,
                "splits": splits.to_dict(),
                "diagnostics": market.diagnostics(),
                "oracle_test": oracle_scores(market, splits, cfg.metric, test_alpha=cfg.test_alpha),
                "planted_behavioural_novelty": {
                    name: behavioural_novelty(
                        signal.loc[formation_panel.dates], library, ic_dates, min_names=cfg.metric.effective_min_names
                    ).to_dict()
                    for name, signal in market.planted_signals.items()
                },
                "generation_seconds": round(time.perf_counter() - tick, 2),
            }
        )
        for label, name, budget in arms:
            tick = time.perf_counter()
            run_tag = f"snr={snr:g},market_seed={seed}"
            proposer = make_proposer(
                name,
                PROPOSER_SEED_OFFSET + seed,
                limits=cfg.limits,
                llm_factory=llm_factory,
                run_tag=run_tag,
                forbidden_tokens=market.panel.symbols,
            )
            run_dir = None if runs_dir is None else Path(runs_dir) / f"snr{snr:g}_seed{seed}_{label.replace('@', '_b')}"
            result = run_search(proposer, market.panel, splits, cfg.search_config(budget), run_dir=run_dir)
            base = {
                "arm": label,
                "proposer": name,
                "budget": budget,
                "snr": snr,
                "seed": seed,
                "proposer_seed": PROPOSER_SEED_OFFSET + seed,
                "served_models": list(result.proposer_provenance.get("served_models", [])),
                "stop_reason": result.stop_reason,
                "complete": result.complete,
                "aborted": result.aborted,
            }
            if result.aborted:
                elapsed = round(time.perf_counter() - tick, 2)
                runs.append(
                    {
                        **base,
                        "runtime_seconds": elapsed,
                        "n_trials": result.n_trials,
                        "n_proposer_errors": result.n_proposer_errors,
                        "ledger_head": result.ledger_head,
                    }
                )
                say(f"snr={snr:g} seed={seed} {label:<14} ABORTED ({result.stop_reason}) trials={result.n_trials}")
                continue
            scores = score_search(result, market, splits, cfg, library=library)
            elapsed = round(time.perf_counter() - tick, 2)
            runs.append({**base, "runtime_seconds": elapsed, **scores, "stop_reason": result.stop_reason})
            say(
                f"snr={snr:g} seed={seed} {label:<14} recovery={scores['recovery_rate']:.2f} "
                f"recall={scores['recall_rate']:.2f} screen={scores['n_screen_survivors']} "
                f"confirmed={scores['n_confirmed']} selected={scores['n_selected']} "
                f"trials={scores['n_trials']} ({elapsed:.1f}s)"
            )
    for label, _, _ in arms:
        status[label] = _arm_status(runs, label)
    planted_runs = [run for run in runs if float(run["snr"]) > 0]
    null_runs = [run for run in runs if float(run["snr"]) == 0]
    return json_safe(
        {
            "banner": BANNER,
            "config": cfg.to_dict(),
            "search_configs": {label: cfg.search_config(budget).to_dict() for label, _, budget in arms},
            "proposer_seed_offset": PROPOSER_SEED_OFFSET,
            "proposers": status,
            "planted": [asdict(factor) for factor in cfg.planted],
            "markets": markets,
            "runs": runs,
            "aggregate": aggregate(planted_runs),
            "null_aggregate": aggregate(null_runs),
            "incomplete_runs": [
                {key: run.get(key) for key in ("arm", "snr", "seed", "stop_reason", "n_trials", "aborted")}
                for run in runs
                if not run.get("complete")
            ],
            "software": software_versions(),
            "runtime_seconds": round(time.perf_counter() - started, 1),
        }
    )


__all__ = [
    "BANNER",
    "BenchmarkConfig",
    "aggregate",
    "make_proposer",
    "oracle_scores",
    "parse_arm",
    "run_benchmark",
    "score_search",
    "true_alpha",
]
