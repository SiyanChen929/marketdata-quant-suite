"""The factor-search protocol.

Phase 1 - search (formation window only, on a panel truncated at its end):
rounds of ``propose -> parse/validate -> dedup by canonical hash ->
evaluate on formation -> feedback``.  Every processed proposal is written to
the hash-chained ledger.  Invalid proposals, evaluation failures and
*degenerate* trials (too few formation dates with a defined IC for a usable
p-value, see ``SearchConfig.min_ic_dates``) are trials too; exact repeats are
logged as duplicates and do not consume budget.

Phase 2 - selection:

1. *Formation screen* (a filter, not the confirmatory test).  Evaluated
   trials are grouped into behavioural classes: two trials whose per-date
   cross-sectional ranks on the formation IC dates agree up to sign have the
   same rank IC series and therefore test the same hypothesis, so each class
   counts once.  Benjamini-Hochberg (or BY/Holm) is applied to the two-sided
   formation p-values of the class representatives with family size
   ``m = max(budget, n_trials) - n_behavioural_duplicates`` - every budgeted
   slot that did not repeat a tested hypothesis, so a run that stops early
   faces the same family as a full one.  Because feedback-driven proposers
   choose later expressions *after* seeing formation results, formation
   p-values are not valid for FDR control on their own.
2. *Validation confirmation* (the multiplicity step that carries the claim).
   Survivors are oriented by the sign of their formation IC and scored on the
   validation window, which no proposer ever saw.  BH (``confirm_method``) at
   ``confirm_alpha`` is applied to their one-sided validation p-values with
   ``m`` = number of candidates carried forward.
3. *Decorrelation*: confirmed candidates in descending validation-ICIR order
   are kept if their mean cross-sectional rank correlation with every kept
   factor is below ``max_abs_corr`` in absolute value, until ``top_k``.

Phase 3 - sealed test: the frozen set is committed to the ledger and revealed
at most once through :class:`~llm_factor_mining.protocol.seal.SealedHoldout`.
A run that ends with no trials, or because the proposer failed repeatedly,
is *aborted*: it is logged but neither committed nor revealed.

p-values refer the Newey-West t-statistic to a t distribution with ``n - 1``
degrees of freedom (a conservative small-sample convention).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import math
from pathlib import Path
import platform
from typing import Any, Union

import numpy as np
import pandas as pd
import scipy

from ._version import __version__
from .data import (
    Panel,
    PanelError,
    is_synthetic,
    panel_values_sha256,
    recompute_content_sha256,
    verify_panel_hash,
)
from .dsl.canonical import canonical_string, canonicalize, complexity, structural_hash
from .dsl.operators import describe_language
from .dsl.validate import DSLLimits, validate
from .evaluate.engine import Evaluator
from .evaluate.metrics import (
    EvaluationWindow,
    MetricConfig,
    embargoed_signal_dates,
    forward_returns,
    ic_summary,
    quantile_buckets,
    rank_ic,
    restrict,
    top_quantile_turnover,
)
from .inference import MultipleTestResult, deflated_sharpe_ratio, multiple_test, sample_moments, t_to_p
from .jsonutil import json_safe, sha256_json, write_json
from .protocol.ledger import TrialLedger
from .protocol.seal import FrozenFactor, FrozenFactorSet, RevealResult, SealedHoldout
from .protocol.splits import SplitError, Splits, truncate_panel
from .proposers.base import (
    EvaluatedFeedback,
    Proposal,
    ProposalContext,
    Proposer,
    ProposerError,
    RejectedFeedback,
)


PanelLoader = Callable[[Union[pd.Timestamp, None]], Panel]
"""``load(end)`` returns the study's panel up to ``end`` (``None`` = all data)."""

BEHAVIOUR_NAMESPACE = b"llm-factor-mining/behaviour-v1"
# What a proposer is told about a degenerate trial: a fixed text without the date counts
# that the ledger records, so no prompt carries the formation sample size.
DEGENERATE_FEEDBACK = "too few formation dates with a defined IC (lookback too long, or signal too sparse or constant)"


# --------------------------------------------------------------------------
# configuration and per-window statistics
# --------------------------------------------------------------------------


def default_max_rounds(budget: int, batch_size: int) -> int:
    """Round cap that scales with the budget: ``max(100, 10 * ceil(budget / batch_size))``.

    100 rounds at the default budget of 200 in batches of 20 (ten times the
    minimum number of rounds), and proportionally more for larger budgets, so
    that a compute-matched arm (``name@budget``) can use its whole budget.
    """

    if batch_size < 1:
        return 100
    return max(100, 10 * math.ceil(max(int(budget), 0) / int(batch_size)))


@dataclass(frozen=True)
class SearchConfig:
    """All knobs of one search run (hashed into the run record)."""

    budget: int = 200
    batch_size: int = 20
    max_rounds: int | None = None  # None: default_max_rounds(budget, batch_size)
    patience: int = 5
    max_proposer_failures: int = 3
    metric: MetricConfig = field(default_factory=MetricConfig)
    limits: DSLLimits = field(default_factory=DSLLimits)
    fdr_method: str = "bh"
    fdr_alpha: float = 0.10
    confirm_method: str = "bh"
    confirm_alpha: float = 0.10
    top_k: int = 5
    max_abs_corr: float = 0.7
    min_validation_icir: float = 0.0
    min_ic_dates: int = 100
    min_ic_fraction: float = 0.5
    behavioural_dedup: bool = True
    feedback_top: int = 8
    feedback_bottom: int = 4
    feedback_rejected: int = 8
    record_timestamps: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds is None:
            object.__setattr__(self, "max_rounds", default_max_rounds(self.budget, self.batch_size))
        if min(self.budget, self.batch_size, self.max_rounds, self.patience, self.top_k) < 1:
            raise ValueError("budget, batch_size, max_rounds, patience and top_k must be positive")
        for name in ("fdr_method", "confirm_method"):
            if getattr(self, name) not in ("bh", "by", "holm"):
                raise ValueError(f"{name} must be 'bh', 'by' or 'holm'")
        for name in ("fdr_alpha", "confirm_alpha"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} must lie in (0, 1)")
        if not 0.0 < self.max_abs_corr <= 1.0:
            raise ValueError("max_abs_corr must lie in (0, 1]")
        if self.min_ic_dates < 2 or not 0.0 <= self.min_ic_fraction <= 1.0:
            raise ValueError("min_ic_dates must be at least 2 and min_ic_fraction must lie in [0, 1]")

    def required_ic_dates(self, n_scorable: int) -> int:
        """Pre-registered minimum number of IC dates for a window with ``n_scorable`` dates."""

        return max(int(self.min_ic_dates), int(math.ceil(self.min_ic_fraction * n_scorable)))

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SearchConfig":
        """Inverse of :meth:`to_dict` (used to reload a run's configuration)."""

        values = dict(payload)
        values["metric"] = MetricConfig(**values["metric"])
        values["limits"] = DSLLimits(**values["limits"])
        return cls(**values)

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())


@dataclass(frozen=True)
class WindowStats:
    """IC statistics of one (oriented or raw) signal on one embargoed window."""

    n_ic_dates: int
    ic_mean: float
    ic_std: float
    icir: float
    t_nw: float
    p_value: float
    turnover: float
    ic_skew: float
    ic_kurtosis: float
    n_scorable_dates: int = 0

    def to_dict(self) -> dict[str, Any]:
        return json_safe(asdict(self))


def nw_p_value(t_stat: float, n: int, *, alternative: str = "two-sided") -> float:
    """p-value of a Newey-West t-statistic against t(n - 1); ``NaN`` when undefined."""

    if n < 2:
        return float("nan")
    return t_to_p(t_stat, df=n - 1, alternative=alternative)  # type: ignore[arg-type]


def window_statistics(
    signal: pd.DataFrame,
    forward: pd.DataFrame,
    window: EvaluationWindow,
    config: MetricConfig,
) -> tuple[WindowStats, pd.Series]:
    """Rank-IC statistics and top-quantile turnover on ``window``'s embargoed dates.

    Numerically identical to the corresponding fields of
    :func:`llm_factor_mining.evaluate.metrics.evaluate_signal`, but skips the
    quantile-return and portfolio computations (``forward`` is precomputed).
    ``p_value`` is two-sided against t(n - 1).
    """

    calendar = pd.DatetimeIndex(signal.index)
    dates = embargoed_signal_dates(calendar, window, lag=config.lag, horizon=config.horizon)
    min_names = config.effective_min_names
    if len(dates) == 0:
        nan = float("nan")
        return WindowStats(0, nan, nan, nan, nan, nan, nan, nan, nan, 0), pd.Series(dtype="float64")
    ic = rank_ic(signal.loc[dates], forward.loc[dates], min_names=min_names).dropna()
    summary = ic_summary(ic, horizon=config.horizon, nw_lags=config.nw_lags)
    first = int(calendar.get_loc(dates[0]))
    last = int(calendar.get_loc(dates[-1]))
    span = signal.iloc[max(first - 1, 0) : last + 1]
    buckets = quantile_buckets(span, config.n_quantiles, min_names=min_names)
    turnover = restrict(top_quantile_turnover(buckets, config.n_quantiles), dates)
    _, skew, kurt, _ = sample_moments(ic.to_numpy())
    stats = WindowStats(
        n_ic_dates=summary.n,
        ic_mean=summary.mean,
        ic_std=summary.std,
        icir=summary.icir,
        t_nw=summary.t_nw,
        p_value=nw_p_value(summary.t_nw, summary.n),
        turnover=float(turnover.mean()) if turnover.notna().any() else float("nan"),
        ic_skew=skew,
        ic_kurtosis=kurt,
        n_scorable_dates=int(len(dates)),
    )
    return stats, ic


def window_capacity_problems(
    calendar: pd.DatetimeIndex,
    splits: Splits,
    config: SearchConfig,
    *,
    n_symbols: int | None = None,
    windows: tuple[str, ...] = ("formation", "validation", "test"),
) -> list[str]:
    """Reasons why ``splits`` on ``calendar`` can never yield a usable p-value.

    A window with fewer than ``config.min_ic_dates`` scorable (embargoed)
    signal dates makes every trial degenerate (formation), every candidate
    degenerate (validation) or the sealed test uninformative (test); fewer
    than ``metric.effective_min_names`` symbols leave every rank IC undefined.
    Such a study would look like an honest null result, so callers refuse it
    before any window is fixed or any trial is spent.
    """

    metric = config.metric
    dates = pd.DatetimeIndex(calendar)
    problems = []
    for window in splits.windows():
        if window.name not in windows:
            continue
        n_scorable = len(embargoed_signal_dates(dates, window, lag=metric.lag, horizon=metric.horizon))
        if n_scorable < config.min_ic_dates:
            problems.append(
                f"the {window.name} window has {n_scorable} scorable signal dates, fewer than "
                f"min_ic_dates = {config.min_ic_dates}, so no IC series on it can be long enough"
            )
    if n_symbols is not None and n_symbols < metric.effective_min_names:
        problems.append(
            f"the panel has {n_symbols} symbols, fewer than the {metric.effective_min_names} names a "
            "cross-sectional rank IC needs on each date"
        )
    return problems


def _require_capacity(problems: list[str]) -> None:
    if problems:
        raise SplitError(
            "these windows cannot select anything (use a longer date range or more symbols): "
            + "; ".join(problems)
        )


def behaviour_signature(signal: pd.DataFrame, dates: pd.DatetimeIndex) -> str:
    """Hash of the sign-normalized per-date cross-sectional ranks on ``dates``.

    Two signals with the same signature have identical per-date rank IC (up to
    sign) against any forward return, hence identical two-sided p-values: they
    test the same hypothesis.  (``volume``, ``log(volume)``, ``cs_rank(volume)``
    and ``-volume`` all share one signature.)
    """

    ranks = signal.loc[dates].rank(axis=1, method="average").to_numpy(dtype="float64")
    counts = np.isfinite(ranks).sum(axis=1, keepdims=True).astype("float64")
    doubled = 2.0 * ranks - (counts + 1.0)  # centred ranks times two: exact integers
    missing = ~np.isfinite(doubled)
    values = np.where(missing, 0.0, np.rint(doubled)).astype("<i8")
    flat = values.ravel()
    nonzero = np.flatnonzero(flat)
    if nonzero.size and flat[nonzero[0]] < 0:
        values = -values
    digest = hashlib.sha256(BEHAVIOUR_NAMESPACE)
    digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
    digest.update(np.packbits(missing.ravel()).tobytes())
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def mean_rank_correlation(left: pd.DataFrame, right: pd.DataFrame, *, min_names: int = 10) -> float:
    """Average over dates of the cross-sectional Spearman correlation of two signals."""

    series = rank_ic(left, right, min_names=min_names).dropna()
    return float(series.mean()) if len(series) else float("nan")


def grammar_card(limits: DSLLimits) -> str:
    """Grammar reference plus the active search-space limits (sent to proposers)."""

    return (
        describe_language()
        + f"\nLimits: depth <= {limits.max_depth}, nodes <= {limits.max_nodes}, "
        f"windows <= {limits.max_window} sessions, cumulative lookback <= {limits.max_lookback} "
        f"sessions, |literal| <= {limits.max_abs_literal:g}."
    )


# --------------------------------------------------------------------------
# trial records and results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial:
    """One non-duplicate processed proposal."""

    index: int
    round_index: int
    status: str  # "evaluated" | "degenerate" | "invalid" | "error"
    expression: str
    canonical: str | None = None
    expression_hash: str | None = None
    codes: tuple[str, ...] = ()
    message: str = ""
    complexity: float | None = None
    formation: WindowStats | None = None
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    behaviour_signature: str | None = None
    behaviour_class: int | None = None

    @property
    def is_class_representative(self) -> bool:
        return self.status == "evaluated" and self.behaviour_class in (None, self.index)

    def to_payload(self) -> dict[str, Any]:
        return json_safe(
            {
                "trial_index": self.index,
                "round": self.round_index,
                "status": self.status,
                "expression": self.expression[:2000],
                "canonical": self.canonical,
                "expression_hash": self.expression_hash,
                "codes": list(self.codes),
                "message": self.message[:500],
                "complexity": self.complexity,
                "formation": None if self.formation is None else self.formation.to_dict(),
                "behaviour_signature": self.behaviour_signature,
                "behaviour_class": self.behaviour_class,
                "rationale": self.rationale[:500],
                "proposal_metadata": dict(self.metadata),
            }
        )

    def feedback(self) -> EvaluatedFeedback:
        assert self.status == "evaluated" and self.formation is not None and self.canonical
        return EvaluatedFeedback(
            expression=self.canonical,
            formation_ic=self.formation.ic_mean,
            formation_icir=self.formation.icir,
            formation_tstat=self.formation.t_nw,
            formation_turnover=self.formation.turnover,
            complexity=float(self.complexity or 0.0),
        )


@dataclass(frozen=True)
class SearchResult:
    """Everything a run produced; ``summary()`` is the JSON headline.

    ``aborted`` runs (no trials, or repeated proposer failures) carry no
    selection, commitment or reveal.
    """

    config: SearchConfig
    splits: Splits
    trials: tuple[Trial, ...]
    n_proposals: int
    n_valid_proposals: int
    n_duplicates: int
    n_rounds: int
    n_proposer_errors: int
    stop_reason: str
    multiple_test: MultipleTestResult | None
    confirmation: MultipleTestResult | None
    funnel: Mapping[str, int]
    selection: tuple[Mapping[str, Any], ...]
    frozen: FrozenFactorSet
    commitment: str | None
    reveal: RevealResult | None
    ledger: TrialLedger
    proposer: Mapping[str, Any]
    proposer_provenance: Mapping[str, Any]
    data_sha256: str | None = None
    aborted: bool = False
    run_dir: Path | None = None

    @property
    def n_trials(self) -> int:
        return len(self.trials)

    def _count(self, status: str) -> int:
        return sum(1 for trial in self.trials if trial.status == status)

    @property
    def n_evaluated(self) -> int:
        return self._count("evaluated")

    @property
    def n_degenerate(self) -> int:
        return self._count("degenerate")

    @property
    def n_invalid(self) -> int:
        return self._count("invalid")

    @property
    def n_errors(self) -> int:
        return self._count("error")

    @property
    def complete(self) -> bool:
        """True when the run used its whole budget and was committed."""

        return not self.aborted and self.stop_reason == "budget"

    @property
    def validity_rate(self) -> float:
        return self.n_valid_proposals / self.n_proposals if self.n_proposals else float("nan")

    @property
    def ledger_head(self) -> str:
        return self.ledger.head_hash

    def provenance_summary(self) -> dict[str, Any]:
        """Proposer provenance without the per-call log (served models, usage, stop reasons)."""

        return json_safe({key: value for key, value in self.proposer_provenance.items() if key != "calls"})

    def warnings(self) -> list[str]:
        """Plain-language reasons why an empty or short selection may not be a null result."""

        notes = []
        degenerate = int(self.funnel.get("n_rejected_degenerate_validation", 0) or 0)
        if degenerate:
            notes.append(
                f"{degenerate} screened candidate(s) had too few validation dates with a defined IC and were "
                "rejected as degenerate: they were never tested on validation"
            )
        if not self.aborted and self.stop_reason != "budget":
            notes.append(f"incomplete run: stopped by {self.stop_reason!r} before the budget was used")
        return notes

    def summary(self) -> dict[str, Any]:
        tested = self.multiple_test
        confirm = self.confirmation
        return json_safe(
            {
                "proposer": self.proposer.get("name"),
                "warnings": self.warnings(),
                "n_trials": self.n_trials,
                "n_evaluated": self.n_evaluated,
                "n_degenerate": self.n_degenerate,
                "n_invalid": self.n_invalid,
                "n_errors": self.n_errors,
                "n_duplicates": self.n_duplicates,
                "n_proposals": self.n_proposals,
                "validity_rate": self.validity_rate,
                "n_rounds": self.n_rounds,
                "n_proposer_errors": self.n_proposer_errors,
                "stop_reason": self.stop_reason,
                "aborted": self.aborted,
                "complete": self.complete,
                "screen": None
                if tested is None
                else {"method": tested.method, "alpha": tested.alpha, "m": tested.m, "n_survivors": tested.n_rejected},
                "confirmation": None
                if confirm is None
                else {"method": confirm.method, "alpha": confirm.alpha, "m": confirm.m, "n_confirmed": confirm.n_rejected},
                "funnel": dict(self.funnel),
                "selected": [factor.to_dict() for factor in self.frozen.factors],
                "commitment": self.commitment,
                "data_sha256": self.data_sha256,
                "ledger_head": self.ledger_head,
                "proposer_provenance": self.provenance_summary(),
            }
        )


# --------------------------------------------------------------------------
# the search loop
# --------------------------------------------------------------------------


def _ranked_feedback(trials: list[Trial]) -> list[EvaluatedFeedback]:
    usable = [t for t in trials if t.status == "evaluated" and t.formation and math.isfinite(t.formation.icir)]
    usable.sort(key=lambda t: (-abs(t.formation.icir), t.expression_hash))  # type: ignore[union-attr]
    return [t.feedback() for t in usable]


def software_versions() -> dict[str, str]:
    return {
        "llm_factor_mining": __version__,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
    }


def _describe(proposer: Proposer) -> dict[str, Any]:
    describe = getattr(proposer, "describe", None)
    info = describe() if callable(describe) else {}
    return json_safe({"name": getattr(proposer, "name", type(proposer).__name__), **info})


def _loader(panel: Panel | PanelLoader) -> tuple[PanelLoader, Panel | None]:
    if isinstance(panel, Panel):
        full = panel

        def load(end: pd.Timestamp | None) -> Panel:
            return full if end is None else truncate_panel(full, end)

        return load, full
    if not callable(panel):
        raise TypeError("run_search needs a Panel or a loader callable load(end) -> Panel")
    return panel, None


def _check_prefix(longer: Panel, shorter: Panel, end: pd.Timestamp, what: str) -> None:
    # Values, dates and symbols only: the source labels of a longer load may include a
    # vendor that supplies bars only after ``end``, which is not a change of the data.
    if panel_values_sha256(truncate_panel(longer, end)) != panel_values_sha256(shorter):
        raise PanelError(f"the data changed between phases: the {what} panel does not extend the earlier one")


def run_search(
    proposer: Proposer,
    panel: Panel | PanelLoader,
    splits: Splits,
    config: SearchConfig | None = None,
    *,
    run_dir: str | Path | None = None,
    reveal: bool = True,
    authorize: Callable[[Mapping[str, Any]], None] | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> SearchResult:
    """Run search, selection, the commitment and (optionally) the sealed reveal.

    ``panel`` is either the full panel or a loader ``load(end)``.  With a
    loader, each phase reads only the data it may use: the search loop the
    formation window, selection the validation window, and the test window is
    loaded only after the factor set is frozen.  (With a full panel the whole
    history is in the caller's memory; the harness still passes proposers
    nothing but formation statistics.)

    Revealing a non-synthetic panel requires a file-backed ledger
    (``run_dir``) and an external reveal register (``authorize``).
    """

    config = config or SearchConfig()
    metric = config.metric
    if splits.lag != metric.lag or splits.max_horizon < metric.horizon:
        raise SplitError("splits were built for a different lag or a shorter horizon than the metric config")
    load, full_panel = _loader(panel)
    start_sha256: str | None = None
    if full_panel is not None:
        splits.check_calendar(full_panel.dates)
        start_sha256 = verify_panel_hash(full_panel)
        _require_capacity(window_capacity_problems(full_panel.dates, splits, config, n_symbols=len(full_panel.symbols)))
    formation_panel = load(splits.formation.end)
    if formation_panel.dates[-1] > splits.formation.end:
        raise SplitError("the loader returned data beyond the formation window")
    if full_panel is None:
        # later windows are not loaded yet; the study registry checks them when it fixes the windows
        _require_capacity(
            window_capacity_problems(
                formation_panel.dates, splits, config, n_symbols=len(formation_panel.symbols), windows=("formation",)
            )
        )
    if reveal and not is_synthetic(formation_panel):
        if run_dir is None or authorize is None:
            raise ValueError(
                "revealing the test window of non-synthetic data requires a run directory (file-backed "
                "ledger) and a reveal register (authorize=...); commit with reveal=False and reveal later"
            )
    run_path = None if run_dir is None else Path(run_dir)
    if run_path is not None:
        if (run_path / "ledger.jsonl").exists():
            raise FileExistsError(f"{run_path} already holds a ledger; use a fresh run directory")
        run_path.mkdir(parents=True, exist_ok=True)
    ledger = TrialLedger(
        None if run_path is None else run_path / "ledger.jsonl", timestamps=config.record_timestamps
    )
    proposer_info = _describe(proposer)

    run_record = {
        "config": config.to_dict(),
        "config_sha256": config.sha256,
        "splits": splits.to_dict(),
        "proposer": proposer_info,
        "data_sha256": start_sha256,
        "formation_panel_sha256": formation_panel.content_sha256,
        "data_sources": list(formation_panel.sources),
        "software": software_versions(),
        "run_metadata": json_safe(dict(run_metadata or {})),
    }
    ledger.append("run_start", run_record)
    if run_path is not None:
        write_json(run_path / "config.json", run_record)

    evaluator = Evaluator(formation_panel, limits=config.limits)
    forward = forward_returns(formation_panel.close, metric.horizon, metric.lag)
    formation_dates = embargoed_signal_dates(
        formation_panel.dates, splits.formation, lag=metric.lag, horizon=metric.horizon
    )
    grammar = grammar_card(config.limits)

    trials: list[Trial] = []
    by_hash: dict[str, Trial] = {}
    by_behaviour: dict[str, int] = {}
    seen_invalid: set[str] = set()
    last_round: list[EvaluatedFeedback] = []
    recent_rejected: list[RejectedFeedback] = []
    n_proposals = n_valid = n_duplicates = n_proposer_errors = 0
    round_index = idle_rounds = consecutive_failures = 0
    stop_reason = "budget"

    while len(trials) < config.budget:
        if round_index >= config.max_rounds:
            stop_reason = "max_rounds"
            break
        ranked = _ranked_feedback(trials)
        remaining = config.budget - len(trials)
        context = ProposalContext(
            round_index=round_index,
            n_requested=min(config.batch_size, remaining),
            grammar=grammar,
            top=tuple(ranked[: config.feedback_top]),
            bottom=tuple(ranked[::-1][: config.feedback_bottom]) if len(ranked) > config.feedback_top else (),
            last_round=tuple(last_round),
            rejected=tuple(recent_rejected[-config.feedback_rejected :]),
            n_trials_so_far=len(trials),
            budget_remaining=remaining,
        )
        try:
            proposals = list(proposer.propose(context))
        except ProposerError as exc:
            n_proposer_errors += 1
            consecutive_failures += 1
            ledger.append(
                "proposer_error",
                {
                    "round": round_index,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:1000],
                    "metadata": json_safe(getattr(exc, "metadata", {})),
                },
            )
            round_index += 1
            if consecutive_failures >= config.max_proposer_failures:
                stop_reason = "proposer_failures"
                break
            continue
        consecutive_failures = 0

        new_trials = 0
        last_round = []
        for proposal in proposals:
            if len(trials) >= config.budget:
                break
            if not isinstance(proposal, Proposal):
                raise TypeError(f"proposer returned {type(proposal).__name__}, expected Proposal")
            n_proposals += 1
            text = proposal.expression
            result = validate(text, config.limits)
            base = {
                "index": len(trials),
                "round_index": round_index,
                "expression": text,
                "rationale": proposal.rationale,
                "metadata": dict(proposal.metadata),
            }
            if not result.ok or result.node is None:
                key = " ".join(text.split()).lower()
                if key in seen_invalid:
                    n_duplicates += 1
                    ledger.append("duplicate", {"round": round_index, "expression": text[:2000], "reason": "repeated invalid text"})
                    continue
                seen_invalid.add(key)
                first = result.errors[0] if result.errors else None
                trial = Trial(
                    status="invalid",
                    codes=result.codes,
                    message="" if first is None else first.message,
                    **base,
                )
                recent_rejected.append(RejectedFeedback(text, result.codes, trial.message))
            else:
                n_valid += 1
                node = canonicalize(result.node)
                digest = structural_hash(node)
                if digest in by_hash:
                    n_duplicates += 1
                    ledger.append(
                        "duplicate",
                        {
                            "round": round_index,
                            "expression": text[:2000],
                            "expression_hash": digest,
                            "duplicate_of": by_hash[digest].index,
                        },
                    )
                    continue
                canonical = canonical_string(node)
                try:
                    signal = evaluator.evaluate(node)
                    stats, _ = window_statistics(signal, forward, splits.formation, metric)
                    required = config.required_ic_dates(stats.n_scorable_dates)
                    if stats.n_ic_dates < required:
                        trial = Trial(
                            status="degenerate",
                            canonical=canonical,
                            expression_hash=digest,
                            codes=("DEGENERATE",),
                            message=(
                                f"only {stats.n_ic_dates} of {stats.n_scorable_dates} formation dates have a "
                                f"defined IC; at least {required} are required"
                            ),
                            complexity=complexity(node),
                            formation=stats,
                            **base,
                        )
                        # the ledger keeps the counts; proposers get a count-free text (no sample size)
                        recent_rejected.append(RejectedFeedback(canonical, trial.codes, DEGENERATE_FEEDBACK))
                    else:
                        signature = behaviour_signature(signal, formation_dates) if config.behavioural_dedup else None
                        representative = (
                            by_behaviour.setdefault(signature, len(trials)) if signature is not None else None
                        )
                        trial = Trial(
                            status="evaluated",
                            canonical=canonical,
                            expression_hash=digest,
                            complexity=complexity(node),
                            formation=stats,
                            behaviour_signature=signature,
                            behaviour_class=representative,
                            **base,
                        )
                        last_round.append(trial.feedback())
                except Exception as exc:  # noqa: BLE001 - numerical failures are logged trials
                    trial = Trial(
                        status="error",
                        canonical=canonical,
                        expression_hash=digest,
                        codes=("EVAL_ERROR",),
                        message=f"{type(exc).__name__}: {exc}",
                        complexity=complexity(node),
                        **base,
                    )
                    # raw exception text can carry shapes or values of the panel: send the type only
                    recent_rejected.append(
                        RejectedFeedback(canonical, ("EVAL_ERROR",), f"evaluation failed ({type(exc).__name__})")
                    )
                by_hash[digest] = trial
            trials.append(trial)
            new_trials += 1
            ledger.append("trial", trial.to_payload())

        recent_rejected = recent_rejected[-config.feedback_rejected :]
        ledger.append(
            "round_end",
            {"round": round_index, "n_proposals": len(proposals), "new_trials": new_trials, "n_trials": len(trials)},
        )
        round_index += 1
        idle_rounds = idle_rounds + 1 if new_trials == 0 else 0
        if idle_rounds >= config.patience:
            stop_reason = "stalled"
            break

    provenance_info = getattr(proposer, "provenance", None)
    proposer_provenance = json_safe(provenance_info()) if callable(provenance_info) else {}
    counts = {
        "n_trials": len(trials),
        "n_evaluated": sum(1 for t in trials if t.status == "evaluated"),
        "n_degenerate": sum(1 for t in trials if t.status == "degenerate"),
        "n_invalid": sum(1 for t in trials if t.status == "invalid"),
        "n_errors": sum(1 for t in trials if t.status == "error"),
    }
    aborted = stop_reason == "proposer_failures" or not trials
    common = {
        "config": config,
        "splits": splits,
        "trials": tuple(trials),
        "n_proposals": n_proposals,
        "n_valid_proposals": n_valid,
        "n_duplicates": n_duplicates,
        "n_rounds": round_index,
        "n_proposer_errors": n_proposer_errors,
        "stop_reason": stop_reason,
        "ledger": ledger,
        "proposer": proposer_info,
        "proposer_provenance": proposer_provenance,
        "run_dir": run_path,
    }
    if aborted:
        ledger.append(
            "run_aborted",
            {
                **counts,
                "stop_reason": stop_reason,
                "n_proposer_errors": n_proposer_errors,
                "reason": "no trials" if not trials else "repeated proposer failures",
                "served_models": proposer_provenance.get("served_models", []),
            },
        )
        result = SearchResult(
            multiple_test=None,
            confirmation=None,
            funnel=counts,
            selection=(),
            frozen=FrozenFactorSet(()),
            commitment=None,
            reveal=None,
            data_sha256=start_sha256,
            aborted=True,
            **common,
        )
        _write_artifacts(result, run_path, formation_panel, None, None)
        return result

    # validation data enter only here, after the search loop has finished
    validation_panel = load(splits.validation.end)
    _check_prefix(validation_panel, formation_panel, splits.formation.end, "validation")
    tested, confirm, selection, frozen, funnel = _select(trials, validation_panel, splits, config)
    funnel = {**counts, **funnel}
    ledger.append(
        "selection",
        {
            "screen": _mt_dict(tested),
            "confirmation": _mt_dict(confirm),
            "funnel": funnel,
            "candidates": list(selection),
            "frozen": frozen.to_dict(),
            "frozen_sha256": frozen.sha256,
        },
    )

    # the test window is loaded only now that the factor set is frozen
    full = load(None)
    if full_panel is None:
        splits.check_calendar(full.dates)
        _check_prefix(full, validation_panel, splits.validation.end, "full")
    data_sha256 = recompute_content_sha256(full)
    holdout = SealedHoldout(full, splits.test, metric, ledger, limits=config.limits)
    commitment = holdout.commit(
        frozen,
        data_sha256=data_sha256,
        context={"n_trials": len(trials), "config_sha256": config.sha256, "funnel": funnel},
    )
    if run_path is not None:
        write_json(
            run_path / "selected.json",
            {
                "frozen": frozen.to_dict(),
                "frozen_sha256": frozen.sha256,
                "commitment": commitment,
                "seal_id": holdout.seal_id,
                "guard_id": holdout.guard_id,
                "data_sha256": data_sha256,
                "screen": _mt_dict(tested),
                "confirmation": _mt_dict(confirm),
                "funnel": funnel,
                "candidates": list(selection),
            },
        )
    commit_head = ledger.head_hash
    revealed = holdout.reveal(frozen, authorize=authorize) if reveal else None

    ledger.append(
        "run_end",
        {
            "n_trials": len(trials),
            "n_duplicates": n_duplicates,
            "n_proposals": n_proposals,
            "n_rounds": round_index,
            "stop_reason": stop_reason,
            "revealed": revealed is not None,
            "served_models": proposer_provenance.get("served_models", []),
        },
    )
    result = SearchResult(
        multiple_test=tested,
        confirmation=confirm,
        funnel=funnel,
        selection=tuple(selection),
        frozen=frozen,
        commitment=commitment,
        reveal=revealed,
        data_sha256=data_sha256,
        **common,
    )
    _write_artifacts(result, run_path, formation_panel, validation_panel, commit_head, full)
    return result


def _mt_dict(result: MultipleTestResult) -> dict[str, Any]:
    return {
        "method": result.method,
        "alpha": result.alpha,
        "m": result.m,
        "n_tested": int(result.rejected.size),
        "n_rejected": result.n_rejected,
    }


def _write_artifacts(
    result: SearchResult,
    run_path: Path | None,
    formation_panel: Panel,
    validation_panel: Panel | None,
    commit_head: str | None,
    full: Panel | None = None,
) -> None:
    if run_path is None:
        return
    if result.reveal is not None:
        write_json(run_path / "reveal.json", result.reveal.payload)
    write_json(
        run_path / "provenance.json",
        {
            "data": None if full is None else full.provenance(),
            "formation_panel_sha256": formation_panel.content_sha256,
            "validation_panel_sha256": None if validation_panel is None else validation_panel.content_sha256,
            "proposer": result.proposer,
            "proposer_provenance": result.provenance_summary(),
            "software": software_versions(),
            "commit_ledger_head": commit_head,
            "ledger_head": result.ledger.head_hash,
            "summary": result.summary(),
        },
    )


def _select(
    trials: list[Trial],
    validation_panel: Panel,
    splits: Splits,
    config: SearchConfig,
) -> tuple[MultipleTestResult, MultipleTestResult, list[dict[str, Any]], FrozenFactorSet, dict[str, int]]:
    metric = config.metric
    evaluated = [trial for trial in trials if trial.status == "evaluated" and trial.formation is not None]
    representatives = [trial for trial in evaluated if trial.is_class_representative]
    n_behaviour_duplicates = len(evaluated) - len(representatives)
    m_screen = max(config.budget, len(trials)) - n_behaviour_duplicates
    pvalues = [trial.formation.p_value for trial in representatives]  # type: ignore[union-attr]
    tested = multiple_test(pvalues, config.fdr_alpha, method=config.fdr_method, m=m_screen)  # type: ignore[arg-type]
    icirs = np.array([trial.formation.icir for trial in representatives], dtype="float64")  # type: ignore[union-attr]
    icirs = icirs[np.isfinite(icirs)]
    sharpe_variance = float(icirs.var(ddof=1)) if icirs.size >= 2 else float("nan")
    n_hypotheses = max(m_screen, 1)

    evaluator = Evaluator(validation_panel, limits=config.limits)
    forward = forward_returns(validation_panel.close, metric.horizon, metric.lag)
    candidates: list[dict[str, Any]] = []
    for trial, survived, adjusted in zip(representatives, tested.rejected, tested.adjusted):
        if not survived:
            continue
        stats = trial.formation
        assert stats is not None and trial.canonical is not None
        orientation = 1 if not (stats.ic_mean < 0) else -1
        signal = evaluator.evaluate(trial.canonical) * float(orientation)
        vstats, _ = window_statistics(signal, forward, splits.validation, metric)
        degenerate = vstats.n_ic_dates < config.required_ic_dates(vstats.n_scorable_dates)
        p_one_sided = 1.0 if degenerate else nw_p_value(vstats.t_nw, vstats.n_ic_dates, alternative="greater")
        dsr = float("nan")
        if math.isfinite(sharpe_variance) and math.isfinite(stats.icir):
            # |ICIR| is a two-sided selection: the null maximum of N absolute values behaves
            # like the maximum of 2N signed values, hence the 2N trials.
            dsr = deflated_sharpe_ratio(
                abs(stats.icir),
                stats.n_ic_dates,
                n_trials=2 * n_hypotheses,
                sharpe_variance=sharpe_variance,
                skew=orientation * stats.ic_skew if math.isfinite(stats.ic_skew) else 0.0,
                kurtosis=stats.ic_kurtosis if math.isfinite(stats.ic_kurtosis) else 3.0,
            )
        candidates.append(
            {
                "trial_index": trial.index,
                "canonical": trial.canonical,
                "expression_hash": trial.expression_hash,
                "orientation": orientation,
                "formation_icir": stats.icir,
                "formation_t": stats.t_nw,
                "formation_p": stats.p_value,
                "formation_p_adjusted": float(adjusted),
                "formation_deflated_icir": dsr,
                "validation": vstats.to_dict(),
                "validation_degenerate": degenerate,
                "validation_p_one_sided": p_one_sided,
                "_validation_icir": vstats.icir,
            }
        )
    confirm = multiple_test(
        [row["validation_p_one_sided"] for row in candidates],
        config.confirm_alpha,
        method=config.confirm_method,  # type: ignore[arg-type]
        m=len(candidates),
    )
    for row, confirmed, adjusted in zip(candidates, confirm.rejected, confirm.adjusted):
        row["validation_confirmed"] = bool(confirmed)
        row["validation_p_adjusted"] = float(adjusted)

    def order_key(row: dict[str, Any]) -> tuple[float, str]:
        value = row["_validation_icir"]
        return (-value if math.isfinite(value) else math.inf, row["expression_hash"])

    candidates.sort(key=order_key)
    kept: list[tuple[dict[str, Any], pd.DataFrame]] = []
    window_dates = validation_panel.dates[
        (validation_panel.dates >= splits.validation.start) & (validation_panel.dates <= splits.validation.end)
    ]
    for row in candidates:
        value = row.pop("_validation_icir")
        if row["validation_degenerate"]:
            row["decision"] = "rejected_degenerate_validation"
            continue
        if not row["validation_confirmed"] or not math.isfinite(value) or value <= config.min_validation_icir:
            row["decision"] = "rejected_validation"
            continue
        if len(kept) >= config.top_k:
            row["decision"] = "not_needed"
            continue
        signal = (evaluator.evaluate(row["canonical"]) * float(row["orientation"])).loc[window_dates]
        worst, partner = 0.0, None
        for other, other_signal in kept:
            corr = mean_rank_correlation(signal, other_signal, min_names=metric.effective_min_names)
            if math.isfinite(corr) and abs(corr) > worst:
                worst, partner = abs(corr), other["expression_hash"]
        row["max_abs_corr_with_kept"] = worst
        row["most_correlated_with"] = partner
        if worst >= config.max_abs_corr:
            row["decision"] = "rejected_correlation"
            continue
        row["decision"] = "selected"
        kept.append((row, signal))
    frozen = FrozenFactorSet(
        tuple(
            FrozenFactor(row["canonical"], row["expression_hash"], int(row["orientation"]))
            for row, _ in kept
        )
    )
    decisions = [row["decision"] for row in candidates]
    funnel = {
        "n_behaviour_classes": len(representatives),
        "n_behaviour_duplicates": n_behaviour_duplicates,
        "m_screen": int(m_screen),
        "n_screen_survivors": tested.n_rejected,
        "n_confirmed": confirm.n_rejected,
        "n_rejected_validation": decisions.count("rejected_validation"),
        "n_rejected_degenerate_validation": decisions.count("rejected_degenerate_validation"),
        "n_rejected_correlation": decisions.count("rejected_correlation"),
        "n_not_needed": decisions.count("not_needed"),
        "n_selected": len(kept),
    }
    return tested, confirm, [json_safe(row) for row in candidates], frozen, funnel


__all__ = [
    "PanelLoader",
    "SearchConfig",
    "SearchResult",
    "Trial",
    "WindowStats",
    "behaviour_signature",
    "default_max_rounds",
    "grammar_card",
    "mean_rank_correlation",
    "nw_p_value",
    "run_search",
    "software_versions",
    "window_capacity_problems",
    "window_statistics",
]
