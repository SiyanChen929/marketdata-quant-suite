"""Sealed hold-out: commit to a frozen factor set, then reveal test metrics once.

Protocol:

1. the search freezes a :class:`FrozenFactorSet` using formation and
   validation data only;
2. :meth:`SealedHoldout.commit` writes a SHA-256 commitment over the factor
   set, the metric configuration, the DSL limits, the test window and the
   panel content hash to the trial ledger *before* any test metric exists;
3. :meth:`SealedHoldout.reveal` recomputes the commitment for the factor set it
   is handed, refuses if it differs from the committed one, refuses a second
   reveal of the same test window on the same data (also across instances
   and writers sharing the ledger, whatever the metric settings), asks an
   optional external authority (the project-wide reveal register), and only
   then computes and logs the test-window metrics.

Refused reveals of a committed seal (a second reveal, a different factor set
or seal terms, a register refusal) are logged too, so the ledger shows every
such attempt; a reveal before any commitment, and refused commits, raise
:class:`SealError` without a ledger record.

What this does *not* prove on its own: commit and reveal records live in a
ledger written by the same process.  A third party can check that a reveal
matched a commitment made before it only if the commitment hash was published
outside the run directory before the reveal (the CLI's two-step
``search`` -> ``reveal`` flow supports exactly that).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..data import Panel, PanelError, verify_panel_hash
from ..dsl.canonical import canonical_string, structural_hash
from ..dsl.parser import parse
from ..dsl.validate import DSLLimits
from ..evaluate.engine import Evaluator
from ..evaluate.engine import op_cs_zscore
from ..evaluate.metrics import EvaluationWindow, FactorEvaluation, MetricConfig, evaluate_signal
from ..jsonutil import json_safe, sha256_json
from .ledger import TrialLedger


COMMIT_KIND = "seal_commit"
REVEAL_KIND = "seal_reveal"
REFUSED_KIND = "seal_refused"
COMPOSITE_NAME = "composite"


class SealError(RuntimeError):
    """A commit or reveal violated the sealed hold-out protocol."""


@dataclass(frozen=True)
class FrozenFactor:
    """One selected factor: canonical expression plus its frozen sign."""

    expression: str
    expression_hash: str
    orientation: int = 1

    def __post_init__(self) -> None:
        if self.orientation not in (1, -1):
            raise ValueError("orientation must be +1 or -1")
        node = parse(self.expression)
        if canonical_string(node) != self.expression:
            raise ValueError("FrozenFactor.expression must be a canonical expression string")
        if structural_hash(node) != self.expression_hash:
            raise ValueError("expression_hash does not match the expression")

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.expression,
            "expression_hash": self.expression_hash,
            "orientation": self.orientation,
        }


@dataclass(frozen=True)
class FrozenFactorSet:
    """Ordered, immutable set of selected factors (order = selection rank)."""

    factors: tuple[FrozenFactor, ...]
    label: str = "selected"

    def __post_init__(self) -> None:
        object.__setattr__(self, "factors", tuple(self.factors))
        hashes = [factor.expression_hash for factor in self.factors]
        if len(hashes) != len(set(hashes)):
            raise ValueError("a frozen factor set cannot contain the same expression twice")

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "factors": [factor.to_dict() for factor in self.factors]}

    @property
    def sha256(self) -> str:
        return sha256_json(self.to_dict())

    def __len__(self) -> int:
        return len(self.factors)


@dataclass(frozen=True)
class RevealResult:
    """Logged reveal payload plus in-memory series for downstream scoring."""

    payload: Mapping[str, Any]
    evaluations: Mapping[str, FactorEvaluation] = field(default_factory=dict)
    signals: Mapping[str, pd.DataFrame] = field(default_factory=dict)


def oriented_signal(evaluator: Evaluator, factor: FrozenFactor) -> pd.DataFrame:
    """Evaluate ``factor`` and apply its frozen orientation."""

    return evaluator.evaluate(factor.expression) * float(factor.orientation)


def composite_signal(signals: list[pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight mean of per-date z-scored signals (NaN-aware)."""

    if not signals:
        raise ValueError("composite of an empty factor set")
    stacked = np.stack([op_cs_zscore(signal).to_numpy() for signal in signals])
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(stacked).sum(axis=0)
        total = np.nansum(stacked, axis=0)
        values = np.where(counts > 0, total / np.maximum(counts, 1), np.nan)
    reference = signals[0]
    return pd.DataFrame(values, index=reference.index, columns=reference.columns)


class SealedHoldout:
    """Single-use gate in front of the test window.

    The one-commit / one-reveal rule is keyed on :attr:`guard_id`, a hash of
    the panel content hash and the test-window dates only: changing the
    metric settings or the language limits yields a different :attr:`seal_id`
    (and commitment) but not a second reveal of the same window on the same
    data.  The ledger is re-scanned on every :meth:`commit` and
    :meth:`reveal` (after :meth:`TrialLedger.sync`), so several instances, or
    several writers of one ledger file, share the rule.  Across *separate*
    ledgers the rule is enforced by an external reveal register passed as
    ``authorize`` (see :mod:`llm_factor_mining.protocol.registry`).
    """

    def __init__(
        self,
        panel: Panel,
        window: EvaluationWindow,
        config: MetricConfig,
        ledger: TrialLedger,
        *,
        limits: DSLLimits | None = None,
    ) -> None:
        self._panel = panel
        self.window = window
        self.config = config
        self.limits = limits or DSLLimits()
        self.ledger = ledger
        self.seal_id = sha256_json(self._seal_terms())
        self.guard_id = sha256_json({"data_sha256": panel.content_sha256, "window": self._window_terms()})

    def _window_terms(self) -> dict[str, Any]:
        return {
            "name": self.window.name,
            "start": None if self.window.start is None else str(self.window.start.date()),
            "end": None if self.window.end is None else str(self.window.end.date()),
        }

    def _seal_terms(self) -> dict[str, Any]:
        return {
            "window": self._window_terms(),
            "data_sha256": self._panel.content_sha256,
            "metric_config": asdict(self.config),
            "limits": asdict(self.limits),
        }

    def _scan(self) -> tuple[Mapping[str, Any] | None, bool]:
        """``(commit payload or None, revealed?)`` for this guard, read from the ledger now."""

        self.ledger.sync()
        commit: Mapping[str, Any] | None = None
        revealed = False
        for record in self.ledger.records:
            if record.payload.get("guard_id") != self.guard_id:
                continue
            if record.kind == COMMIT_KIND and commit is None:
                commit = record.payload
            elif record.kind == REVEAL_KIND:
                revealed = True
        return commit, revealed

    def _current_data_sha256(self) -> str:
        try:
            return verify_panel_hash(self._panel)
        except PanelError as exc:
            raise SealError(f"the panel behind the hold-out changed: {exc}") from exc

    @property
    def data_sha256(self) -> str:
        return self._panel.content_sha256

    @property
    def committed(self) -> bool:
        return self._scan()[0] is not None

    @property
    def revealed(self) -> bool:
        return self._scan()[1]

    @property
    def commitment(self) -> str | None:
        commit, _ = self._scan()
        return None if commit is None else str(commit["commitment"])

    def commitment_for(self, factor_set: FrozenFactorSet) -> str:
        """Commitment hash this seal would record for ``factor_set``."""

        return sha256_json(
            {
                "seal_id": self.seal_id,
                "terms": self._seal_terms(),
                "factor_set": factor_set.to_dict(),
                "factor_set_sha256": factor_set.sha256,
            }
        )

    def commit(
        self,
        factor_set: FrozenFactorSet,
        *,
        data_sha256: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> str:
        """Record the commitment in the ledger; returns the commitment hash.

        ``data_sha256`` should be computed independently of this object (for
        example re-hashed from the values at commit time); it must equal the
        hash recomputed from the panel behind the seal.
        """

        commit, revealed = self._scan()
        if revealed:
            raise SealError("this test window has already been revealed on this data")
        if commit is not None:
            raise SealError("this hold-out already holds a commitment; it cannot be re-committed")
        current = self._current_data_sha256()
        if data_sha256 is not None and data_sha256 != current:
            raise SealError("data hash does not match the panel behind the hold-out")
        commitment = self.commitment_for(factor_set)
        self.ledger.append(
            COMMIT_KIND,
            {
                "seal_id": self.seal_id,
                "guard_id": self.guard_id,
                "commitment": commitment,
                "factor_set_sha256": factor_set.sha256,
                "n_factors": len(factor_set),
                "terms": self._seal_terms(),
                "context": json_safe(dict(context or {})),
            },
        )
        return commitment

    def reveal(
        self,
        factor_set: FrozenFactorSet,
        *,
        authorize: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> RevealResult:
        """Compute and log test metrics for the committed set, at most once.

        ``authorize`` (e.g. a project-wide reveal register) is called with the
        reveal descriptor after every local check and before any test metric
        is computed; it may raise :class:`SealError` to refuse.
        """

        commit, revealed = self._scan()
        if commit is None:
            raise SealError("reveal requested before any commitment")
        if revealed:
            self._refuse("second reveal", factor_set)
            raise SealError("this hold-out has already been revealed; a second reveal is refused")
        if self.commitment_for(factor_set) != commit["commitment"]:
            self._refuse("factor set or seal terms differ from the commitment", factor_set)
            raise SealError("factor set differs from the committed factor set; reveal refused")
        self._current_data_sha256()
        descriptor = {
            "guard_id": self.guard_id,
            "seal_id": self.seal_id,
            "commitment": commit["commitment"],
            "factor_set_sha256": factor_set.sha256,
            "data_sha256": self._panel.content_sha256,
            "window": self._window_terms(),
        }
        if authorize is not None:
            try:
                authorize(descriptor)
            except SealError:
                self._refuse("refused by the reveal register", factor_set)
                raise

        evaluator = Evaluator(self._panel, limits=self.limits)
        evaluations: dict[str, FactorEvaluation] = {}
        signals: dict[str, pd.DataFrame] = {}
        factor_rows: list[dict[str, Any]] = []
        for factor in factor_set.factors:
            signal = oriented_signal(evaluator, factor)
            evaluation = evaluate_signal(
                signal,
                self._panel.close,
                config=self.config,
                window=self.window,
                expression=factor.expression,
                expression_hash=factor.expression_hash,
                panel_sha256=self._panel.content_sha256,
            )
            evaluations[factor.expression_hash] = evaluation
            signals[factor.expression_hash] = signal
            factor_rows.append({**factor.to_dict(), "report": evaluation.report.to_dict()})
        composite: dict[str, Any] | None = None
        if factor_set.factors:
            combined = composite_signal(list(signals.values()))
            evaluation = evaluate_signal(
                combined,
                self._panel.close,
                config=self.config,
                window=self.window,
                expression=COMPOSITE_NAME,
                panel_sha256=self._panel.content_sha256,
            )
            evaluations[COMPOSITE_NAME] = evaluation
            signals[COMPOSITE_NAME] = combined
            composite = evaluation.report.to_dict()
        payload = {
            **descriptor,
            "factors": factor_rows,
            "composite": composite,
        }
        self.ledger.append(REVEAL_KIND, payload)
        return RevealResult(json_safe(payload), evaluations, signals)

    def _refuse(self, reason: str, factor_set: FrozenFactorSet) -> None:
        self.ledger.append(
            REFUSED_KIND,
            {
                "seal_id": self.seal_id,
                "guard_id": self.guard_id,
                "reason": reason,
                "factor_set_sha256": factor_set.sha256,
            },
        )


__all__ = [
    "COMPOSITE_NAME",
    "FrozenFactor",
    "FrozenFactorSet",
    "RevealResult",
    "SealError",
    "SealedHoldout",
    "composite_signal",
    "oriented_signal",
]
