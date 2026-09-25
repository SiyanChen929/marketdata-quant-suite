"""A proposer must never receive validation or test information."""

from __future__ import annotations

import dataclasses
import json

import pytest

from lfm_helpers import small_market, small_splits, tiny_search_config
from llm_factor_mining.data import Panel, panel_from_bars
from llm_factor_mining.evaluate import MetricConfig, evaluate_signal
from llm_factor_mining.jsonutil import json_safe
from llm_factor_mining.proposers import (
    EvaluatedFeedback,
    LeakageError,
    Proposal,
    ProposalContext,
    RandomGrammarProposer,
    RejectedFeedback,
)
from llm_factor_mining.proposers.evolutionary import EvolutionaryProposer
from llm_factor_mining.search import run_search


def test_context_types_have_no_room_for_validation_or_test_metrics() -> None:
    for cls in (ProposalContext, EvaluatedFeedback, RejectedFeedback):
        names = {field.name for field in dataclasses.fields(cls)}
        assert not any("valid" in name or "test" in name or "date" in name or "symbol" in name for name in names)
        assert hasattr(cls, "__slots__")  # no per-instance __dict__ to smuggle extra attributes
    metric_fields = {field.name for field in dataclasses.fields(EvaluatedFeedback)} - {"expression", "complexity"}
    assert metric_fields and all(name.startswith("formation_") for name in metric_fields)
    assert {field.name for field in dataclasses.fields(ProposalContext)} == {
        "round_index",
        "n_requested",
        "grammar",
        "top",
        "bottom",
        "last_round",
        "rejected",
        "n_trials_so_far",
        "budget_remaining",
    }


def test_context_is_frozen_and_slotted() -> None:
    context = ProposalContext(0, 5, "grammar")
    with pytest.raises(dataclasses.FrozenInstanceError):
        context.n_requested = 6  # type: ignore[misc]
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(context, "validation_ic", 0.1)
    feedback = EvaluatedFeedback("close", 0.1, 0.5, 2.0, 0.3, 1.0)
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(feedback, "test_ic", 0.2)


def test_context_rejects_foreign_payloads() -> None:
    good = EvaluatedFeedback("close", 0.1, 0.5, 2.0, 0.3, 1.0)

    @dataclasses.dataclass(frozen=True)
    class Smuggler:
        expression: str = "close"
        validation_ic: float = 0.9

    with pytest.raises(LeakageError):
        ProposalContext(0, 5, "grammar", top=(Smuggler(),))  # type: ignore[arg-type]
    with pytest.raises(LeakageError):
        ProposalContext(0, 5, "grammar", last_round=({"expression": "close", "test_ic": 0.3},))  # type: ignore[arg-type]
    with pytest.raises(LeakageError):
        ProposalContext(0, 5, "grammar", rejected=(good,))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ProposalContext(0, 5, "grammar", top={"close": good})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        EvaluatedFeedback("close", "0.1", 0.5, 2.0, 0.3, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        RejectedFeedback("close +", ("VALIDATION_IC",), "")
    ProposalContext(0, 5, "grammar", top=(good,), bottom=[good])


def test_feedback_factory_refuses_non_formation_reports() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    signal = market.planted_signals["reversal_5"]
    config = MetricConfig(min_names=10)
    for window in (splits.validation, splits.test):
        report = evaluate_signal(signal, market.panel.close, config=config, window=window, expression="x").report
        with pytest.raises(LeakageError):
            EvaluatedFeedback.from_report(report, complexity=1.0)
    report = evaluate_signal(signal, market.panel.close, config=config, window=splits.formation, expression="x").report
    item = EvaluatedFeedback.from_report(report, complexity=2.0)
    assert item.formation_ic == pytest.approx(report.ic_mean)


class SpyProposer:
    """Delegates to a seeded baseline and records every context it receives."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = f"spy-{inner.name}"
        self.contexts: list[ProposalContext] = []

    def propose(self, context: ProposalContext) -> list[Proposal]:
        self.contexts.append(context)
        return self.inner.propose(context)

    def describe(self) -> dict:
        return {"inner": self.inner.describe()}


def _perturb_after(panel_bars, cutoff):
    from lfm_helpers import perturb_after

    return perturb_after(panel_bars, cutoff, seed=5, scale=0.3)


def _context_fingerprint(contexts: list[ProposalContext]) -> str:
    return json.dumps([json_safe(dataclasses.asdict(context)) for context in contexts], sort_keys=True)


@pytest.mark.parametrize("make", [RandomGrammarProposer, EvolutionaryProposer])
def test_proposer_view_is_invariant_to_post_formation_data(make) -> None:
    market = small_market()
    splits = small_splits(market.panel)
    altered_bars = _perturb_after(market.bars, splits.formation.end)
    altered: Panel = panel_from_bars(altered_bars)
    assert altered.content_sha256 != market.panel.content_sha256
    config = tiny_search_config(budget=30, batch_size=10)

    spy_a, spy_b = SpyProposer(make(7)), SpyProposer(make(7))
    result_a = run_search(spy_a, market.panel, splits, config)
    result_b = run_search(spy_b, altered, splits, config)

    assert len(spy_a.contexts) >= 3
    assert _context_fingerprint(spy_a.contexts) == _context_fingerprint(spy_b.contexts)
    # the formation feedback is non-trivial ...
    assert any(context.top for context in spy_a.contexts[1:])
    # ... and the post-formation data really did change what selection and the seal saw
    va = [row["validation"] for row in result_a.selection]
    vb = [row["validation"] for row in result_b.selection]
    assert va != vb or result_a.reveal.payload != result_b.reveal.payload
