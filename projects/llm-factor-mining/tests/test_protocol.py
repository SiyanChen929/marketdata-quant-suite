from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from lfm_helpers import calendar, make_bars, small_splits
from llm_factor_mining.data import panel_from_bars
from llm_factor_mining.dsl import canonical_string, parse, structural_hash
from llm_factor_mining.dsl.library import get
from llm_factor_mining.evaluate import Evaluator, MetricConfig, evaluate_signal
from llm_factor_mining.protocol import (
    FrozenFactor,
    FrozenFactorSet,
    GENESIS_HASH,
    LedgerIntegrityError,
    SealError,
    SealedHoldout,
    SplitError,
    TrialLedger,
    chronological_splits,
    library_novelty,
    split_ic_at_cutoff,
    truncate_panel,
    verify_ledger_file,
)
from llm_factor_mining.protocol.splits import Splits
from llm_factor_mining.evaluate.metrics import EvaluationWindow


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------


def test_chronological_splits_are_ordered_with_embargo() -> None:
    dates = calendar(300)
    splits = chronological_splits(dates, lag=1, max_horizon=5)
    assert splits.embargo_sessions == 6
    f, v, t = splits.windows()
    assert f.start == dates[0] and t.end == dates[-1]
    pos = {d: i for i, d in enumerate(dates)}
    assert pos[v.start] - pos[f.end] - 1 == 6
    assert pos[t.start] - pos[v.end] - 1 == 6
    counts = [pos[w.end] - pos[w.start] + 1 for w in splits.windows()]
    assert sum(counts) + 12 == 300
    assert counts[0] > counts[1] > 0 and counts[2] > 0
    payload = splits.to_dict()
    assert payload["formation"]["start"] == str(dates[0].date())


def test_splits_reject_short_embargo_and_bad_order() -> None:
    dates = calendar(300)
    with pytest.raises(SplitError):
        chronological_splits(dates, lag=1, max_horizon=5, embargo_sessions=3)
    with pytest.raises(SplitError):
        chronological_splits(calendar(50), min_sessions=20)
    good = chronological_splits(dates)
    overlapping = Splits(
        formation=good.formation,
        validation=EvaluationWindow("validation", good.formation.end, good.validation.end),
        test=good.test,
        embargo_sessions=2,
        lag=1,
        max_horizon=1,
    )
    with pytest.raises(SplitError):
        overlapping.check_calendar(dates)
    with pytest.raises(SplitError):
        Splits(good.validation, good.formation, good.test, 2, 1, 1)  # wrong names


def test_truncate_panel_rehashes_and_keeps_values() -> None:
    panel = panel_from_bars(make_bars(120, 12, seed=4))
    cut = panel.dates[79]
    short = truncate_panel(panel, cut)
    assert short.dates[-1] == cut and len(short.dates) == 80
    assert short.content_sha256 != panel.content_sha256
    assert short.metadata["parent_content_sha256"] == panel.content_sha256
    pd.testing.assert_frame_equal(short.close, panel.close.iloc[:80])
    full = Evaluator(panel).evaluate("ts_mean(returns, 5)").iloc[:80]
    part = Evaluator(short).evaluate("ts_mean(returns, 5)")
    pd.testing.assert_frame_equal(full, part)


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


def _filled_ledger(path=None) -> TrialLedger:
    ledger = TrialLedger(path, timestamps=False)
    ledger.append("run_start", {"note": "x"})
    for index in range(4):
        ledger.append("trial", {"trial_index": index, "value": index * 0.5, "missing": float("nan")})
    ledger.append("duplicate", {"expression": "close"})
    return ledger


def test_ledger_hash_chain_and_trial_count(tmp_path) -> None:
    ledger = _filled_ledger(tmp_path / "ledger.jsonl")
    ledger.verify()
    assert len(ledger) == 6 and ledger.n_trials == 4
    assert ledger.records[0].prev_hash == GENESIS_HASH
    for previous, current in zip(ledger.records, ledger.records[1:]):
        assert current.prev_hash == previous.record_hash
    assert ledger.records[1].payload["missing"] is None  # NaN stored as null
    lines = (tmp_path / "ledger.jsonl").read_text().splitlines()
    assert len(lines) == 6 and json.loads(lines[-1])["record_hash"] == ledger.head_hash
    # identical content without timestamps -> identical chain
    assert _filled_ledger().head_hash == ledger.head_hash


def test_ledger_reopen_continues_chain(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    first = _filled_ledger(path)
    head = first.head_hash
    reopened = TrialLedger(path, timestamps=False)
    assert reopened.head_hash == head and reopened.n_trials == 4
    reopened.append("trial", {"trial_index": 4})
    assert verify_ledger_file(path)[-1].prev_hash == head


@pytest.mark.parametrize("attack", ["edit", "delete", "swap", "insert", "partial"])
def test_ledger_detects_tampering(tmp_path, attack: str) -> None:
    path = tmp_path / "ledger.jsonl"
    _filled_ledger(path)
    lines = path.read_text().splitlines()
    if attack == "edit":
        record = json.loads(lines[2])
        record["payload"]["value"] = 99.0
        lines[2] = json.dumps(record)
    elif attack == "delete":
        del lines[3]
    elif attack == "swap":
        lines[2], lines[3] = lines[3], lines[2]
    elif attack == "insert":
        lines.insert(2, lines[2])
    text = "\n".join(lines) + "\n"
    if attack == "partial":
        text = text[:-15]
    path.write_text(text)
    with pytest.raises(LedgerIntegrityError):
        verify_ledger_file(path)
    with pytest.raises(LedgerIntegrityError):
        TrialLedger(path)


def test_ledger_tail_truncation_needs_expected_head(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = _filled_ledger(path)
    head = ledger.head_hash
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    verify_ledger_file(path)  # a clean prefix is internally consistent...
    with pytest.raises(LedgerIntegrityError):
        verify_ledger_file(path, expected_head=head)  # ...but not the recorded head


def test_ledger_in_memory_tamper_detected() -> None:
    ledger = _filled_ledger()
    ledger.records[2].payload["value"] = -1.0  # type: ignore[index]
    with pytest.raises(LedgerIntegrityError):
        ledger.verify()


# --------------------------------------------------------------------------
# sealed hold-out
# --------------------------------------------------------------------------


@pytest.fixture()
def seal_setup():
    panel = panel_from_bars(make_bars(240, 15, seed=21))
    splits = small_splits(panel)
    config = MetricConfig(min_names=10)
    ledger = TrialLedger(timestamps=False)
    node = parse(get("low_volatility_20d").expression)
    factor = FrozenFactor(canonical_string(node), structural_hash(node), -1)
    other = parse(get("reversal_5d").expression)
    frozen = FrozenFactorSet((factor, FrozenFactor(canonical_string(other), structural_hash(other), 1)))
    return panel, splits, config, ledger, frozen


def test_seal_commit_then_single_reveal(seal_setup) -> None:
    panel, splits, config, ledger, frozen = seal_setup
    holdout = SealedHoldout(panel, splits.test, config, ledger)
    commitment = holdout.commit(frozen, data_sha256=panel.content_sha256)
    assert holdout.committed and not holdout.revealed
    assert commitment == holdout.commitment_for(frozen)
    result = holdout.reveal(frozen)
    kinds = [record.kind for record in ledger.records]
    assert kinds == ["seal_commit", "seal_reveal"]  # commitment written before any test metric
    assert result.payload["commitment"] == commitment
    assert len(result.payload["factors"]) == 2 and result.payload["composite"] is not None
    # reveal metrics are exactly the evaluator's test-window metrics of the oriented signal
    first = frozen.factors[0]
    signal = Evaluator(panel).evaluate(first.expression) * -1.0
    direct = evaluate_signal(signal, panel.close, config=config, window=splits.test).report
    assert result.payload["factors"][0]["report"]["ic_mean"] == pytest.approx(direct.ic_mean)
    assert result.payload["factors"][0]["report"]["window"] == "test"

    with pytest.raises(SealError, match="second reveal"):
        holdout.reveal(frozen)
    assert ledger.records[-1].kind == "seal_refused"
    # a fresh instance on the same ledger knows the seal is already open
    again = SealedHoldout(panel, splits.test, config, ledger)
    assert again.revealed
    with pytest.raises(SealError):
        again.reveal(frozen)
    with pytest.raises(SealError):
        again.commit(frozen)


def test_seal_refuses_other_factor_set_and_misordered_calls(seal_setup) -> None:
    panel, splits, config, ledger, frozen = seal_setup
    holdout = SealedHoldout(panel, splits.test, config, ledger)
    with pytest.raises(SealError, match="before any commitment"):
        holdout.reveal(frozen)
    with pytest.raises(SealError, match="data hash"):
        holdout.commit(frozen, data_sha256="0" * 64)
    holdout.commit(frozen)
    with pytest.raises(SealError, match="re-committed"):
        holdout.commit(frozen)
    flipped = FrozenFactorSet(
        (FrozenFactor(frozen.factors[0].expression, frozen.factors[0].expression_hash, 1), frozen.factors[1])
    )
    with pytest.raises(SealError, match="differs"):
        holdout.reveal(flipped)
    subset = FrozenFactorSet(frozen.factors[:1])
    with pytest.raises(SealError, match="differs"):
        holdout.reveal(subset)
    assert [record.kind for record in ledger.records] == ["seal_commit", "seal_refused", "seal_refused"]
    assert not holdout.revealed
    holdout.reveal(frozen)  # the committed set can still be revealed once
    assert holdout.revealed


def test_frozen_factor_validation() -> None:
    node = parse("-ts_std(returns, 20)")
    canonical = canonical_string(node)
    FrozenFactor(canonical, structural_hash(node), 1)
    with pytest.raises(ValueError):
        FrozenFactor("-ts_std(returns, 20)", structural_hash(node), 1)  # not canonical
    with pytest.raises(ValueError):
        FrozenFactor(canonical, "0" * 64, 1)
    with pytest.raises(ValueError):
        FrozenFactor(canonical, structural_hash(node), 0)
    factor = FrozenFactor(canonical, structural_hash(node), 1)
    with pytest.raises(ValueError):
        FrozenFactorSet((factor, factor))
    assert FrozenFactorSet((factor,)).sha256 == FrozenFactorSet((factor,)).sha256


def test_empty_frozen_set_can_be_sealed(seal_setup) -> None:
    panel, splits, config, ledger, _ = seal_setup
    holdout = SealedHoldout(panel, splits.test, config, ledger)
    empty = FrozenFactorSet(())
    holdout.commit(empty)
    result = holdout.reveal(empty)
    assert result.payload["factors"] == [] and result.payload["composite"] is None


# --------------------------------------------------------------------------
# contamination diagnostics
# --------------------------------------------------------------------------


def test_split_ic_at_cutoff_drops_straddling_dates() -> None:
    dates = calendar(200)
    rng = np.random.default_rng(0)
    values = np.where(np.arange(200) < 120, 0.10, 0.0) + rng.normal(0, 0.01, 200)
    ic = pd.Series(values, index=dates)
    result = split_ic_at_cutoff(ic, dates[120], dates, lag=1, horizon=2)
    assert result.n_straddling == 3  # signals at positions 117..119 end at or after the cutoff
    assert result.pre.n == 117 and result.post.n == 80
    assert result.pre.mean == pytest.approx(0.10, abs=0.005)
    assert result.post.mean == pytest.approx(0.0, abs=0.005)
    assert result.diff_z < -10
    payload = result.to_dict()
    assert payload["cutoff"] == str(dates[120].date())
    with pytest.raises(ValueError):
        split_ic_at_cutoff(ic, dates[120], dates[:100])


def test_library_novelty() -> None:
    same = library_novelty(get("low_volatility_20d").expression)
    assert same.similarity == 1.0 and same.novelty == 0.0 and same.nearest == "low_volatility_20d"
    variant = library_novelty("-ts_std(returns, 30)")
    assert variant.similarity < 1.0
    assert variant.abstract_similarity == 1.0 and variant.abstract_nearest in {
        "low_volatility_20d",
        "low_volatility_60d",
    }
    fresh = library_novelty("ts_corr(high, dollar_volume, 7) * sign(delta(low, 3))")
    assert fresh.novelty > 0.5
    assert math.isclose(fresh.to_dict()["novelty"], fresh.novelty)


# --------------------------------------------------------------------------
# seal guard, rescans and data-hash checks
# --------------------------------------------------------------------------


def test_seal_guard_ignores_metric_and_limit_tweaks(seal_setup) -> None:
    from llm_factor_mining.dsl.validate import DSLLimits

    panel, splits, config, ledger, frozen = seal_setup
    first = SealedHoldout(panel, splits.test, config, ledger)
    first.commit(frozen)
    first.reveal(frozen)
    subset = FrozenFactorSet(frozen.factors[:1])
    tweaked = [
        SealedHoldout(panel, splits.test, MetricConfig(min_names=10, cost_bps=10.000001), ledger),
        SealedHoldout(panel, splits.test, config, ledger, limits=DSLLimits(max_nodes=49)),
    ]
    for holdout in tweaked:
        assert holdout.seal_id != first.seal_id and holdout.guard_id == first.guard_id
        assert holdout.revealed
        with pytest.raises(SealError):
            holdout.commit(subset)
        with pytest.raises(SealError):
            holdout.reveal(subset)
    assert len(ledger.of_kind("seal_reveal")) == 1 and len(ledger.of_kind("seal_commit")) == 1


def test_seal_state_is_rescanned_on_every_call(seal_setup) -> None:
    panel, splits, config, ledger, frozen = seal_setup
    a = SealedHoldout(panel, splits.test, config, ledger)
    b = SealedHoldout(panel, splits.test, config, ledger)  # created before either commits
    a.commit(frozen)
    with pytest.raises(SealError, match="re-committed"):
        b.commit(frozen)
    b.reveal(frozen)  # b sees a's commitment
    with pytest.raises(SealError, match="second reveal"):
        a.reveal(frozen)
    assert len(ledger.of_kind("seal_reveal")) == 1


def test_seal_state_is_shared_by_writers_of_one_ledger_file(tmp_path, seal_setup) -> None:
    panel, splits, config, _, frozen = seal_setup
    path = tmp_path / "ledger.jsonl"
    writer_a, writer_b = TrialLedger(path, timestamps=False), TrialLedger(path, timestamps=False)
    SealedHoldout(panel, splits.test, config, writer_a).commit(frozen)
    SealedHoldout(panel, splits.test, config, writer_b).reveal(frozen)
    with pytest.raises(SealError):
        SealedHoldout(panel, splits.test, config, writer_a).reveal(frozen)
    kinds = [record.kind for record in verify_ledger_file(path)]
    assert kinds == ["seal_commit", "seal_reveal", "seal_refused"]


def test_seal_detects_in_place_mutation_of_the_panel(seal_setup) -> None:
    panel, splits, config, ledger, frozen = seal_setup
    holdout = SealedHoldout(panel, splits.test, config, ledger)
    panel.close.iloc[-1, 0] *= 1.5  # frames are mutable although Panel is frozen
    try:
        with pytest.raises(SealError, match="changed"):
            holdout.commit(frozen)
    finally:
        panel.close.iloc[-1, 0] /= 1.5
    assert holdout.commit(frozen)


def test_ledger_sync_adopts_appends_and_rejects_rewrites(tmp_path) -> None:
    path = tmp_path / "ledger.jsonl"
    a, b = TrialLedger(path, timestamps=False), TrialLedger(path, timestamps=False)
    a.append("trial", {"i": 0})
    b.append("trial", {"i": 1})  # b adopts a's record before appending
    a.append("trial", {"i": 2})
    records = verify_ledger_file(path)
    assert [record.payload["i"] for record in records] == [0, 1, 2]
    assert a.head_hash == records[-1].record_hash
    path.write_text("\n".join(path.read_text().splitlines()[:1]) + "\n")  # rewritten behind a's back
    with pytest.raises(LedgerIntegrityError):
        a.append("trial", {"i": 3})


# --------------------------------------------------------------------------
# study registry
# --------------------------------------------------------------------------


def _study(tmp_path, max_reveals: int = 1):
    from llm_factor_mining.protocol import StudyRegistry, StudySpec

    panel = panel_from_bars(make_bars(240, 15, seed=21))
    splits = small_splits(panel)
    registry = StudyRegistry(tmp_path / "registry.jsonl")
    spec = StudySpec("u1", "synthetic", panel.symbols, "2031-01-02", "2031-12-31", (0.6, 0.2, 0.2), 1, 1, max_reveals)
    registry.register_study(spec, splits)
    return registry, spec, panel, splits


def test_registry_fixes_splits_and_counts_exploration(tmp_path) -> None:
    from llm_factor_mining.protocol import RegistryError, StudySpec

    registry, spec, panel, splits = _study(tmp_path)
    assert registry.require("u1").splits.to_dict() == splits.to_dict()
    assert registry.register_study(spec, splits).spec == spec  # identical re-registration is a no-op
    other = chronological_splits(panel.dates, fractions=(0.5, 0.3, 0.2))
    with pytest.raises(RegistryError, match="different terms"):
        registry.register_study(spec, other)
    with pytest.raises(RegistryError, match="not registered"):
        registry.require("u2")
    with pytest.raises(RegistryError):
        StudySpec("bad name", "store", (), "2031-01-02", "2031-12-31", (0.6, 0.2, 0.2), 1, 1)
    assert registry.record_exploration("u1", command="evaluate", expression_hash="x") == 1
    assert registry.record_exploration("u1", command="evaluate", expression_hash="y") == 2
    reopened = type(registry)(tmp_path / "registry.jsonl")
    assert reopened.n_explorations("u1") == 2 and reopened.summary("u1")["n_reveals"] == 0


def test_registry_caps_reveals_and_requires_a_recorded_commitment(tmp_path, seal_setup) -> None:
    registry, _, panel, splits = _study(tmp_path, max_reveals=1)
    _, _, config, _, frozen = seal_setup
    first = SealedHoldout(panel, splits.test, config, TrialLedger(timestamps=False))
    commitment = first.commit(frozen)
    with pytest.raises(SealError, match="never recorded"):
        first.reveal(frozen, authorize=registry.reveal_authorizer("u1", run="a"))
    registry.record_commit("u1", run="a", commitment=commitment)
    first.reveal(frozen, authorize=registry.reveal_authorizer("u1", run="a"))
    # a fresh run (fresh ledger) on the same study cannot reveal again: the cap is project-wide
    second = SealedHoldout(panel, splits.test, config, TrialLedger(timestamps=False))
    registry.record_commit("u1", run="b", commitment=second.commit(frozen))
    with pytest.raises(SealError, match="maximum of 1"):
        second.reveal(frozen, authorize=registry.reveal_authorizer("u1", run="b"))
    assert second.ledger.of_kind("seal_refused")
    kinds = [record.kind for record in registry.ledger.records]
    assert kinds.count("reveal") == 1 and kinds.count("reveal_refused") == 2
    shifted = SealedHoldout(panel, splits.validation, config, TrialLedger(timestamps=False))
    registry.record_commit("u1", run="c", commitment=shifted.commit(frozen))
    with pytest.raises(SealError, match="registered test window"):
        shifted.reveal(frozen, authorize=registry.reveal_authorizer("u1", run="c"))


def test_behavioural_novelty_sees_through_respelled_library_factors() -> None:
    from llm_factor_mining.benchmark.synthetic import DRAWN_PLANTED
    from llm_factor_mining.protocol import behavioural_novelty, library_signals

    panel = panel_from_bars(make_bars(300, 30, seed=5))
    engine = Evaluator(panel)
    library = library_signals(engine)
    dates = panel.dates[80:]
    respelled = behavioural_novelty(engine.evaluate("-ts_sum(returns, 5)"), library, dates)
    assert respelled.nearest in {"reversal_5d", "decayed_reversal_10d"} and respelled.novelty < 0.15
    assert library_novelty("-ts_sum(returns, 5)").novelty > 0.5  # the syntactic score is fooled
    drawn = behavioural_novelty(engine.evaluate(DRAWN_PLANTED.expression), library, dates)
    assert drawn.novelty > 0.6
