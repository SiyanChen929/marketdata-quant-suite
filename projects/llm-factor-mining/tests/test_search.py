from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lfm_helpers import bars_from_close, perturb_after, random_walk_close, small_market, small_splits, tiny_search_config
from llm_factor_mining.data import PanelError, panel_from_bars
from llm_factor_mining.evaluate import Evaluator, MetricConfig, evaluate_signal
from llm_factor_mining.evaluate.metrics import embargoed_signal_dates, forward_returns
from llm_factor_mining.inference import deflated_sharpe_ratio
from llm_factor_mining.protocol import TrialLedger, verify_ledger_file
from llm_factor_mining.protocol.splits import chronological_splits, truncate_panel
from llm_factor_mining.proposers import (
    EvolutionaryProposer,
    FakeBackend,
    LLMProposer,
    Proposal,
    ProposalContext,
    ProposerError,
    RandomGrammarProposer,
)
from llm_factor_mining.search import SearchConfig, behaviour_signature, run_search, window_statistics


class ScriptedProposer:
    """Returns fixed batches (cycling through ``batches``), recording contexts."""

    name = "scripted"

    def __init__(self, batches: list[list[str]]) -> None:
        self.batches = batches
        self.contexts: list[ProposalContext] = []

    def propose(self, context: ProposalContext) -> list[Proposal]:
        self.contexts.append(context)
        batch = self.batches[min(len(self.contexts) - 1, len(self.batches) - 1)]
        return [Proposal(text, "scripted", {"generator": "scripted"}) for text in batch]

    def describe(self) -> dict:
        return {"batches": self.batches}


def test_window_statistics_match_full_metric_report() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    config = MetricConfig(min_names=10)
    signal = market.planted_signals["reversal_5"]
    forward = forward_returns(market.panel.close, config.horizon, config.lag)
    for window in splits.windows():
        stats, ic = window_statistics(signal, forward, window, config)
        report = evaluate_signal(signal, market.panel.close, config=config, window=window).report
        assert stats.n_ic_dates == report.n_ic_dates == len(ic)
        assert stats.ic_mean == pytest.approx(report.ic_mean, rel=1e-12)
        assert stats.icir == pytest.approx(report.icir, rel=1e-12)
        assert stats.t_nw == pytest.approx(report.ic_tstat_nw, rel=1e-12)
        assert stats.turnover == pytest.approx(report.top_quantile_turnover, rel=1e-12)
        assert 0.0 <= stats.p_value <= 1.0
        assert stats.n_scorable_dates >= stats.n_ic_dates


def test_search_counts_every_trial_and_logs_failures() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    batches = [
        ["-ts_mean(returns, 5)", "ts_mean(close,", "nosuchop(close)", "-ts_mean(returns,5)", "ts_mean(close,"],
        ["volume / ts_mean(volume, 20)", "ts_mean(close, 0)", "cs_rank(volume)"],
    ]
    proposer = ScriptedProposer(batches)
    result = run_search(proposer, market.panel, splits, tiny_search_config(budget=7, patience=2))
    statuses = [trial.status for trial in result.trials]
    assert statuses == ["evaluated", "invalid", "invalid", "evaluated", "invalid", "evaluated"]
    # the canonical repeat and the repeated invalid text are duplicates, not trials
    assert result.n_duplicates >= 2
    assert result.ledger.n_trials == result.n_trials == 6
    # the FDR family is fixed at the budget: the run stalled after 6 trials but m = 7
    assert result.stop_reason == "stalled" and not result.complete
    assert result.multiple_test is not None and result.multiple_test.m == 7
    assert result.validity_rate == pytest.approx(result.n_valid_proposals / result.n_proposals)
    codes = {code for trial in result.trials for code in trial.codes}
    assert {"PARSE", "UNKNOWN_OP", "WINDOW_RANGE"} <= codes
    # rejected feedback reaches the next context, formation metrics come back for evaluated ones
    second = proposer.contexts[1]
    assert {item.codes[0] for item in second.rejected} >= {"PARSE", "UNKNOWN_OP"}
    assert second.top and second.top[0].expression == "neg(ts_mean(returns,5))"


def test_behaviour_signature_identifies_rank_identical_expressions() -> None:
    market = small_market()
    engine = Evaluator(market.panel)
    dates = market.panel.dates[40:120]
    same = ["volume", "log(volume)", "cs_rank(volume)", "-volume", "power(volume, 2)", "dollar_volume / close"]
    signatures = {behaviour_signature(engine.evaluate(text), dates) for text in same}
    assert len(signatures) == 1
    assert behaviour_signature(engine.evaluate("close"), dates) not in signatures
    assert behaviour_signature(engine.evaluate("-ts_mean(returns, 5)"), dates) == behaviour_signature(
        engine.evaluate("ts_sum(returns, 5)"), dates
    )


def test_behavioural_duplicates_count_once_in_the_screen() -> None:
    market = small_market(snr=0.6)
    splits = small_splits(market.panel)
    copies = ["volume", "log(volume)", "cs_rank(volume)", "sqrt(volume)", "abs(volume)"]
    result = run_search(
        ScriptedProposer([["-ts_mean(returns, 5)", *copies]]),
        market.panel,
        splits,
        tiny_search_config(budget=6, patience=1),
    )
    classes = [trial.behaviour_class for trial in result.trials]
    assert classes == [0, 1, 1, 1, 1, 1]
    assert result.funnel["n_behaviour_classes"] == 2 and result.funnel["n_behaviour_duplicates"] == 4
    assert result.multiple_test is not None and result.multiple_test.m == 2  # budget 6 minus 4 duplicates
    assert result.multiple_test.rejected.size == 2  # one hypothesis per class
    assert {row["trial_index"] for row in result.selection} <= {0, 1}
    records = [record.payload for record in result.ledger.of_kind("trial")]
    assert records[2]["behaviour_class"] == 1 and records[2]["behaviour_signature"] == records[1]["behaviour_signature"]


def test_degenerate_trials_are_counted_but_never_screened() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    tiny_sample = "delta(delta(ts_corr(ts_min(open,40),ts_sum(ts_cov(close,low,2),40),3),60),60)"
    proposer = ScriptedProposer([["-ts_mean(returns, 5)", tiny_sample, "ts_mean(close, 150)"], ["high"]])
    result = run_search(proposer, market.panel, splits, tiny_search_config(budget=4, patience=1))
    by_expression = {trial.expression: trial for trial in result.trials}
    for text in (tiny_sample, "ts_mean(close, 150)"):
        trial = by_expression[text]
        assert trial.status == "degenerate" and trial.codes == ("DEGENERATE",)
        assert trial.formation is not None
        assert trial.formation.n_ic_dates < result.config.required_ic_dates(trial.formation.n_scorable_dates)
    assert result.n_degenerate == 2
    assert result.multiple_test is not None
    assert result.multiple_test.m == 4 and result.multiple_test.rejected.size == 2  # counted in m, never tested
    assert {item.codes for item in proposer.contexts[1].rejected} == {("DEGENERATE",)}
    assert all(item.expression != "neg(ts_min(open,40))" for item in proposer.contexts[1].top)
    config = SearchConfig(min_ic_dates=100, min_ic_fraction=0.5)
    assert config.required_ic_dates(445) == 223 and config.required_ic_dates(147) == 100


def test_selection_rules_confirmation_orientation_and_decorrelation() -> None:
    market = small_market(snr=0.6)
    splits = small_splits(market.panel)
    batches = [
        [
            "-ts_mean(returns, 5)",
            "ts_mean(returns, 5)",  # rank-identical up to sign: a behavioural duplicate
            "-decay_linear(returns, 5)",  # correlated but not identical -> decorrelation
            "volume / ts_mean(volume, 20)",
            "high",  # price level: no planted signal
        ]
    ]
    result = run_search(
        ScriptedProposer(batches), market.panel, splits, tiny_search_config(budget=5, patience=1, top_k=5)
    )
    assert result.trials[1].behaviour_class == 0
    decisions = {row["canonical"]: row for row in result.selection}
    assert "ts_mean(returns,5)" not in decisions  # never a separate candidate
    selected = {factor.expression: factor.orientation for factor in result.frozen.factors}
    reversal_like = [c for c in ("neg(ts_mean(returns,5))", "neg(decay_linear(returns,5))") if c in decisions]
    assert sum(decisions[c]["decision"] == "selected" for c in reversal_like) == 1
    assert any(decisions[c]["decision"] == "rejected_correlation" for c in reversal_like)
    assert result.confirmation is not None and result.confirmation.m == len(result.selection)
    for row in result.selection:
        assert row["formation_p_adjusted"] <= result.config.fdr_alpha
        assert row["validation"]["icir"] is not None
        if row["decision"] == "selected":
            assert row["validation_confirmed"] and row["validation_p_adjusted"] <= result.config.confirm_alpha
    assert all(orientation in (1, -1) for orientation in selected.values())


def test_deflated_sharpe_diagnostic_uses_twice_the_family_for_absolute_icir() -> None:
    market = small_market(snr=0.6)
    splits = small_splits(market.panel)
    result = run_search(
        ScriptedProposer([["-ts_mean(returns, 5)", "volume / ts_mean(volume, 20)", "high", "low"]]),
        market.panel,
        splits,
        tiny_search_config(budget=4, patience=1),
    )
    representatives = [t for t in result.trials if t.is_class_representative]
    icirs = np.array([t.formation.icir for t in representatives])
    for row in result.selection:
        trial = result.trials[row["trial_index"]]
        stats = trial.formation
        expected = deflated_sharpe_ratio(
            abs(stats.icir),
            stats.n_ic_dates,
            n_trials=2 * result.multiple_test.m,
            sharpe_variance=float(icirs.var(ddof=1)),
            skew=row["orientation"] * stats.ic_skew,
            kurtosis=stats.ic_kurtosis,
        )
        assert row["formation_deflated_icir"] == pytest.approx(expected)


def test_search_is_deterministic_and_writes_run_directory(tmp_path) -> None:
    market = small_market()
    splits = small_splits(market.panel)
    config = tiny_search_config(budget=30, batch_size=10)
    first = run_search(EvolutionaryProposer(3), market.panel, splits, config, run_dir=tmp_path / "a")
    second = run_search(EvolutionaryProposer(3), market.panel, splits, config, run_dir=tmp_path / "b")
    assert first.ledger_head == second.ledger_head
    assert first.frozen == second.frozen and first.commitment == second.commitment
    assert [t.expression for t in first.trials] == [t.expression for t in second.trials]
    other = run_search(EvolutionaryProposer(4), market.panel, splits, config)
    assert [t.expression for t in other.trials] != [t.expression for t in first.trials]

    run = tmp_path / "a"
    for name in ("config.json", "ledger.jsonl", "selected.json", "reveal.json", "provenance.json"):
        assert (run / name).is_file(), name
    provenance = json.loads((run / "provenance.json").read_text())
    records = verify_ledger_file(run / "ledger.jsonl", expected_head=provenance["ledger_head"])
    kinds = [record.kind for record in records]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    assert kinds.index("selection") < kinds.index("seal_commit") < kinds.index("seal_reveal")
    assert kinds.count("trial") == first.n_trials == 30
    assert provenance["data"]["content_sha256"] == market.panel.content_sha256
    assert provenance["data"]["finality"] == "confirmed" and provenance["data"]["sources"] == ["synthetic"]
    assert provenance["commit_ledger_head"] in {record.record_hash for record in records}
    selected = json.loads((run / "selected.json").read_text())
    assert selected["commitment"] == first.commitment and selected["data_sha256"] == market.panel.content_sha256
    reveal = json.loads((run / "reveal.json").read_text())
    assert reveal["commitment"] == first.commitment
    summary = first.summary()
    assert summary["complete"] and "proposer_provenance" in summary and summary["funnel"]["n_trials"] == 30
    with pytest.raises(FileExistsError):
        run_search(EvolutionaryProposer(3), market.panel, splits, config, run_dir=run)


def test_selection_never_sees_the_test_window() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    config = tiny_search_config(budget=30, batch_size=10)
    altered = panel_from_bars(perturb_after(market.bars, splits.validation.end, seed=8, scale=0.3))
    base = run_search(RandomGrammarProposer(9), market.panel, splits, config)
    moved = run_search(RandomGrammarProposer(9), altered, splits, config)
    assert base.selection == moved.selection
    assert base.frozen == moved.frozen
    assert base.reveal is not None and moved.reveal is not None
    if base.frozen.factors:
        assert base.reveal.payload["factors"] != moved.reveal.payload["factors"]


def test_loader_reads_each_window_only_when_its_phase_starts() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    calls: list[pd.Timestamp | None] = []

    def load(end):
        calls.append(end)
        return market.panel if end is None else truncate_panel(market.panel, end)

    class Watcher(ScriptedProposer):
        def propose(self, context):
            assert calls == [splits.formation.end]  # nothing past the formation window is loaded yet
            return super().propose(context)

    result = run_search(Watcher([["-ts_mean(returns, 5)", "high"]]), load, splits, tiny_search_config(budget=2))
    assert calls == [splits.formation.end, splits.validation.end, None]
    direct = run_search(ScriptedProposer([["-ts_mean(returns, 5)", "high"]]), market.panel, splits, tiny_search_config(budget=2))
    assert result.commitment == direct.commitment

    def inconsistent(end):
        panel = market.panel if end is None else truncate_panel(market.panel, end)
        if end == splits.validation.end:
            return panel_from_bars(perturb_after(market.bars, splits.formation.start, seed=3))
        return panel

    with pytest.raises(PanelError, match="changed between phases"):
        run_search(ScriptedProposer([["high"]]), inconsistent, splits, tiny_search_config(budget=1))


def test_non_synthetic_reveal_needs_a_ledger_file_and_a_register(tmp_path) -> None:
    close = random_walk_close(260, 15, seed=3)
    panel = panel_from_bars(bars_from_close(close, seed=4, source="marketdata"))
    splits = chronological_splits(panel.dates, lag=1, max_horizon=1)
    config = tiny_search_config(budget=2)
    with pytest.raises(ValueError, match="reveal register"):
        run_search(ScriptedProposer([["high", "low"]]), panel, splits, config)
    with pytest.raises(ValueError, match="reveal register"):
        run_search(ScriptedProposer([["high", "low"]]), panel, splits, config, run_dir=tmp_path / "a")
    committed = run_search(ScriptedProposer([["high", "low"]]), panel, splits, config, run_dir=tmp_path / "b", reveal=False)
    assert committed.reveal is None and committed.commitment is not None
    assert TrialLedger(tmp_path / "b" / "ledger.jsonl").of_kind("seal_commit")


def test_proposer_errors_abort_the_run_without_a_commitment(tmp_path) -> None:
    market = small_market()
    splits = small_splits(market.panel)

    class Failing:
        name = "failing"

        def propose(self, context):
            raise ProposerError("model declined")

        def describe(self):
            return {}

    result = run_search(
        Failing(), market.panel, splits, tiny_search_config(max_proposer_failures=2), run_dir=tmp_path / "run"
    )
    assert result.stop_reason == "proposer_failures" and result.n_trials == 0
    assert result.aborted and not result.complete
    assert result.commitment is None and result.reveal is None and result.multiple_test is None
    kinds = [record.kind for record in result.ledger.records]
    assert kinds.count("proposer_error") == 2 and kinds[-1] == "run_aborted"
    assert "seal_commit" not in kinds and "seal_reveal" not in kinds
    assert not (tmp_path / "run" / "reveal.json").exists() and (tmp_path / "run" / "provenance.json").is_file()


def test_llm_proposer_runs_through_search_with_fake_backend(tmp_path) -> None:
    market = small_market()
    splits = small_splits(market.panel)

    def respond(system: str, user: str, schema) -> dict:
        assert "SYN0" not in user and "2031" not in user  # anonymized: no tickers or dates
        return {
            "proposals": [
                {"expression": "-ts_mean(returns, 5)", "rationale": "reversal", "economic_mechanism": "overreaction"},
                {"expression": "volume / ts_mean(volume, 20)", "rationale": "attention", "economic_mechanism": "attention"},
                {"expression": "ts_mean(close", "rationale": "typo", "economic_mechanism": "none"},
                {"expression": "-ts_corr(returns, delta(log(volume), 1), 10)", "rationale": "volume", "economic_mechanism": "liquidity"},
            ]
        }

    proposer = LLMProposer(FakeBackend(respond), forbidden_tokens=market.panel.symbols)
    result = run_search(proposer, market.panel, splits, tiny_search_config(budget=8, batch_size=4, patience=2), run_dir=tmp_path / "llm")
    assert result.n_invalid == 1 and result.n_evaluated == 3
    provenance = json.loads((tmp_path / "llm" / "provenance.json").read_text())
    assert provenance["proposer_provenance"]["served_models"] == ["fake-model"]
    assert "calls" not in provenance["proposer_provenance"]
    assert provenance["proposer"]["user_template_sha256"]
    assert result.summary()["proposer_provenance"]["served_models"] == ["fake-model"]
    trial = result.trials[0]
    assert trial.metadata["served_model"] == "fake-model" and trial.metadata["economic_mechanism"] == "overreaction"


def test_search_config_validation_and_lag_check() -> None:
    with pytest.raises(ValueError):
        SearchConfig(budget=0)
    with pytest.raises(ValueError):
        SearchConfig(fdr_method="bonferroni")
    with pytest.raises(ValueError):
        SearchConfig(confirm_alpha=1.5)
    with pytest.raises(ValueError):
        SearchConfig(min_ic_fraction=2.0)
    market = small_market()
    splits = small_splits(market.panel)
    with pytest.raises(ValueError):
        run_search(RandomGrammarProposer(0), market.panel, splits, tiny_search_config(metric=MetricConfig(horizon=5)))
    config = SearchConfig(budget=17, top_k=2)
    assert SearchConfig.from_dict(config.to_dict()) == config


def test_formation_dates_used_for_signatures_are_the_ic_dates() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    formation = truncate_panel(market.panel, splits.formation.end)
    dates = embargoed_signal_dates(formation.dates, splits.formation, lag=1, horizon=1)
    assert dates[-1] < splits.formation.end and len(dates) == len(formation.dates) - 2


def test_a_bar_source_that_appears_later_is_not_a_change_of_the_data(tmp_path) -> None:
    # regression: a vendor switch inside the test (or validation) window made the phase-prefix check
    # fail on unchanged data, because the longer load lists one more source label
    from llm_factor_mining.data import align_symbols, panel_values_sha256

    market = small_market()
    splits = small_splits(market.panel)
    for switch in (splits.test.start, splits.validation.start):
        bars = market.bars.copy()
        bars["source"] = "vendor_a"
        bars.loc[pd.to_datetime(bars["date"]) >= switch, "source"] = "vendor_b"
        symbols = sorted(bars["symbol"].unique())

        def load(end, bars=bars, symbols=symbols):  # like cli.store_loader: read bars up to ``end``
            part = bars if end is None else bars[pd.to_datetime(bars["date"]) <= pd.Timestamp(end)]
            return align_symbols(panel_from_bars(part.reset_index(drop=True)), symbols)

        shorter = load(splits.formation.end)
        longer = truncate_panel(load(None), splits.formation.end)
        assert shorter.sources == ("vendor_a",) and longer.sources == ("vendor_a", "vendor_b")
        assert panel_values_sha256(shorter) == panel_values_sha256(longer)
        run_dir = tmp_path / f"run-{switch.date()}"
        result = run_search(
            ScriptedProposer([["-ts_mean(returns, 5)", "high"]]), load, splits, tiny_search_config(budget=2),
            run_dir=run_dir, reveal=False,
        )
        assert result.commitment is not None and (run_dir / "selected.json").is_file()
        assert result.data_sha256 == load(None).content_sha256  # the committed hash still covers the sources


def test_windows_too_short_for_a_p_value_are_refused_before_the_run_starts(tmp_path) -> None:
    from llm_factor_mining.protocol.splits import SplitError
    from llm_factor_mining.search import window_capacity_problems

    market = small_market()  # 260 sessions: validation and test windows hold about 50 scorable dates
    splits = small_splits(market.panel)
    config = tiny_search_config(min_ic_dates=100)
    problems = window_capacity_problems(market.panel.dates, splits, config, n_symbols=len(market.panel.symbols))
    assert [problem.split(" window")[0] for problem in problems] == ["the validation", "the test"]
    with pytest.raises(SplitError, match="cannot select anything"):
        run_search(RandomGrammarProposer(0), market.panel, splits, config, run_dir=tmp_path / "run")
    assert not (tmp_path / "run").exists()  # nothing was spent or written
    # too few names for a rank IC on any date
    narrow = window_capacity_problems(market.panel.dates, splits, tiny_search_config(), n_symbols=6)
    assert narrow and "6 symbols" in narrow[-1]
    assert window_capacity_problems(market.panel.dates, splits, tiny_search_config(), n_symbols=24) == []
    # a loader is checked on the formation window it has loaded (the registry checks the rest)
    with pytest.raises(SplitError, match="formation window"):
        run_search(
            RandomGrammarProposer(0), lambda end: truncate_panel(market.panel, end) if end is not None else market.panel,
            splits, tiny_search_config(min_ic_dates=200),
        )


def test_degenerate_validation_is_reported_as_a_warning() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    # most names have no bars during the first 35 validation sessions: rank ICs are undefined there,
    # so the screened candidates have too few validation IC dates (the formation window is intact)
    dates = pd.to_datetime(market.bars["date"])
    validation = market.panel.dates[market.panel.dates >= splits.validation.start][:35]
    keep = ~(dates.isin(validation) & ~market.bars["symbol"].isin(market.panel.symbols[:4]))
    panel = panel_from_bars(market.bars.loc[keep].reset_index(drop=True))
    result = run_search(
        ScriptedProposer([["-ts_mean(returns, 5)", "volume / ts_mean(volume, 20)"]]), panel, splits,
        tiny_search_config(budget=2),
    )
    assert result.funnel["n_screen_survivors"] >= 1
    assert result.funnel["n_rejected_degenerate_validation"] == result.funnel["n_screen_survivors"]
    assert result.frozen.factors == ()
    assert any("never tested on validation" in warning for warning in result.summary()["warnings"])


def test_round_cap_scales_with_the_budget() -> None:
    from llm_factor_mining.search import default_max_rounds

    assert SearchConfig().max_rounds == 100  # the default arm: 200 trials in batches of 20
    assert SearchConfig(budget=10_000, batch_size=20).max_rounds == 5_000
    assert SearchConfig(budget=10_000, max_rounds=7).max_rounds == 7  # an explicit cap is kept
    assert default_max_rounds(150, 1) == 1_500
    config = SearchConfig(budget=10_000, batch_size=20)
    assert SearchConfig.from_dict(config.to_dict()) == config
    # 120 trials in batches of 1 would have hit a fixed 100-round cap
    market = small_market()
    result = run_search(
        RandomGrammarProposer(4), market.panel, small_splits(market.panel), tiny_search_config(budget=120, batch_size=1),
        reveal=False,
    )
    assert result.complete and result.n_trials == 120 and result.n_rounds > 100


def test_proposers_never_see_date_counts_or_raw_exception_text(monkeypatch) -> None:
    import llm_factor_mining.search as search_module
    from llm_factor_mining.search import DEGENERATE_FEEDBACK

    market = small_market()
    splits = small_splits(market.panel)
    formation = truncate_panel(market.panel, splits.formation.end)
    n_scorable = len(embargoed_signal_dates(formation.dates, splits.formation, lag=1, horizon=1))
    responses = [
        {"proposals": [{"expression": "delay(close, 120)", "rationale": "r", "economic_mechanism": "m"}]},
        {"proposals": [{"expression": "close", "rationale": "r", "economic_mechanism": "m"}]},
    ]
    backend = FakeBackend(responses)
    result = run_search(
        LLMProposer(backend, forbidden_tokens=market.panel.symbols), market.panel, splits,
        tiny_search_config(budget=2, batch_size=1, min_ic_dates=40, min_ic_fraction=0.9),
    )
    degenerate = result.trials[0]
    assert degenerate.status == "degenerate" and str(n_scorable) in degenerate.message  # the ledger keeps counts
    lines = [line for line in backend.calls[1]["user"].splitlines() if "delay(close,120)" in line]
    assert lines == [f"- delay(close,120) -> DEGENERATE: {DEGENERATE_FEEDBACK}"]
    assert not any(char.isdigit() for char in DEGENERATE_FEEDBACK)

    class Failing(search_module.Evaluator):
        def evaluate(self, node):
            if "low" in str(node):
                raise ValueError(f"operands could not be broadcast together with shapes ({n_scorable},24)")
            return super().evaluate(node)

    monkeypatch.setattr(search_module, "Evaluator", Failing)
    backend = FakeBackend(
        [
            {"proposals": [{"expression": "low", "rationale": "r", "economic_mechanism": "m"}]},
            {"proposals": [{"expression": "close", "rationale": "r", "economic_mechanism": "m"}]},
        ]
    )
    result = run_search(
        LLMProposer(backend), market.panel, splits, tiny_search_config(budget=2, batch_size=1), reveal=False
    )
    assert result.trials[0].status == "error" and str(n_scorable) in result.trials[0].message
    lines = [line for line in backend.calls[1]["user"].splitlines() if line.startswith("- low ->")]
    assert lines == ["- low -> EVAL_ERROR: evaluation failed (ValueError)"]
