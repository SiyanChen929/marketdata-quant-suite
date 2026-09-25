from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from lfm_helpers import small_market, small_splits
from llm_factor_mining.benchmark import (
    BANNER,
    BenchmarkConfig,
    PlantedFactor,
    SyntheticMarketConfig,
    best_matches,
    recovery_matrix,
    render_markdown,
    run_benchmark,
    simulate_market,
    write_summary,
)
from llm_factor_mining.dsl.nodes import Call
from llm_factor_mining.dsl.parser import parse
from llm_factor_mining.evaluate import Evaluator, MetricConfig, evaluate_signal
from llm_factor_mining.proposers import FakeBackend, LLMProposer
from quant_marketdata import CANONICAL_COLUMNS


def test_market_bars_follow_the_canonical_confirmed_contract() -> None:
    market = small_market()
    assert list(market.bars.columns) == list(CANONICAL_COLUMNS)
    assert set(market.bars["source"]) == {"synthetic"} and set(market.bars["finality"]) == {"confirmed"}
    panel = market.panel
    assert panel.finality == "confirmed" and panel.sources == ("synthetic",)
    assert (panel.high >= panel.close).all().all() and (panel.low <= panel.open).all().all()
    assert (panel.low > 0).all().all() and (panel.volume >= 1).all().all()
    assert market.residual <= market.config.tol and market.iterations >= 2


def test_market_is_deterministic_by_seed() -> None:
    config = SyntheticMarketConfig(n_symbols=20, n_dates=150, snr=0.2, seed=4)
    first, second = simulate_market(config), simulate_market(config)
    assert first.panel.content_sha256 == second.panel.content_sha256
    other = simulate_market(SyntheticMarketConfig(n_symbols=20, n_dates=150, snr=0.2, seed=5))
    assert other.panel.content_sha256 != first.panel.content_sha256


def test_planted_returns_solve_the_fixed_point_exactly() -> None:
    # the realized returns equal base + s * sum_k w_k z_k[t - lag - 1] with z_k from the final prices
    config = SyntheticMarketConfig(n_symbols=20, n_dates=160, snr=0.5, seed=2)
    market = simulate_market(config)
    null = simulate_market(SyntheticMarketConfig(n_symbols=20, n_dates=160, snr=0.0, seed=2))
    returns = market.panel.close.pct_change().to_numpy()
    base = null.panel.close.pct_change().to_numpy()
    alpha = np.zeros_like(returns)
    for factor in config.planted:
        z = market.planted_signals[factor.name].to_numpy()
        shifted = np.full_like(z, np.nan)
        shifted[2:] = z[:-2]
        alpha += market.weights[factor.name] * np.nan_to_num(shifted)
    alpha *= market.alpha_scale
    np.testing.assert_allclose(returns[1:], (base + alpha)[1:], atol=1e-10)


def test_planted_signal_is_tradeable_after_the_execution_lag() -> None:
    strong = small_market(snr=0.6)
    config = MetricConfig(min_names=10)
    for name, signal in strong.planted_signals.items():
        report = evaluate_signal(signal, strong.panel.close, config=config).report
        assert report.ic_mean > 0.05 and report.ic_tstat_nw > 3, name
    null = simulate_market(SyntheticMarketConfig(n_symbols=24, n_dates=260, snr=0.0, seed=0))
    for name, signal in null.planted_signals.items():
        report = evaluate_signal(signal, null.panel.close, config=config).report
        assert abs(report.ic_tstat_nw) < 3.5, name


def test_planted_signals_match_their_dsl_definition() -> None:
    market = small_market()
    engine = Evaluator(market.panel)
    for factor in market.config.planted:
        expected = engine.evaluate(Call("cs_zscore", (parse(factor.expression),)))
        pd.testing.assert_frame_equal(expected, market.planted_signals[factor.name])


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        PlantedFactor("bad", "ts_mean(close, 0)")
    with pytest.raises(ValueError):
        SyntheticMarketConfig(planted=(PlantedFactor("a", "close"), PlantedFactor("a", "open")))
    with pytest.raises(ValueError):
        BenchmarkConfig(proposers=("oracle",))


def test_recovery_matrix_identifies_the_planted_signal() -> None:
    market = small_market()
    splits = small_splits(market.panel)
    dates = market.panel.dates[market.panel.dates >= splits.test.start]
    engine = Evaluator(market.panel)
    candidates = {
        "same": market.planted_signals["reversal_5"],
        "flipped": -engine.evaluate("ts_sum(returns, 5)"),
        "noise": engine.evaluate("high"),
    }
    matrix = recovery_matrix(candidates, market.planted_signals, dates, min_names=10)
    assert matrix["reversal_5"]["same"] == pytest.approx(1.0)
    assert matrix["reversal_5"]["flipped"] > 0.99
    best = best_matches(matrix)
    assert best["reversal_5"]["best"] in {"same", "flipped"}
    assert best["reversal_5"]["abs_rho"] == pytest.approx(1.0)


def test_benchmark_smoke_run_and_report(tmp_path) -> None:
    config = BenchmarkConfig(
        seeds=(0,), snr_levels=(0.4,), null_seeds=(1,), n_symbols=20, n_dates=240, budget=16, batch_size=8,
        proposers=("random", "evolutionary", "llm", "random@24"), min_ic_dates=30,
    )
    summary = run_benchmark(config, llm_status="not run: no credentials", runs_dir=tmp_path / "runs")
    assert summary["banner"] == BANNER
    assert summary["proposers"] == {"llm": "not run: no credentials", "random": "run", "evolutionary": "run", "random@24": "run"}
    assert {run["arm"] for run in summary["runs"]} == {"random", "evolutionary", "random@24"}
    for run in summary["runs"]:
        assert run["n_trials"] == run["budget"] and run["complete"]
        assert 0.0 <= run["recovery_rate"] <= 1.0 and 0.0 <= run["recall_rate"] <= 1.0
        assert run["n_selected"] <= run["n_confirmed"] <= run["n_screen_survivors"] <= run["n_behaviour_classes"]
        assert len(run["ledger_head"]) == 64
        if float(run["snr"]) == 0.0 and run["n_selected"]:
            assert run["fdp_true"] == 1.0  # nothing is planted in a null market
    assert {row["arm"] for row in summary["aggregate"]} == {"random", "evolutionary", "random@24"}
    assert {row["snr"] for row in summary["null_aggregate"]} == {0.0}
    assert summary["markets"][0]["oracle_test"]["composite"]["ic_mean"] > 0
    assert 0.0 <= summary["markets"][0]["oracle_test"]["power"] <= 1.0
    novelty = summary["markets"][0]["planted_behavioural_novelty"]
    assert novelty["reversal_5"]["novelty"] < 0.1 and novelty["drawn_40"]["novelty"] > 0.5
    assert (tmp_path / "runs" / "snr0.4_seed0_random" / "ledger.jsonl").is_file()
    assert (tmp_path / "runs" / "snr0.4_seed0_random_b24" / "ledger.jsonl").is_file()

    json_path, md_path = write_summary(summary, tmp_path / "out", command="python run.py --tiny")
    payload = json.loads(json_path.read_text())
    assert payload["command"] == "python run.py --tiny"
    text = md_path.read_text()
    # A fresh run must leave summary.md exactly as re-rendered from summary.json
    # (test_paper_tables.py checks the committed pair the same way).
    assert text == render_markdown(payload, command="python run.py --tiny")
    assert BANNER in text and "python run.py --tiny" in text and "not run: no credentials" in text
    assert render_markdown(payload, command="x").count("| 0.40 |") >= 3
    assert "## Null markets (snr = 0): false selections" in text and "None: every run used its full budget." in text


def test_incomplete_and_aborted_runs_are_flagged_not_averaged() -> None:
    from llm_factor_mining.proposers.llm import LLMAPIError

    class Broken(FakeBackend):
        def complete_json(self, system, user, schema, *, request_tag=""):
            raise LLMAPIError("API returned status 418", {"status_code": 418})

    config = BenchmarkConfig(
        seeds=(0,), snr_levels=(0.4,), null_seeds=(), n_symbols=20, n_dates=240, budget=6, batch_size=3,
        proposers=("random",), min_ic_dates=30,
    )
    summary = run_benchmark(config, llm_factory=lambda seed, **kw: LLMProposer(Broken([]), replicate=seed, **kw))
    llm_run = next(run for run in summary["runs"] if run["arm"] == "llm")
    assert llm_run["aborted"] and not llm_run["complete"] and "recovery_rate" not in llm_run
    assert summary["proposers"]["llm"].startswith("run; 1 of 1 runs incomplete")
    assert summary["incomplete_runs"] == [
        {"arm": "llm", "snr": 0.4, "seed": 0, "stop_reason": "proposer_failures", "n_trials": 0, "aborted": True}
    ]
    llm_row = next(row for row in summary["aggregate"] if row["arm"] == "llm")
    assert llm_row["n_runs"] == 1 and llm_row["n_complete"] == 0
    assert llm_row["metrics"]["recovery_rate"]["mean"] is None


def test_recovery_of_oriented_factors_is_signed() -> None:
    matrix = {"planted": {"cand": -0.9, "other": 0.2}}
    assert best_matches(matrix)["planted"] == {"best": "cand", "abs_rho": 0.9, "rho": -0.9}
    signed = best_matches(matrix, signed=True)["planted"]
    assert signed["best"] == "other" and signed["rho"] == 0.2  # an anti-correlated factor recovers nothing


def test_every_planted_signal_is_reachable_by_every_baseline() -> None:
    import math

    from llm_factor_mining.benchmark import DEFAULT_PLANTED, PLANTED_FAMILIES
    from llm_factor_mining.dsl.validate import DSLLimits
    from llm_factor_mining.proposers.trees import GrammarConfig, strip_orientation, tree_log_probability

    grammar, limits = GrammarConfig(), DSLLimits()
    assert {factor.family for factor in DEFAULT_PLANTED} == set(PLANTED_FAMILIES)
    for factor in DEFAULT_PLANTED:
        node = strip_orientation(factor.node)  # the protocol orients factors, so the sign is free
        # random grammar (A): non-zero probability of sampling this exact tree
        assert math.isfinite(tree_log_probability(node, grammar)), factor.name
        # genetic programming (B): every component is on its menus and the tree fits its limits
        assert math.isfinite(tree_log_probability(factor.node, grammar, max_depth=limits.max_depth)), factor.name
    # the window-1 change of the library-family signal was unreachable before 1 joined the menu
    no_one = GrammarConfig(windows=(2, 3, 5, 10, 20, 40, 60))
    corr = strip_orientation(parse("-ts_corr(returns, delta(log(volume), 1), 10)"))
    assert tree_log_probability(corr, no_one) == -math.inf


def test_planted_signal_draw_is_reproducible() -> None:
    from pathlib import Path

    from llm_factor_mining.benchmark import DRAWN_PLANTED, FIXED_PLANTED
    from llm_factor_mining.benchmark.draw import assess_candidate, candidate_tree
    from llm_factor_mining.dsl.library import LIBRARY
    from llm_factor_mining.dsl.nodes import to_expression

    record = json.loads(
        (Path(__file__).resolve().parents[1] / "results" / "planted_signal_draw.json").read_text(encoding="utf-8")
    )
    accepted = record["accepted"]
    assert to_expression(candidate_tree(accepted["k"])) == accepted["expression"] == DRAWN_PLANTED.expression
    assert [item["k"] for item in record["rejected"]] == list(range(accepted["k"]))
    for item in record["rejected"]:
        assert to_expression(candidate_tree(item["k"])) == item["expression"] and item["reasons"]
    market = simulate_market(SyntheticMarketConfig(snr=0.0, seed=424242, planted=FIXED_PLANTED))
    engine = Evaluator(market.panel, max_cache_entries=64)
    references = {factor.name: engine.evaluate(factor.node) for factor in LIBRARY}
    references.update({f"planted:{name}": signal for name, signal in market.planted_signals.items()})
    check = assess_candidate(
        candidate_tree(accepted["k"]), engine, references, fixed_planted=FIXED_PLANTED, check_convergence=False
    )
    assert check["reasons"] == [] and check["max_abs_corr"] == pytest.approx(accepted["max_abs_corr"])


def test_benchmark_llm_arm_with_fake_backend(tmp_path) -> None:
    def respond(system, user, schema):
        return {
            "proposals": [
                {"expression": "-ts_mean(returns, 5)", "rationale": "r", "economic_mechanism": "m"},
                {"expression": "volume / ts_mean(volume, 20)", "rationale": "r", "economic_mechanism": "m"},
            ]
        }

    config = BenchmarkConfig(
        seeds=(0,), snr_levels=(0.4,), null_seeds=(), n_symbols=20, n_dates=240, budget=6, batch_size=4,
        proposers=("random",),
    )
    backends: list[FakeBackend] = []
    proposers: list[LLMProposer] = []

    def factory(seed, *, run_tag="", forbidden_tokens=()):
        backends.append(FakeBackend(respond))
        proposers.append(LLMProposer(backends[-1], replicate=seed, run_tag=run_tag, forbidden_tokens=forbidden_tokens))
        return proposers[-1]

    summary = run_benchmark(config, llm_factory=factory)
    # the scripted model repeats itself, so its run stalls: it is scored but flagged, never averaged in
    assert summary["proposers"] == {"random": "run", "llm": "run; 1 of 1 runs incomplete (stop reasons: stalled)"}
    llm_run = next(run for run in summary["runs"] if run["proposer"] == "llm")
    assert not llm_run["complete"] and llm_run["n_trials"] == 2
    assert next(row for row in summary["aggregate"] if row["arm"] == "llm")["n_complete"] == 0
    assert llm_run["served_models"] == ["fake-model"]
    assert llm_run["recall_best_abs_rho"]["reversal_5"] > 0.99
    # each market cell gets its own request tag, and prompts are checked against the panel's symbols
    assert backends[0].calls[0]["tag"] == "replicate=10000;snr=0.4,market_seed=0"
    assert proposers[0].forbidden_tokens == tuple(f"SYN{index:03d}" for index in range(20))
