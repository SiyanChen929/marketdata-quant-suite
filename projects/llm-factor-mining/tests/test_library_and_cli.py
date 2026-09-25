from __future__ import annotations

import json

import numpy as np
import pytest

from lfm_helpers import make_bars
from llm_factor_mining.cli import main
from llm_factor_mining.data import panel_from_bars
from llm_factor_mining.dsl import LIBRARY, OPERATORS, library_nodes, validate
from llm_factor_mining.dsl.library import get
from llm_factor_mining.evaluate import Evaluator, MetricConfig, evaluate_factor


def test_library_entries_are_valid_and_distinct() -> None:
    names = [factor.name for factor in LIBRARY]
    assert len(names) == len(set(names))
    hashes = [factor.hash for factor in LIBRARY]
    assert len(hashes) == len(set(hashes))
    for factor in LIBRARY:
        result = validate(factor.expression)
        assert result.ok, (factor.name, result.errors)
        assert factor.family and factor.rationale
    assert len(library_nodes()) == len(LIBRARY)
    assert get("momentum_12_1").family == "momentum"
    with pytest.raises(KeyError):
        get("does_not_exist")


def test_library_covers_operator_families() -> None:
    used = set().union(*(validate(factor.expression).stats.operators for factor in LIBRARY))
    categories = {OPERATORS[name].category for name in used}
    assert categories == {"elementwise", "time_series", "cross_sectional"}


def test_library_evaluates_on_synthetic_panel() -> None:
    panel = panel_from_bars(make_bars(300, 15, seed=31))
    engine = Evaluator(panel)
    for factor in LIBRARY:
        values = engine.evaluate(factor.expression)
        tail = values.iloc[-20:].to_numpy()
        assert np.isfinite(tail).mean() > 0.9, factor.name
    results = evaluate_factor(
        panel, get("low_volatility_20d").expression, config=MetricConfig(min_names=10), evaluator=engine
    )
    assert results["full"].report.n_ic_dates > 200


def test_cli_validate_library_and_grammar(capsys) -> None:
    assert main(["validate", "cs_rank(-ts_std(returns, 20))"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] and payload["canonical"] == "cs_rank(neg(ts_std(returns,20)))"
    assert len(payload["hash"]) == 64

    assert main(["validate", "ts_mean(close, 0)"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["errors"][0]["code"] == "WINDOW_RANGE"

    assert main(["library"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == len(LIBRARY)
    assert {json.loads(line)["name"] for line in lines} == {factor.name for factor in LIBRARY}

    assert main(["grammar"]) == 0
    assert "ts_corr" in capsys.readouterr().out


def test_library_labels_reproduced_published_alphas() -> None:
    sources = {factor.name: factor.source for factor in LIBRARY}
    assert sources["volume_change_vs_intraday_return"] == "Kakushadze (2016), Alpha#2"
    assert sources["open_volume_corr_10d"] == "Kakushadze (2016), Alpha#6"
    assert sources["rank_volume_close_cov_5d"] == "Kakushadze (2016), Alpha#13"
    for factor in LIBRARY:
        if factor.kakushadze_style and "Alpha#" not in factor.source:
            assert factor.source == "project variant in the style of Kakushadze (2016)"
