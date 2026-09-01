from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.config import AppConfig, RegimeConfig
from quant_system.optimization.factor_mining import (
    add_activation_diagnostics,
    build_factor_mining_panel,
    decorrelate_factor_candidates,
    factor_activation_diagnostics,
    mine_factors,
)
from quant_system.strategies.base import StrategyContext


def _factor_features() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2023-01-02", periods=260)
    symbols = [f"S{i:03d}" for i in range(70)]
    rows = []
    for idx, symbol in enumerate(symbols):
        alpha = (idx - 35) / 35
        close = 50 + idx
        for date in dates:
            close *= 1.0 + 0.0004 * alpha + rng.normal(0, 0.001)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000 + idx * 1_000,
                    "ret_21d": alpha + rng.normal(0, 0.02),
                    "rs_21d": alpha + rng.normal(0, 0.02),
                    "vol_20d": 0.02 + abs(alpha) * 0.01,
                    "ma_20": close * 0.98,
                    "ma_50": close * 0.95,
                    "ma_200": close * 0.90,
                    "high_252": close * 1.02,
                    "low_252": close * 0.70,
                    "atr_14": close * 0.03,
                    "volume_ma_20": 1_000_000,
                    "tradable": True,
                }
            )
    return pd.DataFrame(rows)


def test_build_factor_mining_panel_adds_forward_returns_and_derived_factors() -> None:
    panel = build_factor_mining_panel(_factor_features(), StrategyContext(), horizons=(5,))

    assert "fwd_5d" in panel.columns
    assert "price_vs_ma_20" in panel.columns
    assert "mom_63_skip5" in panel.columns
    assert panel["fwd_5d"].notna().any()


def test_mine_factors_returns_candidates_without_using_forward_for_selection() -> None:
    config = AppConfig(
        regime=RegimeConfig(
            train_start="2023-01-02",
            train_end="2023-06-30",
            validation_start="2023-07-03",
            validation_end="2023-10-31",
            forward_start="2023-11-01",
        )
    )

    tables = mine_factors(_factor_features(), StrategyContext(), config, horizons=(5,), max_factors=20, min_names_per_date=30)
    summary = tables["summary"]

    assert not summary.empty
    assert "forward_mean_ic_aligned_not_used" in summary.columns
    assert summary["selection_note"].str.contains("not used").all()
    assert (summary["candidate_status"] == "candidate").any()


def test_decorrelate_factor_candidates_keeps_representatives() -> None:
    selected = pd.DataFrame(
        [
            {"factor": "a", "selection_score": 0.10},
            {"factor": "b", "selection_score": 0.09},
            {"factor": "c", "selection_score": 0.08},
        ]
    )
    corr = pd.DataFrame(
        [
            {"factor_a": "a", "factor_b": "b", "abs_correlation": 0.95},
            {"factor_a": "a", "factor_b": "c", "abs_correlation": 0.20},
        ]
    )

    out = decorrelate_factor_candidates(selected, corr, max_abs_corr=0.80)

    assert out[out["factor"].eq("a")].iloc[0]["decorrelation_status"] == "keep"
    b = out[out["factor"].eq("b")].iloc[0]
    assert b["decorrelation_status"] == "drop_collinear"
    assert b["representative_factor"] == "a"


def test_factor_activation_diagnostics_detects_changed_top_names() -> None:
    config = AppConfig(
        regime=RegimeConfig(
            train_start="2023-01-02",
            train_end="2023-06-30",
            validation_start="2023-07-03",
            validation_end="2023-10-31",
            forward_start="2023-11-01",
        )
    )
    features = _factor_features()
    panel = build_factor_mining_panel(features, StrategyContext(), horizons=(5,))
    scores = features[["date", "symbol", "rs_21d"]].copy()
    # Make the baseline deliberately stale so a useful factor must replace names.
    scores["final_score"] = -pd.to_numeric(scores["rs_21d"], errors="coerce")
    selected = pd.DataFrame(
        [
            {
                "factor": "rs_21d",
                "direction": "high_is_good",
                "selection_score": 0.1,
                "candidate_status": "candidate",
                "decorrelation_status": "keep",
            }
        ]
    )

    activation = factor_activation_diagnostics(panel, scores, selected, config, horizon=5, top_n=10, min_names_per_date=30)

    validation = activation[activation["period"].eq("validation")].iloc[0]
    assert validation["mean_changed_names"] > 0
    assert validation["mean_overlap_rate"] < 1.0
    assert validation["mean_forward_return_lift"] > 0


def test_add_activation_diagnostics_adds_table() -> None:
    config = AppConfig(
        regime=RegimeConfig(
            train_start="2023-01-02",
            train_end="2023-06-30",
            validation_start="2023-07-03",
            validation_end="2023-10-31",
            forward_start="2023-11-01",
        )
    )
    features = _factor_features()
    tables = mine_factors(features, StrategyContext(), config, horizons=(5,), max_factors=20, min_names_per_date=30)
    signals = features[["date", "symbol", "rs_21d"]].copy()
    signals["final_score"] = -pd.to_numeric(signals["rs_21d"], errors="coerce")

    out = add_activation_diagnostics(tables, signals, config, horizon=5, top_n=10, min_names_per_date=30)

    assert "activation" in out
    assert not out["activation"].empty
