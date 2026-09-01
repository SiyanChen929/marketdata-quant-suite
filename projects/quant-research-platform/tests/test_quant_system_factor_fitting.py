from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.config import AppConfig, RegimeConfig
from quant_system.optimization.factor_fitting import fit_factor_model
from quant_system.strategies.base import StrategyContext


def _fit_features() -> pd.DataFrame:
    rng = np.random.default_rng(17)
    dates = pd.bdate_range("2023-01-02", periods=280)
    symbols = [f"S{i:03d}" for i in range(80)]
    rows = []
    for idx, symbol in enumerate(symbols):
        alpha = (idx - 40) / 40
        close = 30 + idx
        for date in dates:
            close *= 1.0 + 0.0007 * alpha + rng.normal(0, 0.001)
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000_000 + idx * 100,
                    "ret_21d": alpha + rng.normal(0, 0.01),
                    "rs_21d": alpha + rng.normal(0, 0.01),
                    "ret_63d": alpha + rng.normal(0, 0.01),
                    "rs_63d": alpha + rng.normal(0, 0.01),
                    "vol_20d": 0.02 + abs(alpha) * 0.002,
                    "ma_20": close * 0.99,
                    "ma_50": close * 0.96,
                    "ma_200": close * 0.91,
                    "high_252": close * 1.02,
                    "low_252": close * 0.70,
                    "atr_14": close * 0.02,
                    "volume_ma_20": 1_000_000,
                    "tradable": True,
                }
            )
    return pd.DataFrame(rows)


def test_factor_fit_reports_train_validation_forward_and_weights() -> None:
    config = AppConfig(
        regime=RegimeConfig(
            train_start="2023-01-02",
            train_end="2023-06-30",
            validation_start="2023-07-03",
            validation_end="2023-10-31",
            forward_start="2023-11-01",
        )
    )

    tables = fit_factor_model(_fit_features(), StrategyContext(), config, horizon=5, max_factors=5, min_names_per_date=30)

    assert not tables["weights"].empty
    assert abs(tables["weights"]["fitted_weight"].abs().sum() - 1.0) < 1e-8
    assert (tables["weights"]["fitted_weight"] >= -1e-12).all()
    assert {"train", "validation", "forward"}.issubset(set(tables["performance"]["period"]))
    assert "fitted_factor_score" in tables["latest_scores"].columns
    forward = tables["performance"][tables["performance"]["period"].eq("forward")].iloc[0]
    assert not bool(forward["used_for_fit"])
