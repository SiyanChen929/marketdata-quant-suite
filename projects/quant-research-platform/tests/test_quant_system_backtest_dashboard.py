from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_system.backtest.dashboard import (
    comparison_curves,
    discover_runs,
    enrich_equity_curve,
    monthly_return_matrix,
    trade_activity,
)


def _write_run(root, name: str, values: list[float]):
    run = root / name
    run.mkdir(parents=True)
    (run / "metrics.json").write_text(json.dumps({"total_return": values[-1] / values[0] - 1, "sharpe": 1.2}), encoding="utf-8")
    pd.DataFrame({"date": pd.date_range("2024-01-30", periods=len(values), freq="D"), "equity": values}).to_csv(run / "equity_curve.csv", index=False)
    return run


def test_discover_and_compare_runs(tmp_path):
    first = _write_run(tmp_path, "first", [100.0, 105.0, 103.0])
    second = _write_run(tmp_path, "second", [200.0, 210.0, 220.0])

    discovered = discover_runs(tmp_path)
    curves, metrics = comparison_curves([first, second])

    assert set(discovered["run_id"]) == {"first", "second"}
    assert set(curves["run_id"]) == {"first", "second"}
    assert curves.groupby("run_id")["normalized_equity"].first().eq(100.0).all()
    assert set(metrics["run_id"]) == {"first", "second"}


def test_equity_diagnostics_monthly_returns_and_trade_activity():
    dates = pd.date_range("2024-01-30", periods=5, freq="D")
    equity = pd.DataFrame({"date": dates, "equity": [100.0, 110.0, 99.0, 101.0, 102.0]})
    enriched = enrich_equity_curve(equity, rolling_window=5)
    monthly = monthly_return_matrix(equity)
    trades = pd.DataFrame(
        {
            "date": ["2024-01-30", "2024-01-30"],
            "symbol": ["AAA", "BBB"],
            "quantity": [2, 3],
            "price": [10, 20],
            "fee": [1, 2],
            "estimated_liquidity_cost": [0.5, 0.7],
        }
    )
    activity = trade_activity(trades)

    assert enriched["drawdown"].min() == pytest.approx(-0.1)
    assert enriched["normalized_equity"].iloc[0] == 100.0
    assert list(monthly.columns) == list(range(1, 13))
    assert activity.loc[0, "notional"] == 80
    assert activity.loc[0, "fees"] == 3
