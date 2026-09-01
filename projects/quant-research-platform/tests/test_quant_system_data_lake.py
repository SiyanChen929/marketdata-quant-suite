from __future__ import annotations

import pandas as pd
import pytest

from quant_system.data.lake import QuantSystemLake
from quant_system.data.lake_provider import LakeMarketDataProvider


def test_quant_lake_materializes_and_reads_daily_prices(tmp_path):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "date": ["2024-01-02", "2024-01-03", "2024-01-03"],
            "open": [10.0, 10.5, 20.0],
            "high": [11.0, 11.2, 21.0],
            "low": [9.5, 10.1, 19.5],
            "close": [10.8, 11.0, 20.5],
            "adj_close": [10.8, 11.0, 20.5],
            "volume": [1000, 1200, 2000],
            "source": ["synthetic", "synthetic", "synthetic"],
            "finality": ["confirmed", "confirmed", "confirmed"],
        }
    ).to_csv(csv_path, index=False)
    lake = QuantSystemLake(tmp_path / "lake")

    lake.materialize_daily_from_csv(csv_path)
    out = lake.read_daily(["AAA"], "2024-01-03", None)

    assert lake.has_fresh_daily(csv_path)
    assert out["symbol"].tolist() == ["AAA"]
    assert out["close"].tolist() == [11.0]
    assert out["finality"].tolist() == ["confirmed"]
    assert lake.daily_symbols() == ["AAA", "BBB"]


def test_lake_market_data_provider_uses_daily_lake(tmp_path):
    csv_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-02"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.8],
            "adj_close": [10.8],
            "volume": [1000],
            "source": ["synthetic"],
            "finality": ["confirmed"],
        }
    ).to_csv(csv_path, index=False)
    lake = QuantSystemLake(tmp_path / "lake")
    lake.materialize_daily_from_csv(csv_path)

    provider = LakeMarketDataProvider(lake)
    out = provider.get_bulk_ohlcv(["AAA"], "2024-01-01", None)

    assert len(out) == 1
    assert out.loc[0, "adj_close"] == 10.8


def test_daily_lake_rejects_unlabeled_or_provisional_rows(tmp_path):
    base = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "date": ["2024-01-02"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.5],
            "close": [10.8],
            "adj_close": [10.8],
            "volume": [1000],
        }
    )
    lake = QuantSystemLake(tmp_path / "lake")
    unlabeled = tmp_path / "unlabeled.csv"
    base.to_csv(unlabeled, index=False)
    with pytest.raises(ValueError, match="source.*finality|finality.*source"):
        lake.materialize_daily_from_csv(unlabeled)

    provisional = tmp_path / "provisional.csv"
    base.assign(source="marketdata.app", finality="provisional").to_csv(provisional, index=False)
    with pytest.raises(ValueError, match="confirmed"):
        lake.materialize_daily_from_csv(provisional)


def test_quant_lake_materializes_and_reads_intraday(tmp_path):
    csv_path = tmp_path / "intraday.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "datetime": ["2024-01-02 09:30:00", "2024-01-02 09:35:00", "2024-01-03 09:30:00"],
            "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
            "open": [10.0, 10.2, 20.0],
            "high": [10.3, 10.4, 20.5],
            "low": [9.9, 10.1, 19.8],
            "close": [10.2, 10.3, 20.2],
            "volume": [1000, 1200, 2000],
        }
    ).to_csv(csv_path, index=False)
    lake = QuantSystemLake(tmp_path / "lake")

    lake.materialize_intraday_from_csv(csv_path, "5min")
    lake.materialize_intraday_summary_from_csv(csv_path, "5min")
    out = lake.read_intraday(["AAA"], "2024-01-02", "2024-01-02", "5min")
    summary = lake.read_intraday_summary(["AAA"], "2024-01-02", "2024-01-02", "5min")

    assert lake.has_fresh_intraday(csv_path, "5min")
    assert len(out) == 2
    assert out["symbol"].unique().tolist() == ["AAA"]
    assert lake.has_fresh_intraday_summary(csv_path, "5min")
    assert len(summary) == 1
    assert summary.loc[0, "first_open"] == 10.0
    assert summary.loc[0, "last_close"] == 10.3
