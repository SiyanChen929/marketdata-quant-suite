from __future__ import annotations

import pandas as pd
import pytest

from quant_system.cli import _merge_downloaded_ohlcv


def test_download_merge_preserves_existing_newer_rows(tmp_path) -> None:
    path = tmp_path / "marketdata.csv"
    existing = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2026-05-06", "open": 10, "high": 10, "low": 10, "close": 10, "adj_close": 10, "volume": 100, "source": "marketdata.app", "finality": "confirmed"},
            {"symbol": "AAA", "date": "2026-05-08", "open": 12, "high": 12, "low": 12, "close": 12, "adj_close": 12, "volume": 120, "source": "marketdata.app", "finality": "confirmed"},
        ]
    )
    existing.to_csv(path, index=False)
    downloaded = pd.DataFrame(
        [
            {"symbol": "AAA", "date": "2026-05-06", "open": 11, "high": 11, "low": 11, "close": 11, "adj_close": 11, "volume": 110, "source": "marketdata.app", "finality": "confirmed"},
        ]
    )

    merged = _merge_downloaded_ohlcv(path, downloaded)

    by_date = merged.set_index("date")
    assert by_date.loc["2026-05-06", "close"] == 11
    assert by_date.loc["2026-05-08", "close"] == 12
    assert merged["date"].max() == "2026-05-08"


def test_download_merge_preserves_existing_when_provider_empty(tmp_path) -> None:
    path = tmp_path / "marketdata.csv"
    existing = pd.DataFrame(
        [{"symbol": "AAA", "date": "2026-05-08", "open": 12, "high": 12, "low": 12, "close": 12, "adj_close": 12, "volume": 120, "source": "marketdata.app", "finality": "confirmed"}]
    )
    existing.to_csv(path, index=False)

    merged = _merge_downloaded_ohlcv(path, pd.DataFrame())

    assert len(merged) == 1
    assert merged.iloc[0]["date"] == "2026-05-08"


def test_download_merge_rejects_unlabeled_existing_cache(tmp_path) -> None:
    path = tmp_path / "marketdata.csv"
    pd.DataFrame(
        [{"symbol": "AAA", "date": "2026-05-08", "open": 12, "high": 12, "low": 12, "close": 12, "adj_close": 12, "volume": 120}]
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="source/finality"):
        _merge_downloaded_ohlcv(path, pd.DataFrame())
