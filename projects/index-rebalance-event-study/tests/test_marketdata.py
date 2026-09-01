from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import pytest

from index_rebalance_event_study.marketdata import (
    CanonicalBarError,
    ConfirmedDailyBars,
    external_data_home,
    validate_confirmed_daily_bars,
)


def _bars(finality: str = "confirmed") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2035-01-09", "2035-01-09"],
            "symbol": ["ZZZADD", "ZZZDEL"],
            "open": [100.0, 50.0],
            "high": [102.0, 51.0],
            "low": [99.0, 48.0],
            "close": [101.0, 49.0],
            "volume": [1_000, 2_000],
            "source": ["synthetic_test", "synthetic_test"],
            "finality": [finality, finality],
        }
    )


@dataclass
class FakeClient:
    response: pd.DataFrame
    calls: list[dict[str, object]] = field(default_factory=list)

    def get_bulk_daily_bars(self, **kwargs: object) -> pd.DataFrame:
        self.calls.append(kwargs)
        return self.response.copy()


def test_adapter_forces_confirmed_bulk_request() -> None:
    client = FakeClient(_bars())
    adapter = ConfirmedDailyBars(client)
    result = adapter.get_bulk_daily_bars(
        ["zzzdel", "ZZZADD", "ZZZADD"],
        "2035-01-09",
        "2035-01-09",
    )
    assert len(result) == 2
    assert client.calls == [
        {
            "symbols": ["ZZZADD", "ZZZDEL"],
            "start": "2035-01-09",
            "end": "2035-01-09",
            "finality": "confirmed",
            "refresh": False,
        }
    ]


def test_provisional_bar_is_rejected() -> None:
    with pytest.raises(CanonicalBarError, match="confirmed"):
        validate_confirmed_daily_bars(_bars(finality="provisional"))


def test_incoherent_ohlc_is_rejected() -> None:
    bars = _bars()
    bars.loc[0, "high"] = 98.0
    with pytest.raises(CanonicalBarError, match="OHLC"):
        validate_confirmed_daily_bars(bars)


def test_duplicate_date_symbol_is_rejected() -> None:
    bars = pd.concat([_bars().head(1), _bars().head(1)], ignore_index=True)
    with pytest.raises(CanonicalBarError, match="Duplicate"):
        validate_confirmed_daily_bars(bars)


def test_data_home_comes_from_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path))
    assert external_data_home() == tmp_path.resolve()
