from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json

import pandas as pd
import pytest

from quant_marketdata import (
    CANONICAL_COLUMNS,
    CacheIntegrityError,
    CredentialError,
    DataContractError,
    MarketDataStore,
    wide_close,
)


def test_store_uses_external_environment_root(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path))
    assert MarketDataStore().root == tmp_path.resolve()


def test_store_requires_external_root_when_environment_missing(monkeypatch):
    monkeypatch.delenv("QUANT_DATA_HOME", raising=False)
    with pytest.raises(CredentialError, match="QUANT_DATA_HOME"):
        MarketDataStore()


def test_confirmed_store_partitions_upserts_and_filters(tmp_path, canonical_bars):
    store = MarketDataStore(tmp_path)
    written = store.write_bars(canonical_bars)
    assert tuple(written.columns) == CANONICAL_COLUMNS
    assert (tmp_path / "lake" / "confirmed" / "bars" / "resolution=D" / "year=2023" / "bars.parquet").is_file()
    assert (tmp_path / "lake" / "confirmed" / "bars" / "resolution=D" / "year=2024" / "bars.parquet").is_file()

    replacement = canonical_bars.iloc[[1]].copy()
    replacement["close"] = 102.5
    replacement["high"] = 103.5
    store.write_bars(replacement)

    selected = store.read_bars("aapl", "2024-01-01", "2024-12-31")
    assert len(selected) == 1
    assert selected.loc[0, "close"] == 102.5
    assert selected.loc[0, "symbol"] == "AAPL"
    assert len(selected.attrs["marketdata_manifest_sha256"]) == 64
    assert selected.attrs["marketdata_resolution"] == "D"
    manifest = json.loads(
        (tmp_path / "manifests" / "marketdata" / "confirmed.json").read_text()
    )
    assert manifest["finality"] == "confirmed"
    assert manifest["resolutions"] == ["D"]
    assert len(manifest["partitions"]) == 2


def test_confirmed_write_rejects_provisional_rows(tmp_path, canonical_bars):
    provisional = canonical_bars.copy()
    provisional["finality"] = "provisional"
    with pytest.raises(DataContractError, match="confirmed writes"):
        MarketDataStore(tmp_path).write_bars(provisional, finality="confirmed")


def test_provisional_and_confirmed_roots_cannot_mix(tmp_path, canonical_bars):
    store = MarketDataStore(tmp_path)
    store.write_bars(canonical_bars, finality="confirmed")
    provisional = canonical_bars.copy()
    provisional["finality"] = "provisional"
    provisional["close"] += 0.25
    provisional["high"] += 0.25
    store.write_bars(provisional, finality="provisional")

    confirmed_read = store.read_bars(finality="confirmed")
    provisional_read = store.read_bars(finality="provisional")
    assert confirmed_read["finality"].eq("confirmed").all()
    assert provisional_read["finality"].eq("provisional").all()
    assert not confirmed_read["close"].equals(provisional_read["close"])


def test_partition_checksum_detects_tampering(tmp_path, canonical_bars):
    store = MarketDataStore(tmp_path)
    store.write_bars(canonical_bars)
    path = tmp_path / "lake" / "confirmed" / "bars" / "resolution=D" / "year=2024" / "bars.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(CacheIntegrityError, match="checksum mismatch"):
        store.read_bars(start="2024-01-01", end="2024-12-31")


def test_parallel_writes_preserve_all_symbols_and_manifest(tmp_path, canonical_bars):
    store = MarketDataStore(tmp_path)
    seed = canonical_bars[canonical_bars["date"].dt.year.eq(2024)].copy()

    def write_symbol(symbol: str) -> None:
        frame = seed.copy()
        frame["symbol"] = symbol
        store.write_bars(frame)

    symbols = [f"S{index:02d}" for index in range(12)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write_symbol, symbols))

    stored = store.read_bars(start="2024-01-01", end="2024-12-31")
    assert sorted(stored["symbol"].unique()) == symbols
    assert len(stored) == len(seed) * len(symbols)


def test_resolutions_are_physically_isolated(tmp_path, canonical_bars):
    store = MarketDataStore(tmp_path)
    daily = canonical_bars.iloc[[1]].copy()
    intraday = daily.copy()
    intraday["date"] = pd.Timestamp("2024-01-02 09:35:00")
    intraday["finality"] = "provisional"

    store.write_bars(daily, finality="confirmed", resolution="D")
    store.write_bars(intraday, finality="provisional", resolution="5")

    assert len(store.read_bars(finality="confirmed", resolution="D")) == 1
    assert store.read_bars(finality="provisional", resolution="D").empty
    selected = store.read_bars(finality="provisional", resolution="5")
    assert selected.loc[0, "date"] == pd.Timestamp("2024-01-02 09:35:00")


def test_request_hash_is_deterministic_and_sensitive_to_finality(tmp_path):
    store = MarketDataStore(tmp_path)
    left = {"symbol": "AAPL", "start": "2024-01-01", "finality": "confirmed"}
    right = {"finality": "confirmed", "start": "2024-01-01", "symbol": "AAPL"}
    provisional = {**left, "finality": "provisional"}
    assert store.request_hash(left) == store.request_hash(right)
    assert store.request_hash(left) != store.request_hash(provisional)


def test_wide_close_returns_sorted_date_symbol_panel(canonical_bars):
    second = canonical_bars.copy()
    second["symbol"] = "MSFT"
    second["close"] += 10
    second["high"] += 10
    second["low"] += 10
    second["open"] += 10
    long = pd.concat([second, canonical_bars], ignore_index=True)
    panel = wide_close(long)
    assert list(panel.columns) == ["AAPL", "MSFT"]
    assert panel.index.name == "date"
    assert panel.loc[pd.Timestamp("2024-01-02"), "MSFT"] == 112.0
