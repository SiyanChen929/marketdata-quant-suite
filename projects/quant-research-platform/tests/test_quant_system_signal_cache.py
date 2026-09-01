from __future__ import annotations

import pickle

import pandas as pd

from quant_system import research
from quant_system.config import AppConfig
from quant_system.research import (
    _research_bundle_cache_path,
    _signal_cache_path,
    _signal_code_fingerprint,
    prepare_research_bundle,
)
from quant_system.strategies.base import StrategyContext


def test_signal_cache_key_includes_strategy_code_fingerprint() -> None:
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02", "2026-01-02"]),
            "symbol": ["AAA", "BBB"],
        }
    )
    path = _signal_cache_path(AppConfig(), features)
    fingerprint = _signal_code_fingerprint()

    assert path.parent.name == "signals"
    assert fingerprint["sha256"]
    assert any(str(row["path"]).endswith("momentum_longs_short.py") for row in fingerprint["files"])


def test_marketdata_cache_keys_follow_authoritative_manifest(tmp_path, monkeypatch) -> None:
    manifest = tmp_path / "manifests" / "marketdata" / "confirmed.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"updated_at":"one"}', encoding="utf-8")
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path))
    config = AppConfig(data_provider="marketdata", data_path=str(tmp_path / "lake" / "confirmed"))
    features = pd.DataFrame({"date": pd.to_datetime(["2026-01-02"]), "symbol": ["AAA"]})

    first_signal = _signal_cache_path(config, features)
    first_bundle = _research_bundle_cache_path(config)
    manifest.write_text('{"updated_at":"a-longer-second-value"}', encoding="utf-8")

    assert _signal_cache_path(config, features) != first_signal
    assert _research_bundle_cache_path(config) != first_bundle


def _stub_bundle(monkeypatch) -> tuple[pd.DataFrame, pd.DataFrame, StrategyContext]:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-02"]),
            "symbol": ["AAA"],
            "close": [100.0],
            "adj_close": [100.0],
            "volume": [1_000.0],
            "source": ["marketdata.app"],
            "finality": ["confirmed"],
        }
    )
    candidate = prices.assign(tradable=True)
    features = prices.assign(feature_score=1.0)
    context = StrategyContext(benchmark="SPY")
    monkeypatch.setattr(research, "prepare_research_data", lambda _config: (prices, [], candidate))
    monkeypatch.setattr(research, "build_features_and_context", lambda *_args: (features, context))
    return prices, features, context


def test_marketdata_bundle_never_persists_vendor_prices(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path / "marketdata"))
    prices, _, _ = _stub_bundle(monkeypatch)
    config = AppConfig(data_provider="marketdata")

    returned_prices, warnings, *_ = prepare_research_bundle(config)

    assert returned_prices.equals(prices)
    assert not _research_bundle_cache_path(config).exists()
    assert any("disabled for MarketData" in warning for warning in warnings)


def test_csv_bundle_cache_contains_derived_data_only(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    _, features, _ = _stub_bundle(monkeypatch)
    config = AppConfig(data_provider="csv")

    prepare_research_bundle(config)
    cache_path = _research_bundle_cache_path(config)

    with cache_path.open("rb") as handle:
        cached = pickle.load(handle)
    assert "prices" not in cached

    monkeypatch.setattr(research, "prepare_research_data", lambda _config: (_ for _ in ()).throw(AssertionError("cache miss")))
    returned_prices, warnings, *_ = prepare_research_bundle(config)
    assert returned_prices[["date", "symbol", "close"]].equals(features[["date", "symbol", "close"]])
    assert any("Reused research bundle cache" in warning for warning in warnings)
