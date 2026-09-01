from __future__ import annotations

import json
import pandas as pd
import pytest
from quant_marketdata import DataContractError, MarketDataStore

from quant_system.config import AppConfig, UniverseConfig
from quant_system.data.audit import audit_configured_daily_prices, audit_intraday_summary, run_data_audit
from quant_system.experiments.registry import compare_runs, record_run
from quant_system.options.chain import CSVOptionChainProvider, MarketDataOptionChainProvider, validate_overlay_against_chain


def test_data_audit_writes_reports(tmp_path):
    prices = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA", "AAA", "BBB"],
            "date": ["2024-01-02", "2024-01-03", "2024-01-03"],
            "open": [10.0, 10.5, 20.0],
            "high": [11.0, 11.0, 21.0],
            "low": [9.8, 10.2, 19.5],
            "close": [10.6, 10.9, 20.5],
            "adj_close": [10.6, 10.9, 20.5],
            "volume": [1000, 1200, 2000],
        }
    ).to_csv(prices, index=False)
    fundamentals = tmp_path / "fundamentals.csv"
    pd.DataFrame({"symbol": ["AAA"], "known_date": ["2024-01-03"], "eps_yoy_growth": [0.2]}).to_csv(fundamentals, index=False)
    events = tmp_path / "events.csv"
    pd.DataFrame({"symbol": ["AAA"], "date": ["2024-01-03"], "event_type": ["earnings"]}).to_csv(events, index=False)
    config = AppConfig(
        data_path=str(prices),
        fundamentals_path=str(fundamentals),
        events_path=str(events),
        universe=UniverseConfig(custom_symbols=("AAA", "BBB")),
    )

    reports = run_data_audit(config, tmp_path / "audit")

    assert (tmp_path / "audit" / "data_audit.html").exists()
    assert reports["daily_price_audit"].shape[0] == 2
    assert "fundamental_coverage_audit" in reports


def test_run_registry_records_and_compares_runs(tmp_path):
    config = AppConfig(output_dir=str(tmp_path / "runs"))
    run_dir = tmp_path / "runs" / "20260101_000000"
    run_dir.mkdir(parents=True)
    (run_dir / "data_provenance.json").write_text(
        json.dumps(
            {
                "data_provider": "marketdata",
                "sources": ["marketdata.app"],
                "finalities": ["confirmed"],
                "data_max_date": "2025-12-31T00:00:00",
                "data_manifest_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    record_run(run_dir, config, {"cagr": 0.2, "sharpe": 1.4, "max_drawdown": -0.1, "total_return": 0.5}, fast=True)
    table = compare_runs(tmp_path / "runs", limit=5)

    assert table.loc[0, "run_id"] == "20260101_000000"
    assert table.loc[0, "fast"] in {True, "True"}
    assert table.loc[0, "sharpe"] == 1.4
    assert table.loc[0, "data_source"] == "marketdata.app"
    assert table.loc[0, "data_finality"] == "confirmed"
    assert table.loc[0, "data_manifest_sha256"] == "a" * 64


def test_option_chain_provider_validates_overlay(tmp_path):
    chain_path = tmp_path / "chain.csv"
    pd.DataFrame(
        {
            "symbol": ["AAA"],
            "quote_date": ["2024-01-03"],
            "expiration": ["2024-02-16"],
            "option_type": ["call"],
            "strike": [105.0],
            "bid": [4.9],
            "ask": [5.1],
            "volume": [50],
            "open_interest": [500],
            "implied_volatility": [0.45],
            "delta": [0.52],
        }
    ).to_csv(chain_path, index=False)
    provider = CSVOptionChainProvider(chain_path)
    chain = provider.get_chain("AAA", "2024-01-03")
    overlay = pd.DataFrame(
        {
            "symbol": ["AAA"],
            "signal_date": ["2024-01-03"],
            "target_dte_min": [20],
            "target_dte_max": [60],
            "target_delta": [0.55],
            "executable": [False],
        }
    )

    checked = validate_overlay_against_chain(overlay, chain, min_open_interest=100, min_volume=10, max_spread_pct_mid=0.10)

    assert checked.loc[0, "executable"] is True or bool(checked.loc[0, "executable"])
    assert checked.loc[0, "chain_validation_status"] == "validated_liquid_contract"
    assert checked.loc[0, "contract_strike"] == 105.0


def test_marketdata_option_chain_provider_normalizes_runtime_chain(monkeypatch, tmp_path):
    payload = pd.DataFrame(
        {
            "option_symbol": ["AAA240216C00105000"],
            "underlying": ["AAA"],
            "query_date": ["2024-01-03"],
            "expiration": ["2024-02-16"],
            "side": ["call"],
            "strike": [105.0],
            "bid": [4.9],
            "ask": [5.1],
            "mid": [5.0],
            "volume": [50],
            "open_interest": [500],
            "iv": [0.45],
            "delta": [0.52],
        }
    )

    def fake_fetch(symbol, **kwargs):
        assert symbol == "AAA"
        assert kwargs["client"] is not None
        assert kwargs["date"] == "2024-01-03"
        assert kwargs["min_open_interest"] == 100
        assert kwargs["min_volume"] == 10
        return payload

    monkeypatch.setenv("MARKETDATA_TOKEN", "TOKEN")
    monkeypatch.setattr("quant_system.options.chain._fetch_marketdata_option_chain", fake_fetch)
    provider = MarketDataOptionChainProvider(cache_dir=tmp_path, min_open_interest=100, min_volume=10)

    chain = provider.get_chain("AAA", "2024-01-03")

    assert chain.loc[0, "symbol"] == "AAA"
    assert chain.loc[0, "option_type"] == "call"
    assert chain.loc[0, "option_symbol"] == "AAA240216C00105000"
    assert chain.loc[0, "implied_volatility"] == 0.45


def test_marketdata_option_chain_rejects_constructor_credentials(tmp_path):
    with pytest.raises(ValueError, match="MARKETDATA_TOKEN"):
        MarketDataOptionChainProvider(api_key="TOKEN", cache_dir=tmp_path)


def test_historical_option_chain_never_falls_back_to_latest(monkeypatch, tmp_path):
    calls = []

    def fake_fetch(symbol, **kwargs):
        calls.append(kwargs)
        return pd.DataFrame()

    monkeypatch.setattr("quant_system.options.chain._fetch_marketdata_option_chain", fake_fetch)
    provider = MarketDataOptionChainProvider(cache_dir=tmp_path)

    chain = provider.get_chain("AAA", "2024-01-03")

    assert chain.empty
    assert len(calls) == 1
    assert calls[0]["date"] == "2024-01-03"


def test_data_audit_includes_marketdata_runtime_reports(monkeypatch, tmp_path):
    shared_data = tmp_path / "shared-marketdata"
    monkeypatch.setenv("QUANT_DATA_HOME", str(shared_data))
    MarketDataStore().write_bars(
        pd.DataFrame(
            {
                "symbol": ["AAA"],
                "date": ["2024-01-03"],
                "open": [10.0],
                "high": [11.0],
                "low": [9.8],
                "close": [10.6],
                "volume": [1000],
                "source": ["marketdata.app"],
                "finality": ["confirmed"],
            }
        ),
        finality="confirmed",
    )
    legacy_path = tmp_path / "legacy-price-path-must-not-be-read"
    legacy_path.mkdir()

    class FakeEarningsClient:
        last_errors = {}

        def get_bulk_earnings(self, symbols, start, end):
            return pd.DataFrame(
                {
                    "symbol": ["AAA", "AAA", "AAA", "AAA", "AAA"],
                    "report_date": ["2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15", "2024-01-15"],
                    "fiscal_period_end": ["2022-12-31", "2023-03-31", "2023-06-30", "2023-09-30", "2023-12-31"],
                    "reported_eps": [1.0, 1.1, 1.2, 1.3, 1.6],
                    "estimated_eps": [0.9, 1.0, 1.1, 1.2, 1.4],
                    "surprise_eps_pct": [0.1, 0.1, 0.1, 0.1, 0.14],
                }
            )

    monkeypatch.setattr("quant_system.data.audit.MarketDataEarningsClient", FakeEarningsClient)
    config = AppConfig(
        data_provider="marketdata",
        data_path=str(legacy_path),
        universe=UniverseConfig(custom_symbols=("AAA",)),
    )

    reports = run_data_audit(config, tmp_path / "audit")

    assert reports["daily_price_audit"].loc[0, "status"] == "ok"
    assert reports["daily_price_audit"].loc[0, "latest_date"] == pd.Timestamp("2024-01-03")
    assert reports["marketdata_runtime_earnings_audit"].loc[0, "status"] == "ok"
    assert reports["marketdata_runtime_fundamentals_audit"].loc[0, "status"] == "ok"
    assert reports["marketdata_runtime_fundamentals_audit"].loc[0, "eps_yoy_growth_count"] >= 1


@pytest.mark.parametrize(
    ("column", "value"),
    [("source", "legacy.csv"), ("finality", "provisional")],
)
def test_marketdata_daily_audit_rejects_wrong_provenance(monkeypatch, column, value):
    bars = pd.DataFrame(
        {
            "date": ["2024-01-03"],
            "symbol": ["AAA"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.8],
            "close": [10.6],
            "volume": [1000],
            "source": ["marketdata.app"],
            "finality": ["confirmed"],
        }
    )
    bars[column] = value

    class FakeStore:
        def read_bars(self, **kwargs):
            assert kwargs["finality"] == "confirmed"
            assert kwargs["resolution"] == "D"
            return bars

    monkeypatch.setattr("quant_system.data.audit.MarketDataStore", FakeStore)

    with pytest.raises(DataContractError):
        audit_configured_daily_prices(
            AppConfig(data_provider="marketdata"),
            ["AAA"],
        )


def test_marketdata_intraday_audit_reads_shared_provisional_resolution(monkeypatch, tmp_path):
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path))
    MarketDataStore().write_bars(
        pd.DataFrame(
            {
                "date": ["2024-01-03 09:35:00"],
                "symbol": ["AAA"],
                "open": [10.0],
                "high": [10.2],
                "low": [9.9],
                "close": [10.1],
                "volume": [1000],
                "source": ["marketdata.app"],
                "finality": ["provisional"],
            }
        ),
        finality="provisional",
        resolution="5",
    )

    audit = audit_intraday_summary(
        AppConfig(data_provider="marketdata", start_date="2024-01-01"),
        ["AAA"],
    )

    assert audit.loc[0, "status"] == "ok_shared_provisional_bars"
    assert audit.loc[0, "row_count"] == 1
