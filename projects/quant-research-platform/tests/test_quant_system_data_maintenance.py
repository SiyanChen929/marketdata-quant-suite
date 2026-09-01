from __future__ import annotations

from datetime import datetime, timezone
import os

import pandas as pd
import pytest

from quant_system.data.maintenance import (
    build_cleanup_plan,
    consolidate_daily_ohlcv,
    execute_cleanup,
    storage_inventory,
)


def _ohlcv(symbol: str, date: str, close: float) -> dict[str, object]:
    return {
        "symbol": symbol,
        "date": date,
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "adj_close": close,
        "volume": 1000,
    }


def test_consolidate_daily_ohlcv_deduplicates_and_prefers_later_source(tmp_path):
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    destination = tmp_path / "processed" / "prices.parquet"
    pd.DataFrame([_ohlcv("aaa", "2024-01-02", 10.0), _ohlcv("BBB", "2024-01-02", 20.0)]).to_csv(first, index=False)
    pd.DataFrame([_ohlcv("AAA", "2024-01-02", 11.0), _ohlcv("AAA", "2024-01-03", 12.0)]).to_csv(second, index=False)

    combined, audit = consolidate_daily_ohlcv([first, second], destination)

    assert destination.exists()
    assert len(combined) == 3
    assert combined.loc[(combined["symbol"] == "AAA") & (combined["date"] == pd.Timestamp("2024-01-02")), "close"].item() == 11.0
    assert audit["status"].tolist() == ["ok", "ok"]
    assert audit["duplicates_removed"].iloc[0] == 1


def test_consolidate_daily_ohlcv_preserves_lineage(tmp_path):
    source = tmp_path / "lineaged.csv"
    destination = tmp_path / "prices.parquet"
    frame = pd.DataFrame([_ohlcv("AAA", "2024-01-02", 10.0)])
    frame["source"] = "marketdata.app"
    frame["finality"] = "confirmed"
    frame.to_csv(source, index=False)

    combined, _ = consolidate_daily_ohlcv([source], destination)

    assert combined.loc[0, "source"] == "marketdata.app"
    assert combined.loc[0, "finality"] == "confirmed"


def test_cleanup_plan_only_targets_old_rebuildable_cache_outside_retention(tmp_path):
    cache = tmp_path / "cache"
    signals = cache / "signals"
    bundles = cache / "research_bundles"
    raw = cache / "marketdata.csv"
    signals.mkdir(parents=True)
    bundles.mkdir(parents=True)
    raw.write_text("important", encoding="utf-8")
    old_time = datetime(2024, 1, 1, tzinfo=timezone.utc).timestamp()
    for name in ("one.pkl", "two.pkl", "three.pkl"):
        path = signals / name
        path.write_bytes(name.encode())
        os.utime(path, (old_time, old_time))
        old_time += 1
    bundle = bundles / "bundle.pkl"
    bundle.write_bytes(b"bundle")
    os.utime(bundle, (old_time, old_time))

    plan = build_cleanup_plan(
        cache,
        keep_signals=1,
        keep_bundles=1,
        min_age_days=1,
        now=datetime(2024, 2, 1, tzinfo=timezone.utc),
    )

    assert set(plan["category"]) == {"signal_cache"}
    assert set(plan["path"].map(lambda value: value.rsplit("/", 1)[-1])) == {"one.pkl", "two.pkl"}
    assert str(raw) not in set(plan["path"])

    with pytest.raises(ValueError, match="confirmed"):
        execute_cleanup(plan, cache)
    result = execute_cleanup(plan, cache, confirmed=True)
    assert result["status"].eq("deleted").all()
    assert raw.exists()
    assert (signals / "three.pkl").exists()
    assert bundle.exists()


def test_cleanup_rejects_path_outside_allowed_cache(tmp_path):
    cache = tmp_path / "cache"
    (cache / "signals").mkdir(parents=True)
    protected = tmp_path / "protected.pkl"
    protected.write_bytes(b"keep")
    plan = pd.DataFrame([{"path": str(protected)}])

    result = execute_cleanup(plan, cache, confirmed=True)

    assert result.loc[0, "status"] == "skipped"
    assert result.loc[0, "error"] == "outside_allowed_rebuildable_cache"
    assert protected.exists()


def test_storage_inventory_does_not_double_count_rebuildable_caches(tmp_path):
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    (data / "cache" / "signals").mkdir(parents=True)
    (data / "cache" / "research_bundles").mkdir(parents=True)
    (data / "cache" / "signals" / "a.pkl").write_bytes(b"a" * 10)
    (data / "cache" / "research_bundles" / "b.pkl").write_bytes(b"b" * 20)
    (data / "cache" / "market.csv").write_bytes(b"m" * 30)

    inventory = storage_inventory(data, runs).set_index("category")

    assert inventory.loc["signal_cache", "size_bytes"] == 10
    assert inventory.loc["research_bundle_cache", "size_bytes"] == 20
    assert inventory.loc["other_cache", "size_bytes"] == 30
    assert not bool(inventory.loc["other_cache", "rebuildable_cache"])
