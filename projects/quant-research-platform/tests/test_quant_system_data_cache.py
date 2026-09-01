from __future__ import annotations

import json

import pandas as pd
import pytest
from quant_marketdata import CredentialError

from quant_system.data.cache import CacheIntegrityError, DataCache


def _request(endpoint: str = "/stocks/earnings/AAA/") -> dict[str, object]:
    return {
        "provider": "marketdata.app",
        "dataset": "stock_earnings",
        "base_url": "https://api.marketdata.app/v1",
        "endpoint": endpoint,
        "symbol": "AAA",
        "start": "2025-01-01",
        "end": "2026-01-01",
        "Authorization": "Bearer must-never-be-persisted",
    }


def test_default_cache_requires_external_quant_data_home(monkeypatch) -> None:
    monkeypatch.delenv("QUANT_DATA_HOME", raising=False)

    with pytest.raises(CredentialError, match="QUANT_DATA_HOME"):
        DataCache()


def test_cache_roundtrip_writes_redacted_provenance_manifest(tmp_path) -> None:
    cache = DataCache(tmp_path)
    frame = pd.DataFrame({"symbol": ["AAA"], "reported_eps": [1.25]})

    cache.write("earnings", frame, request=_request())
    loaded = cache.read("earnings", request=_request())
    manifest = json.loads(cache.manifest_path_for("earnings").read_text(encoding="utf-8"))

    assert loaded is not None
    pd.testing.assert_frame_equal(loaded, frame)
    assert manifest["request"]["provider"] == "marketdata.app"
    assert "Authorization" not in manifest["request"]
    assert "must-never-be-persisted" not in cache.manifest_path_for("earnings").read_text(encoding="utf-8")
    assert manifest["rows"] == 1
    assert manifest["csv_sha256"]


def test_cache_rejects_tampered_csv(tmp_path) -> None:
    cache = DataCache(tmp_path)
    cache.write("earnings", pd.DataFrame({"symbol": ["AAA"]}), request=_request())
    cache.path_for("earnings").write_text("symbol\nBBB\n", encoding="utf-8")

    with pytest.raises(CacheIntegrityError, match="checksum mismatch"):
        cache.read("earnings", request=_request())


def test_cache_request_change_is_a_miss(tmp_path) -> None:
    cache = DataCache(tmp_path)
    cache.write("earnings", pd.DataFrame({"symbol": ["AAA"]}), request=_request())

    assert cache.read("earnings", request=_request("/stocks/earnings-v2/AAA/")) is None


def test_legacy_unmanifested_csv_is_a_miss_and_can_be_upgraded(tmp_path) -> None:
    cache = DataCache(tmp_path)
    pd.DataFrame({"symbol": ["AAA"]}).to_csv(cache.path_for("earnings"), index=False)

    assert cache.read("earnings", request=_request()) is None

    cache.write("earnings", pd.DataFrame({"symbol": ["AAA"]}), request=_request())
    assert cache.read("earnings", request=_request()) is not None


def test_cache_manifest_strips_url_credentials_and_query_parameters(tmp_path) -> None:
    cache = DataCache(tmp_path)
    request = _request("/stocks/earnings/AAA/?api_key=endpoint-secret")
    request["base_url"] = "https://user:base-secret@api.marketdata.app/v1?token=query-secret"

    cache.write("earnings", pd.DataFrame({"symbol": ["AAA"]}), request=request)
    raw_manifest = cache.manifest_path_for("earnings").read_text(encoding="utf-8")
    manifest = json.loads(raw_manifest)

    assert manifest["request"]["base_url"] == "https://api.marketdata.app/v1"
    assert manifest["request"]["endpoint"] == "/stocks/earnings/AAA/"
    assert "secret" not in raw_manifest
