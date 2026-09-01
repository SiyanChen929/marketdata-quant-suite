from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from quant_marketdata import DataContractError


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    module_path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py").replace("-", "_"), module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_expected_components_are_present() -> None:
    expected = [
        ROOT / "packages" / "quant-marketdata" / "src" / "quant_marketdata",
        ROOT / "projects" / "quant-research-platform" / "src" / "quant_system",
        ROOT / "projects" / "equity-pairs-research" / "src" / "equity_pairs",
        ROOT / "projects" / "index-rebalance-event-study" / "src" / "index_rebalance_event_study",
    ]
    assert all(path.is_dir() for path in expected)


def test_projects_declare_the_shared_marketdata_package() -> None:
    projects = [
        ROOT / "projects" / "quant-research-platform" / "pyproject.toml",
        ROOT / "projects" / "equity-pairs-research" / "pyproject.toml",
        ROOT / "projects" / "index-rebalance-event-study" / "pyproject.toml",
    ]
    for project in projects:
        text = project.read_text(encoding="utf-8").lower()
        assert "quant-marketdata" in text, project
        assert "yfinance" not in text, project


def test_status_script_never_returns_the_token(monkeypatch, tmp_path) -> None:
    module = _load_script("system-status.py")
    monkeypatch.setenv("MARKETDATA_TOKEN", "extremely-secret-token")
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path / "quant-data"))
    result = module.status()
    assert result["credential_configured"] is True
    assert "extremely-secret-token" not in repr(result)
    assert result["data_home_is_external"] is True


def test_public_tree_has_no_large_files() -> None:
    large = [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and path.stat().st_size > 5 * 1024 * 1024
    ]
    assert large == []


def test_legacy_importer_adds_explicit_lineage() -> None:
    module = _load_script("import-bars.py")
    raw = pd.DataFrame(
        {
            "date": ["2025-01-02"],
            "symbol": ["spy"],
            "open": [100.0],
            "high": [102.0],
            "low": [99.0],
            "close": [101.0],
            "volume": [1_000_000],
        }
    )
    with pytest.raises(DataContractError, match="attest-completed-sessions"):
        module.prepare(raw, finality="confirmed", source="marketdata.app")
    bars = module.prepare(
        raw,
        finality="confirmed",
        source="marketdata.app",
        attest_confirmed=True,
    )
    assert bars.loc[0, "symbol"] == "SPY"
    assert bars.loc[0, "source"] == "marketdata.app"
    assert bars.loc[0, "finality"] == "confirmed"


def test_download_entrypoint_normalizes_symbols(tmp_path) -> None:
    module = _load_script("download-bars.py")
    symbol_file = tmp_path / "symbols.txt"
    symbol_file.write_text("# benchmark\nspy\nMSFT\n", encoding="utf-8")
    assert module.parse_symbols(["aapl, SPY"], symbol_file) == ["AAPL", "MSFT", "SPY"]


def test_repository_audit_rejects_generated_residue(tmp_path) -> None:
    module = _load_script("repository-audit.py")
    (tmp_path / "package" / "__pycache__").mkdir(parents=True)
    (tmp_path / "package" / "__pycache__" / "module.pyc").write_bytes(b"compiled")
    (tmp_path / "orphan.pyc").write_bytes(b"compiled")
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".DS_Store").write_bytes(b"finder")

    failures = module.generated_artifact_failures(tmp_path)

    assert any("compiled Python artifact" in failure for failure in failures)
    assert any("Finder metadata artifact" in failure for failure in failures)
    assert any("package/__pycache__" in failure for failure in failures)
    assert any(".pytest_cache" in failure for failure in failures)


@pytest.mark.parametrize("endpoint_parts", [("stocks", "candles", "D", "AAPL"), ("options", "chain", "AAPL")])
def test_repository_audit_rejects_direct_marketdata_endpoints(tmp_path, endpoint_parts) -> None:
    module = _load_script("repository-audit.py")
    source = tmp_path / "projects" / "demo" / "src" / "provider.py"
    source.parent.mkdir(parents=True)
    endpoint = "https://api.marketdata.app/v1/" + "/".join(endpoint_parts) + "/"
    source.write_text(f"ENDPOINT = {endpoint!r}\n", encoding="utf-8")

    failures = module.direct_endpoint_failures(tmp_path)

    assert len(failures) == 1
    assert "outside shared gateway" in failures[0]


def test_repository_audit_allows_endpoints_inside_shared_gateway(tmp_path) -> None:
    module = _load_script("repository-audit.py")
    source = tmp_path / "packages" / "quant-marketdata" / "src" / "quant_marketdata" / "client.py"
    source.parent.mkdir(parents=True)
    endpoint = "https://api.marketdata.app/v1/" + "/".join(("stocks", "candles", "D", "AAPL")) + "/"
    source.write_text(f"ENDPOINT = {endpoint!r}\n", encoding="utf-8")

    assert module.direct_endpoint_failures(tmp_path) == []


def test_platform_marketdata_adapters_use_shared_gateway() -> None:
    module = _load_script("repository-audit.py")

    assert module.platform_adapter_failures(ROOT) == []


def test_public_code_has_no_direct_marketdata_price_endpoint() -> None:
    module = _load_script("repository-audit.py")

    assert module.direct_endpoint_failures(ROOT) == []
