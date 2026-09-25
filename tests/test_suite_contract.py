from __future__ import annotations

import importlib.util
import re
import tomllib
from pathlib import Path

import pandas as pd
import pytest
from quant_marketdata import DataContractError


ROOT = Path(__file__).resolve().parents[1]
PROJECT_PACKAGES = {
    "quant-research-platform": "quant_system",
    "equity-pairs-research": "equity_pairs",
    "index-rebalance-event-study": "index_rebalance_event_study",
    "llm-factor-mining": "llm_factor_mining",
    "marketdata-agent": "marketdata_agent",
}
LLM_RESEARCH_PROJECTS = ("llm-factor-mining", "marketdata-agent")


def _load_script(name: str):
    module_path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py").replace("-", "_"), module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _pyproject(project: str) -> dict:
    with (ROOT / "projects" / project / "pyproject.toml").open("rb") as handle:
        return tomllib.load(handle)


def _requirement_names(requirements: list[str]) -> set[str]:
    return {re.split(r"[\s\[<>=!~;]", item, maxsplit=1)[0].lower() for item in requirements}


def test_expected_components_are_present() -> None:
    expected = [ROOT / "packages" / "quant-marketdata" / "src" / "quant_marketdata"]
    expected += [
        ROOT / "projects" / project / "src" / package
        for project, package in PROJECT_PACKAGES.items()
    ]
    missing = [path for path in expected if not path.is_dir()]
    assert missing == []


def test_every_project_directory_is_a_registered_component() -> None:
    on_disk = {path.parent.name for path in (ROOT / "projects").glob("*/pyproject.toml")}
    assert on_disk == set(PROJECT_PACKAGES)


def test_projects_declare_the_shared_marketdata_package() -> None:
    for project in PROJECT_PACKAGES:
        metadata = _pyproject(project)["project"]
        assert "quant-marketdata" in _requirement_names(metadata["dependencies"]), project
        text = (ROOT / "projects" / project / "pyproject.toml").read_text(encoding="utf-8").lower()
        assert "yfinance" not in text, project


def test_suite_contract_lists_every_project_on_confirmed_prices() -> None:
    text = (ROOT / "configs" / "suite.yml").read_text(encoding="utf-8")
    entries = re.findall(
        r"^  - id: (\S+)\n    role: (\S+)\n    price_input: (\S+)$", text, flags=re.MULTILINE
    )
    assert {project for project, _, _ in entries} == set(PROJECT_PACKAGES)
    assert all(price_input == "confirmed" for _, _, price_input in entries)
    roles = {project: role for project, role, _ in entries}
    assert roles["llm-factor-mining"] == "llm_guided_factor_discovery_research"
    assert roles["marketdata-agent"] == "governed_llm_copilot_research"


def test_install_test_and_ci_scripts_cover_every_component() -> None:
    bootstrap = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    runner = (ROOT / "scripts" / "test-all.sh").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    components = ["packages/quant-marketdata"] + [f"projects/{name}" for name in PROJECT_PACKAGES]
    for component in components:
        assert f'-e "{component}[dev]"' in bootstrap, component
        assert f"$SUITE_ROOT/{component}/src" in runner, component
        assert f"-p no:cacheprovider {component}/tests" in runner, component
        assert f"{component}/pyproject.toml" in workflow, component
    install_lines = [line for line in bootstrap.splitlines() if not line.lstrip().startswith("#")]
    assert not any("llm]" in line for line in install_lines), "the LLM SDK extra must stay optional"


def test_llm_sdk_is_an_optional_extra_of_the_llm_projects() -> None:
    for project in LLM_RESEARCH_PROJECTS:
        metadata = _pyproject(project)["project"]
        assert "anthropic" not in _requirement_names(metadata["dependencies"]), project
        assert "anthropic" in _requirement_names(metadata["optional-dependencies"]["llm"]), project
        assert metadata["requires-python"] == ">=3.11", project


def test_citation_metadata_names_the_repository() -> None:
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^cff-version: 1\.2\.0$", text, flags=re.MULTILINE)
    assert "repository-code: \"https://github.com/SiyanChen929/marketdata-quant-suite\"" in text
    assert "doi:" not in text.lower()


def test_documentation_links_resolve() -> None:
    documents = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]
    broken = []
    for document in documents:
        text = document.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"^(?:https?:|mailto:|#)", target):
                continue
            path = (document.parent / target.split("#", 1)[0]).resolve()
            if not path.exists():
                broken.append(f"{document.relative_to(ROOT)} -> {target}")
    assert broken == []


def test_readme_result_excerpts_are_verbatim_copies() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    blocks = re.findall(
        r"<!-- verbatim: (\S+) -->\n(.*?)<!-- /verbatim -->", readme, flags=re.DOTALL
    )
    assert len(blocks) >= 2
    for source, block in blocks:
        committed = (ROOT / source).read_text(encoding="utf-8").splitlines()
        rows = [line for line in block.splitlines() if line.strip()]
        assert rows, source
        missing = [line for line in rows if line not in committed]
        assert missing == [], source


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
