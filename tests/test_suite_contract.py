from __future__ import annotations

import importlib.util
import re
import subprocess
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


VERBATIM_BLOCK = re.compile(r"<!-- verbatim: (\S+) -->\n(.*?)<!-- /verbatim -->", re.DOTALL)


def _paragraphs(lines: list[str]) -> list[tuple[str, ...]]:
    """Split lines into maximal runs of non-blank lines (a table is one run)."""

    runs: list[tuple[str, ...]] = []
    current: list[str] = []
    for line in [*lines, ""]:
        if line.strip():
            current.append(line)
        elif current:
            runs.append(tuple(current))
            current = []
    return runs


def _verbatim_excerpt_failures(readme: str, root: Path) -> list[str]:
    """Every paragraph of an excerpt must be a whole paragraph of its source, in
    source order, so that a row cannot be dropped, reordered or edited."""

    failures: list[str] = []
    for source, block in VERBATIM_BLOCK.findall(readme):
        committed = _paragraphs((root / source).read_text(encoding="utf-8").splitlines())
        excerpt = _paragraphs(block.splitlines())
        if not excerpt:
            failures.append(f"{source}: empty excerpt")
        position = -1
        for paragraph in excerpt:
            try:
                position = committed.index(paragraph, position + 1)
            except ValueError:
                failures.append(
                    f"{source}: not a complete, in-order copy of a source paragraph: {paragraph[0]!r}"
                )
    return failures


def test_readme_result_excerpts_are_verbatim_copies() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert len(VERBATIM_BLOCK.findall(readme)) >= 2
    assert _verbatim_excerpt_failures(readme, ROOT) == []


def test_verbatim_check_rejects_dropped_reordered_and_edited_rows(tmp_path) -> None:
    table = ["| arm | runs |", "|---|---|", "| a | 1 |", "| b | 2 |", "| c | 3 |"]
    banner = "> **SYNTHETIC.**"
    (tmp_path / "summary.md").write_text(
        "\n".join(["# Summary", "", banner, "", "Prose.", "", *table, "", "## Next", ""]),
        encoding="utf-8",
    )

    def readme(*paragraphs: list[str]) -> str:
        body = "\n\n".join("\n".join(paragraph) for paragraph in paragraphs)
        return f"Intro\n\n<!-- verbatim: summary.md -->\n{body}\n\n<!-- /verbatim -->\n"

    assert _verbatim_excerpt_failures(readme([banner], table), tmp_path) == []
    rejected = {
        "middle row dropped": readme([banner], table[:3] + table[4:]),
        "last row dropped": readme([banner], table[:-1]),
        "first data row dropped": readme([banner], table[:2] + table[3:]),
        "rows reordered": readme([banner], table[:2] + [table[3], table[2], table[4]]),
        "paragraphs reordered": readme(table, [banner]),
        "row edited": readme([banner], table[:-1] + ["| c | 30 |"]),
        "empty": readme([]),
    }
    for case, text in rejected.items():
        assert _verbatim_excerpt_failures(text, tmp_path), case


def test_status_script_never_returns_the_token(monkeypatch, tmp_path) -> None:
    module = _load_script("system-status.py")
    monkeypatch.setenv("MARKETDATA_TOKEN", "extremely-secret-token")
    monkeypatch.setenv("QUANT_DATA_HOME", str(tmp_path / "quant-data"))
    result = module.status()
    assert result["credential_configured"] is True
    assert "extremely-secret-token" not in repr(result)
    assert result["data_home_is_external"] is True


def test_public_tree_has_no_large_files() -> None:
    # The same file set as the audit: tracked files in a Git checkout, so that a
    # virtual environment created inside the clone (as the Quick start does) and
    # other ignored local files are not mistaken for repository content.
    module = _load_script("repository-audit.py")
    large = [path for path in module.repository_files(ROOT) if path.stat().st_size > module.MAX_BYTES]
    assert large == []


def _sparse_file(path: Path, size: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.truncate(size)
    return path


@pytest.mark.parametrize("git_checkout", [False, True], ids=["plain-tree", "git-checkout"])
def test_large_file_scan_ignores_a_local_virtualenv(tmp_path, git_checkout) -> None:
    module = _load_script("repository-audit.py")
    root = tmp_path / "suite"
    interpreter = _sparse_file(tmp_path / "python3.11", module.MAX_BYTES + 1)
    (root / ".venv" / "bin").mkdir(parents=True)
    (root / ".venv" / "bin" / "python").symlink_to(interpreter)
    _sparse_file(root / ".venv" / "lib" / "_duckdb.so", module.MAX_BYTES + 1)
    (root / "README.md").write_text("suite\n", encoding="utf-8")
    tracked_large = _sparse_file(root / "results" / "large.json", module.MAX_BYTES + 1)
    if git_checkout:
        (root / ".gitignore").write_text(".venv/\n", encoding="utf-8")
        subprocess.run(["git", "init", "-q"], cwd=root, check=True)
        subprocess.run(["git", "add", ".gitignore", "README.md", "results"], cwd=root, check=True)

    files = module.repository_files(root)
    failures, _ = module.audit_failures(root)

    assert not any(".venv" in path.relative_to(root).parts for path in files)
    assert tracked_large in files
    assert [failure for failure in failures if failure.startswith("large file")] == [
        "large file: results/large.json (5.0 MiB)"
    ]


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


# Fake credentials are assembled at run time so that this file never contains one.
_FAKE_VALUE = "a1B2c3D4" * 4
_FAKE_ANTHROPIC_KEY = "sk-" + "ant-api03-" + "Q7x_" * 24 + "AA"


@pytest.mark.parametrize(
    ("filename", "text", "label"),
    [
        ("config.sh", f"MARKETDATA_TOKEN={_FAKE_VALUE}\n", "embedded MarketData token"),
        ("settings.py", f'MARKETDATA_TOKEN = "{_FAKE_VALUE}"\n', "embedded MarketData token"),
        ("settings.py", f"MARKETDATA_API_KEY = '{_FAKE_VALUE}'\n", "embedded MarketData token"),
        ("config.yml", f"env:\n  MARKETDATA_TOKEN: {_FAKE_VALUE}\n", "embedded MarketData token"),
        ("config.json", f'{{"MARKETDATA_TOKEN": "{_FAKE_VALUE}"}}\n', "embedded MarketData token"),
        ("macro.py", f'FRED_API_KEY = "{_FAKE_VALUE}"\n', "embedded FRED key"),
        ("run.sh", f"export ANTHROPIC_API_KEY={_FAKE_VALUE}\n", "embedded Anthropic API key"),
        ("client.py", f'client = Anthropic(api_key="{_FAKE_ANTHROPIC_KEY}")\n', "Anthropic API key literal"),
        ("episodes.jsonl", f'{{"headers": {{"x-api-key": "{_FAKE_ANTHROPIC_KEY}"}}}}\n', "Anthropic API key literal"),
        ("paper.tex", f"% key {_FAKE_ANTHROPIC_KEY}\n", "Anthropic API key literal"),
        ("CITATION.cff", f"note: {_FAKE_ANTHROPIC_KEY}\n", "Anthropic API key literal"),
    ],
    ids=[
        "shell",
        "python-double-quoted",
        "python-single-quoted",
        "yaml",
        "json",
        "fred-quoted",
        "anthropic-env",
        "anthropic-literal-py",
        "anthropic-literal-jsonl",
        "anthropic-literal-tex",
        "anthropic-literal-cff",
    ],
)
def test_repository_audit_detects_embedded_credentials(tmp_path, filename, text, label) -> None:
    module = _load_script("repository-audit.py")
    (tmp_path / filename).write_text(text, encoding="utf-8")

    failures, _ = module.audit_failures(tmp_path)

    assert f"{label}: {filename}" in failures


def test_repository_audit_accepts_placeholders_and_variable_references(tmp_path) -> None:
    module = _load_script("repository-audit.py")
    (tmp_path / ".env.example").write_text(
        "MARKETDATA_TOKEN=replace_me\nFRED_API_KEY=replace_me_if_used\nANTHROPIC_API_KEY=\n",
        encoding="utf-8",
    )
    (tmp_path / "settings.py").write_text(
        'import os\nMARKETDATA_TOKEN = os.getenv("MARKETDATA_TOKEN")\n'
        'ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]\n'
        'REDACT = "sk-ant-api03-abcdefghijklmnop"  # short dummy in a redaction test\n',
        encoding="utf-8",
    )
    (tmp_path / "ci.yml").write_text(
        "env:\n  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}\n"
        '  MARKETDATA_TOKEN: "$MARKETDATA_TOKEN"\n',
        encoding="utf-8",
    )

    failures, _ = module.audit_failures(tmp_path)

    assert not [failure for failure in failures if "key" in failure or "token" in failure]


def test_env_example_lists_every_credential_without_a_value() -> None:
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    entries = dict(line.split("=", 1) for line in lines if line and not line.startswith("#"))
    assert entries["MARKETDATA_TOKEN"] == "replace_me"
    assert entries["FRED_API_KEY"].startswith("replace_me")
    # The LLM key is optional; it stays commented out so that sourcing .env
    # never exports an empty key.
    assert "ANTHROPIC_API_KEY" not in entries
    assert "# ANTHROPIC_API_KEY=" in lines


def test_ci_runs_the_llm_research_lines_with_the_sdk_installed() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    job = workflow.split("\n  llm-sdk:\n", 1)
    assert len(job) == 2, "CI needs a job that installs the optional llm extra"
    body = job[1]
    assert "import anthropic, httpx2" in body
    assert "secrets." not in body, "the SDK job must run without an API key"
    for project in LLM_RESEARCH_PROJECTS:
        assert f'-e "projects/{project}[llm,dev]"' in body, project
        assert f"-p no:cacheprovider projects/{project}/tests" in body, project


def test_install_pins_the_numeric_stack_recorded_in_the_committed_results() -> None:
    pins = dict(
        line.split("==", 1)
        for line in (ROOT / "configs" / "constraints.txt").read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    )
    summary = (
        ROOT / "projects" / "llm-factor-mining" / "results" / "synthetic_benchmark" / "summary.md"
    ).read_text(encoding="utf-8")
    software = re.search(r"Software: (.+?)\.\n", summary)
    assert software, "the committed summary records its software stack"
    recorded = dict(item.rsplit(" ", 1) for item in software.group(1).split(", "))
    assert pins == {name: recorded[name] for name in ("numpy", "pandas", "scipy")}
    bootstrap = (ROOT / "scripts" / "bootstrap.sh").read_text(encoding="utf-8")
    install_lines = [line for line in bootstrap.splitlines() if line.startswith("python -m pip install")]
    assert install_lines and all('-c "$CONSTRAINTS"' in line for line in install_lines)
    assert 'CONSTRAINTS="configs/constraints.txt"' in bootstrap
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    ci_installs = [line for line in workflow.splitlines() if "pip install" in line]
    assert ci_installs and all("-c configs/constraints.txt" in line for line in ci_installs)


_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _agenda_window(label: str) -> tuple[str, str]:
    """``"Oct 2026"``, ``"Mar – Apr 2027"`` or ``"Dec 2026 – Jan 2027"`` -> ISO months."""

    parts = [part.split() for part in label.split("–")]
    end_month, end_year = parts[-1]
    start_month, start_year = parts[0] if len(parts[0]) == 2 else (parts[0][0], end_year)
    return tuple(
        f"{year}-{_MONTHS.index(month) + 1:02d}"
        for month, year in ((start_month, start_year), (end_month, end_year))
    )


def test_agenda_milestones_match_the_factor_mining_plan_timeline() -> None:
    agenda = (ROOT / "docs" / "research-agenda.md").read_text(encoding="utf-8")
    milestones = agenda.split("## Milestones", 1)[1].split("\n## ", 1)[0]
    agenda_windows = [
        _agenda_window(row.split("|")[1].strip())
        for row in milestones.splitlines()
        if row.startswith("| ") and not row.startswith("| Window")
    ]
    plan = (ROOT / "projects" / "llm-factor-mining" / "docs" / "research-plan.md").read_text(
        encoding="utf-8"
    )
    timeline = plan.split("Timeline", 1)[1].split("\n## ", 1)[0]
    plan_windows = []
    for row in timeline.splitlines():
        cells = [cell.strip() for cell in row.split("|")[1:-1]]
        if len(cells) == 3 and re.fullmatch(r"\d{4}-\d{2}(?: to \d{4}-\d{2})?", cells[1]):
            start, _, end = cells[1].partition(" to ")
            plan_windows.append((start, end or start))
    assert plan_windows, "the plan's timeline table was not found"
    assert agenda_windows == plan_windows
