"""Documentation integrity: the paper and project docs quote the committed results exactly.

These tests fail when ``results/synthetic_benchmark/summary.json`` changes
without regenerating ``paper/tables/*.tex`` (``scripts/render_paper_tables.py``)
or without updating the tables quoted verbatim in ``README.md`` and
``docs/result-card.md``.  They also check that the manuscript's citations,
cross-references, inputs and generated macros resolve.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys
from types import ModuleType

import pytest


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results" / "synthetic_benchmark"
HEADLINE = "## Headline: planted markets (mean ± sd across complete runs)"
RECOVERY = "## Recovery of each planted signal (complete runs)"
NULL = "## Null markets (snr = 0): false selections"
NULL_RECALL = "## Null markets (snr = 0): recall of the planted expressions"
PAPER = PROJECT / "paper"
# entries of the approved reference list that are flagged for metadata verification
VERIFY_KEYS = frozenset({"bailey2017pbo", "glasserman2023lookahead", "wang2023alphagpt", "yu2023alphagen"})


def _load_renderer() -> ModuleType:
    path = PROJECT / "scripts" / "render_paper_tables.py"
    spec = importlib.util.spec_from_file_location("render_paper_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def renderer() -> ModuleType:
    return _load_renderer()


def _markdown_table(text: str, heading: str) -> list[str]:
    """Header, separator and body rows of the first table after ``heading``."""

    lines = text.splitlines()
    start = lines.index(heading)
    rows = []
    for line in lines[start + 1 :]:
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break
    assert rows, f"no table after {heading!r}"
    return rows


def _manuscript() -> str:
    def expand(path: Path) -> str:
        text = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in path.read_text(encoding="utf-8").splitlines())

        def include(match: re.Match[str]) -> str:
            target = PAPER / f"{match.group(1)}.tex"
            assert target.is_file(), f"missing \\input target {target}"
            return expand(target)

        return re.sub(r"\\input\{([^}]+)\}", include, text)

    return expand(PAPER / "main.tex")


def test_committed_paper_tables_are_current(summary, renderer) -> None:
    assert renderer.stale_tables(summary, PAPER / "tables") == []


def _latex_body(table: str) -> list[list[str]]:
    body = table.split("\\midrule\n", 1)[1].split("\\bottomrule", 1)[0]
    rows = [row.rstrip(" \\\n") for row in body.strip().split("\\\\\n") if row.strip()]
    return [[cell.strip().replace("$\\pm$", "±") for cell in row.split("&")] for row in rows]


def _markdown_body(markdown: str, heading: str) -> list[list[str]]:
    return [[cell.strip() for cell in row.strip("|").split("|")] for row in _markdown_table(markdown, heading)[2:]]


def test_latex_cells_agree_with_the_markdown_summary(summary, renderer) -> None:
    markdown = (RESULTS / "summary.md").read_text(encoding="utf-8")
    headline = _markdown_body(markdown, HEADLINE)
    assert len(headline) == len(summary["aggregate"])
    assert _latex_body(renderer.headline_table(summary)) == headline
    null = _markdown_body(markdown, NULL)
    assert len(null) == len(summary["null_aggregate"])
    assert _latex_body(renderer.null_table(summary)) == null


def test_docs_quote_the_committed_tables_verbatim() -> None:
    markdown = (RESULTS / "summary.md").read_text(encoding="utf-8")
    headline = _markdown_table(markdown, HEADLINE)
    recovery = _markdown_table(markdown, RECOVERY)
    null = _markdown_table(markdown, NULL)
    null_recall = _markdown_table(markdown, NULL_RECALL)
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    card = (PROJECT / "docs" / "result-card.md").read_text(encoding="utf-8")
    for table in (headline, recovery, null, null_recall):
        assert "\n".join(table) in readme
        assert "\n".join(table) in card
    for text in (readme, card, markdown):
        assert "SYNTHETIC DATA - NOT EVIDENCE ABOUT REAL MARKETS" in text


def test_manuscript_references_resolve() -> None:
    text = _manuscript()
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    entries = dict(re.findall(r"@\w+\{([^,]+),(.*?)\n\}", bib, flags=re.S))
    cited = {
        key.strip()
        for group in re.findall(r"\\cite[tp]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", text)
        for key in group.split(",")
    }
    assert cited and cited <= set(entries)
    assert set(entries) <= cited, "every bibliography entry should be cited"
    for key in VERIFY_KEYS & set(entries):
        assert "verify metadata before submission" in entries[key]
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    assert set(re.findall(r"\\ref\{([^}]+)\}", text)) <= labels


def test_manuscript_macros_resolve() -> None:
    text = _manuscript()
    macros = (PAPER / "tables" / "synthetic_macros.tex").read_text(encoding="utf-8")
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", macros))
    used = set(re.findall(r"\\(Syn\w+)", text))
    assert used and used <= defined
    keys = set(re.findall(r"\\csname syn@(?:mean|msd)@([^\\]+)\\endcsname", macros))
    lookups = re.findall(r"\\SynMean(?:Sd)?\{([^}]*)\}\{([^}]*)\}\{([^}]*)\}", text)
    assert lookups
    assert {"@".join(parts) for parts in lookups} <= keys
    null_keys = set(re.findall(r"\\csname syn@null(?:mean|msd)@([^\\]+)\\endcsname", macros))
    null_lookups = re.findall(r"\\SynNullMean(?:Sd)?\{([^}]*)\}\{([^}]*)\}", text)
    assert {"@".join(parts) for parts in null_lookups} <= null_keys
    selecting = set(re.findall(r"\\csname syn@nullsel@([^\\]+)\\endcsname", macros))
    assert set(re.findall(r"\\SynNullSelecting\{([^}]*)\}", text)) <= selecting
    # pending results stay visibly pending
    assert text.count("\\todo{") >= 5


def test_markdown_summary_is_the_rendering_of_the_json_summary(summary) -> None:
    from llm_factor_mining.benchmark.report import render_markdown

    markdown = (RESULTS / "summary.md").read_text(encoding="utf-8")
    assert markdown == render_markdown(summary, command=summary["command"])
