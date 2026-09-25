"""The documentation, the committed results and the paper stay consistent.

These checks keep the project's claims honest without a TeX installation:

* every test the safety model cites exists;
* the README headline is a verbatim copy of the committed benchmark summary;
* the paper's tables are exactly what ``scripts/render_paper_tables.py``
  renders from ``results/benchmark/``, and every result macro the paper uses is
  defined there, so no result is typed by hand;
* the paper's language-model results section contains no numbers;
* every citation resolves, and bibliographies contain only approved entries,
  with the flagged entries marked for verification;
* LaTeX sources have balanced braces and environments, and every ``\\input``
  exists.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest


PROJECT = Path(__file__).resolve().parents[1]
DOCS = PROJECT / "docs"
PAPER = PROJECT / "paper"
RESULTS = PROJECT / "results" / "benchmark"
TEX_FILES = sorted([PAPER / "main.tex", *(PAPER / "sections").glob("*.tex")])

# The approved reference list: key -> (year, fragment of the title, lower case).
APPROVED = {
    "kakushadze2016alphas": ("2016", "101 formulaic alphas"),
    "harvey2016cross": ("2016", "cross-section of expected returns"),
    "bailey2014deflated": ("2014", "deflated sharpe ratio"),
    "bailey2017pbo": ("2017", "probability of backtest overfitting"),
    "benjamini1995fdr": ("1995", "false discovery rate"),
    "benjamini2001fdr": ("2001", "false discovery rate"),
    "white2000reality": ("2000", "reality check for data snooping"),
    "hansen2005spa": ("2005", "superior predictive ability"),
    "romano2005stepwise": ("2005", "stepwise multiple testing"),
    "neweywest1987": ("1987", "covariance matrix"),
    "koza1992genetic": ("1992", "genetic programming"),
    "gu2020empirical": ("2020", "empirical asset pricing via machine learning"),
    "romeraparedes2024funsearch": ("2024", "mathematical discoveries from program search"),
    "lopezlira2023chatgpt": ("2023", "can chatgpt forecast stock price movements"),
    "glasserman2023lookahead": ("2023", "look-ahead bias"),
    "wang2023alphagpt": ("2023", "alpha-gpt"),
    "yu2023alphagen": ("2023", "synergistic formulaic alpha collections"),
    "yao2023react": ("2023", "react"),
    "schick2023toolformer": ("2023", "toolformer"),
    "liu2024agentbench": ("2024", "agentbench"),
    "ji2023hallucination": ("2023", "survey of hallucination"),
    "chen2021finqa": ("2021", "finqa"),
    "islam2023financebench": ("2023", "financebench"),
    "wu2023bloomberggpt": ("2023", "bloomberggpt"),
}
MUST_VERIFY = {"bailey2017pbo", "glasserman2023lookahead", "wang2023alphagpt", "yu2023alphagen"}
VERIFY_NOTE = "verify metadata before submission"
GENERATED_MACRO = re.compile(r"\\((?:res|Suite|Data|Env|Run|ver|pow)[A-Z][A-Za-z]*)")


def _load_renderer() -> ModuleType:
    path = PROJECT / "scripts" / "render_paper_tables.py"
    spec = importlib.util.spec_from_file_location("render_paper_tables", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _strip_tex_comments(text: str) -> str:
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.splitlines())


def _bib_entries(path: Path) -> dict[str, dict[str, str]]:
    entries: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"^@(\w+)\{([^,\s]+),\s*\n(.*?)^\}", path.read_text(encoding="utf-8"), re.M | re.S):
        fields = {
            name.lower(): value
            for name, value in re.findall(r"^\s*(\w+)\s*=\s*\{(.*)\},?\s*$", match.group(3), re.M)
        }
        assert match.group(2) not in entries, f"duplicate bib key {match.group(2)} in {path.name}"
        entries[match.group(2)] = fields
    return entries


def _plain_title(title: str) -> str:
    return re.sub(r"\\[A-Za-z]+|[{}\\']", "", title).lower()


# Safety model --------------------------------------------------------------------------------


def test_every_test_cited_by_the_docs_exists():
    references = []
    for doc in (DOCS / "safety-model.md", DOCS / "research-plan.md", PROJECT / "README.md"):
        references += re.findall(r"(tests/test_\w+\.py)::(test_\w+)", doc.read_text(encoding="utf-8"))
    assert len(references) >= 40
    for file_name, test_name in references:
        path = PROJECT / file_name
        assert path.is_file(), f"{file_name} does not exist"
        assert re.search(rf"^def {test_name}\(", path.read_text(encoding="utf-8"), re.M), f"{file_name}::{test_name} does not exist"


def test_safety_model_covers_every_required_risk():
    text = (DOCS / "safety-model.md").read_text(encoding="utf-8").lower()
    for risk in ("look-ahead", "hallucinated numbers", "unauthorized trading", "prompt injection", "provisional data", "audit tampering"):
        assert f"| {risk}" in text, f"no controls-table row for {risk}"
        assert "residual risk" in text


# README ---------------------------------------------------------------------------------------


def test_readme_headline_is_copied_verbatim_from_the_committed_results():
    summary = (RESULTS / "summary.md").read_text(encoding="utf-8")
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    banner = next(line for line in summary.splitlines() if line.startswith("> **"))
    section = summary.split("## Headline", 1)[1].split("\n## ", 1)[0]
    table = "\n".join(line for line in section.splitlines() if line.startswith("|"))
    assert table.count("\n") >= 5
    assert banner in readme
    assert table in readme


# Paper -----------------------------------------------------------------------------------------


def test_paper_tables_are_generated_from_the_committed_results():
    rendered = _load_renderer().render_all(RESULTS)
    for name, text in rendered.items():
        assert (PAPER / "tables" / name).read_text(encoding="utf-8") == text, f"paper/tables/{name} is stale"


def test_every_result_macro_used_by_the_paper_is_generated():
    defined = set(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}", (PAPER / "tables" / "macros.tex").read_text(encoding="utf-8")))
    for path in TEX_FILES:
        used = set(GENERATED_MACRO.findall(_strip_tex_comments(path.read_text(encoding="utf-8"))))
        assert used <= defined, f"{path.name} uses undefined result macros {sorted(used - defined)}"


def test_language_model_results_are_marked_pending_and_contain_no_numbers():
    text = (PAPER / "sections" / "results.tex").read_text(encoding="utf-8")
    llm = _strip_tex_comments(text.split("\\subsection{Language-model results}", 1)[1])
    assert "\\pending{" in llm and "TODO(llm-results)" in text
    body = re.sub(r"\\(?:ref|label)\{[^}]*\}", "", llm)
    assert not re.search(r"\d", body), "the language-model results section must not contain numbers yet"


def test_every_input_exists_and_latex_sources_are_balanced():
    for path in TEX_FILES:
        text = _strip_tex_comments(path.read_text(encoding="utf-8"))
        for target in re.findall(r"\\input\{([^}]+)\}", text):
            assert (PAPER / f"{target}.tex").is_file(), f"{path.name}: \\input{{{target}}} does not exist"
        unescaped = re.sub(r"\\[{}]", "", text)
        assert unescaped.count("{") == unescaped.count("}"), f"{path.name}: unbalanced braces"
        stack: list[str] = []
        for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
            if kind == "begin":
                stack.append(env)
            else:
                assert stack and stack.pop() == env, f"{path.name}: \\end{{{env}}} does not match"
        assert not stack, f"{path.name}: unclosed environments {stack}"


@pytest.mark.parametrize("bib", [PAPER / "references.bib", DOCS / "references.bib"], ids=lambda p: p.parent.name)
def test_bibliographies_contain_only_approved_entries(bib):
    entries = _bib_entries(bib)
    assert entries
    for key, fields in entries.items():
        assert key in APPROVED, f"{bib.name}: {key} is not on the approved reference list"
        year, fragment = APPROVED[key]
        assert fields.get("year") == year, f"{key}: year {fields.get('year')} != {year}"
        assert fragment in _plain_title(fields.get("title", "")), f"{key}: unexpected title {fields.get('title')!r}"
        if key in MUST_VERIFY:
            assert fields.get("note") == VERIFY_NOTE, f"{key} must carry the note {VERIFY_NOTE!r}"


def test_every_paper_citation_resolves():
    keys = set(_bib_entries(PAPER / "references.bib"))
    cited: set[str] = set()
    for path in TEX_FILES:
        for group in re.findall(r"\\cite[pt]?\*?(?:\[[^\]]*\])*\{([^}]+)\}", _strip_tex_comments(path.read_text(encoding="utf-8"))):
            cited.update(key.strip() for key in group.split(","))
    assert len(cited) >= 15
    assert cited <= keys, f"undefined citations: {sorted(cited - keys)}"


# Quoted numbers -------------------------------------------------------------------------------


def test_the_research_plan_quotes_the_power_analysis_verbatim():
    plan = (DOCS / "research-plan.md").read_text(encoding="utf-8")
    section = plan.split("**Power**", 1)[1].split("**Negative results.**", 1)[0]
    power = (PROJECT / "results" / "power" / "power.md").read_text(encoding="utf-8")
    rows = [line for line in section.splitlines() if re.match(r"^\| H\d", line)]
    assert len(rows) >= 8
    for row in rows:
        assert row in power, f"research-plan.md quotes a power row that power.md does not contain: {row}"
    shift = json.loads((PROJECT / "results" / "power" / "power.json").read_text(encoding="utf-8"))["largest_cluster_shift"]
    assert f"({shift['shift']:.2f})" in section


def _kn(entry):
    return f"{entry['k']}/{entry['n']}"


def _pct(entry, ci=False):
    text = f"{100 * entry['rate']:.1f}%"
    if ci:
        low, high = entry["ci95"]
        text += f" [{100 * low:.1f}, {100 * high:.1f}]"
    return text


def test_the_protocol_results_table_matches_the_committed_summary():
    summary = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["agents"]
    protocol = (DOCS / "evaluation-protocol.md").read_text(encoding="utf-8")
    rows = {m.group(1): line for line in protocol.splitlines() if (m := re.match(r"^\| `(\w+)` \| \d", line))}
    assert set(rows) == set(summary)
    for name, agent in summary.items():
        o = agent["overall"]
        grounding = "n/a (no claims)" if not o["grounding_rate"]["n"] else f"{_pct(o['grounding_rate'])} ({_kn(o['grounding_rate'])})"
        expected = (
            f"| `{name}` | {_pct(o['accuracy'], ci=True)} ({_kn(o['accuracy'])}) | {grounding} | "
            f"{_kn(o['lookahead_attempt_episodes'])} | {_kn(o['leak_episodes'])} | {_kn(o['hindsight_match_numeric'])} | "
            f"{_kn(o['false_abstention'])} |"
        )
        assert rows[name] == expected


def test_quoted_stress_test_rates_come_from_the_stress_results():
    stress = (PROJECT / "results" / "verifier" / "verifier_stress.md").read_text(encoding="utf-8")
    passages = [
        (PROJECT / "README.md").read_text(encoding="utf-8").split("A separate stress test", 1)[1].split("\n\n", 1)[0],
        (DOCS / "evaluation-protocol.md").read_text(encoding="utf-8").split("**Stress test**", 1)[1].split("\n\n", 1)[0],
    ]
    for passage in passages:
        rates = re.findall(r"\d+\.\d%", passage)
        assert rates
        for value in rates:
            assert f"| {value} [" in stress, f"{value} is quoted but not in verifier_stress.md"
