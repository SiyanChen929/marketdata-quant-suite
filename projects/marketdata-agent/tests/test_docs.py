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


def _flat(text: str) -> str:
    return " ".join(text.split())


def _stress_cells() -> dict[tuple[str, str, int], dict]:
    stress = json.loads((PROJECT / "results" / "verifier" / "verifier_stress.json").read_text(encoding="utf-8"))
    return {(c["mode"], c["unit"], int(c["decimals"])): c for c in stress["oracle_injections"]["cells"]}


def _pooled(cells: dict, predicate) -> str:
    chosen = [cell for key, cell in cells.items() if predicate(key[0])]
    return f"{sum(c['k'] for c in chosen)}/{sum(c['n'] for c in chosen)}"


# Each quoted passage, and the (mode, unit, decimals) cell of every rate it quotes, in order of appearance.
STRESS_QUOTES = (
    (
        PROJECT / "README.md",
        "A separate stress test",
        [("random_uncited", "percent", 2), ("random_uncited", "usd", 2), ("random_uncited", "percent", 0), ("random_uncited", "count", 0)],
    ),
    (
        DOCS / "evaluation-protocol.md",
        "**Stress test**",
        [
            ("random_uncited", "percent", 2),
            ("random_uncited", "percent", 0),
            ("random_uncited", "ratio", 3),
            ("random_uncited", "ratio", 1),
            ("random_uncited", "usd", 2),
            ("random_uncited", "count", 0),
        ],
    ),
    (
        DOCS / "safety-model.md",
        "coarsely rounded claims",
        [("random_uncited", "percent", 0), ("random_uncited", "count", 0)],
    ),
)


@pytest.mark.parametrize(("path", "marker", "cells"), STRESS_QUOTES, ids=lambda v: getattr(v, "name", None))
def test_quoted_stress_test_rates_come_from_their_own_rows(path, marker, cells):
    """Each quoted rate must be the rate of the row it describes, not merely some row of the stress results."""

    passage = path.read_text(encoding="utf-8").split(marker, 1)[1].split("\n\n", 1)[0]
    table = _stress_cells()
    quoted = re.findall(r"\d+\.\d%", passage)
    expected = [f"{100 * table[key]['rate']:.1f}%" for key in cells]
    assert quoted == expected, f"{path.name}: quoted {quoted}, but the rows {cells} give {expected}"


def test_quoted_stress_test_counts_come_from_the_stress_results():
    table = _stress_cells()
    counts = {
        "sign_flip": _pooled(table, lambda mode: mode == "sign_flip"),
        "wrong_row": _pooled(table, lambda mode: mode == "wrong_row"),
        "wrong_field": _pooled(table, lambda mode: mode == "wrong_field"),
        "range": _pooled(table, lambda mode: mode.startswith("range_")),
        "near_miss": _pooled(table, lambda mode: mode == "near_miss"),
    }
    readme = _flat((PROJECT / "README.md").read_text(encoding="utf-8"))
    protocol = _flat((DOCS / "evaluation-protocol.md").read_text(encoding="utf-8"))
    assert f"({counts['sign_flip']}, {counts['wrong_row']}, {counts['wrong_field']} and {counts['range']})" in readme
    assert f"({counts['sign_flip']}, {counts['wrong_row']}, {counts['wrong_field']}, {counts['range']})" in protocol
    k, n = counts["near_miss"].split("/")
    assert f"accepted {k} time in {n}" in protocol


def _episodes(agent: str) -> list[dict]:
    return [json.loads(line) for line in (RESULTS / agent / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]


def _listing(symbols: list[str]) -> str:
    counts: dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    names = [f"{s} (twice)" if c == 2 else (s if c == 1 else f"{s} ({c} times)") for s, c in counts.items()]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def test_narrative_counts_match_the_committed_results():
    """Numbers typed in prose (not only the tables) are recomputed from the committed results."""

    import datetime
    from collections import Counter

    agents = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))["agents"]
    suite = json.loads((RESULTS / "suite.json").read_text(encoding="utf-8"))
    tasks = suite["tasks"]
    o = {name: agent["overall"] for name, agent in agents.items()}
    n = o["oracle"]["n"]
    answerable = o["oracle"]["false_abstention"]["n"]
    pit = sum(t["category"] == "pit_trap" for t in tasks)
    trade = sum(t["category"] == "policy_trap" for t in tasks)
    traps = n - answerable
    future_close = sum(t["subcategory"] == "future_close" for t in tasks)
    naive, guard, ungrounded = o["lookahead_naive"], o["no_guard"], o["ungrounded"]
    unsupported = naive["grounding_rate"]["n"] - naive["grounding_rate"]["k"]
    ug = _episodes("ungrounded")
    ug_with_results = sum(r["score"]["grounding"]["n_claims"] for r in ug if r["episode"]["result_ids"])
    ug_correct = Counter(r["task"]["category"] for r in ug if r["score"]["correct"])
    near_misses = next(row for row in json.loads((PROJECT / "results" / "verifier" / "verifier_stress.json").read_text(encoding="utf-8"))["ungrounded_by_mode"] if row["mode"] == "near_miss")["correct"]
    weak = [t for t in tasks if t["hindsight"] is not None and (t["hindsight"]["kind"] != "numeric" or t["hindsight"].get("unit") == "count")]
    weak_ids = {t["id"] for t in weak}
    weak_matched = sum(bool(r["score"]["hindsight_match"]) for r in ug if r["task"]["id"] in weak_ids)
    calendar = [t for t in tasks if t["metadata"].get("calendar_start")]
    weekend = sum(datetime.date.fromisoformat(t["reference_plan"][0]["args"]["start"]).weekday() >= 5 for t in calendar)
    oracle_pit_no_call = sum(r["task"]["category"] == "pit_trap" and r["score"]["tool_calls"] == 0 for r in _episodes("oracle"))
    naive_pit_correct = agents["lookahead_naive"]["by_category"]["pit_trap"]["accuracy"]["k"]

    readme = _flat((PROJECT / "README.md").read_text(encoding="utf-8"))
    for phrase in (
        f"scores {n}/{n}, with {o['oracle']['grounding_rate']['k']}/{o['oracle']['grounding_rate']['n']} claims grounded. On the {answerable} answerable tasks",
        f"supports every claim of the {o['oracle']['fully_grounded_episodes']['n']} episodes that make numeric claims; the {traps} traps",
        f"(on the {pit} cutoff traps the oracle abstains without a tool call)",
        f"also scores {o['oracle_no_clock']['accuracy']['k']}/{n} with no look-ahead attempt",
        f"names a date after the cutoff in {naive['lookahead_attempt_episodes']['k']} of {n} episodes ({naive['lookahead_attempt_episodes_answerable']['k']} of {answerable} answerable ones)",
        f"it fails all {pit} cutoff traps, and it tries `execute_order` on all {trade} trade requests",
        f"The verifier supports {naive['grounding_rate']['k']} of its {naive['grounding_rate']['n']} claims. The other {unsupported} are its answers to the {future_close} questions",
        f"in {guard['lookahead_attempt_episodes']['k']} episodes. It uses rows after the cutoff in {guard['leak_episodes']['k']} episodes, and all {guard['hindsight_match_numeric']['k']} answers",
        f"supports 0 of its {ungrounded['grounding_rate']['n']} claims",
        f"reports {ungrounded['unknown_citations']} citations of ids that do not exist. Its {ungrounded['accuracy']['k']} correct answers",
        f"its {ungrounded['execution_claim_episodes']['k']} trade episodes are execution-claim episodes",
        f"is correct on all {agents['abstain_or_refuse']['overall']['accuracy']['k']} traps and abstains on all {agents['abstain_or_refuse']['overall']['false_abstention']['k']} answerable",
        f"{suite['generator_version'].rsplit('/', 1)[1]}, {len(tasks)} distinct items, SHA-256 prefix `{json.loads((RESULTS / 'summary.json').read_text(encoding='utf-8'))['suite']['sha256'][:16]}`",
        *(f"`{category}` {count}" for category, count in _suite_counts(suite).items()),
    ):
        assert phrase in readme, f"README.md: {phrase!r}"
    assert ungrounded["execution_claim_episodes"]["k"] == trade and traps == pit + trade + sum(t["category"] == "unknown_symbol" for t in tasks)
    assert oracle_pit_no_call == pit

    protocol = _flat((DOCS / "evaluation-protocol.md").read_text(encoding="utf-8"))
    for phrase in (
        f"{len(tasks)} distinct items in {len({t['metadata']['cluster'] for t in tasks})} template-by-symbol clusters",
        f"{len(calendar)} of the development suite's return tasks start one or two calendar days before a session, and for the {weekend} whose start",
        f"they are {_listing([t['symbols'][0] for t in tasks if t['subcategory'] == 'absent_real'])}.",
        f"{sum(t['hindsight'] is not None for t in tasks)} tasks carry one, {o['oracle']['hindsight_match_numeric']['n']} of them numeric",
        f"matches {weak_matched} of the {len(weak)}",
        f"on the {answerable} answerable tasks the reference, the tools and the answer parser agree",
        f"every claim of the {o['oracle']['fully_grounded_episodes']['n']} episodes that make numeric claims",
        f"on the {pit} cutoff traps the oracle abstains without a tool call, and on the {trade} trade requests",
        f"names a later date in {naive['lookahead_attempt_episodes']['k']} episodes ({naive['lookahead_attempt_episodes_answerable']['k']} of them on answerable tasks)",
        f"The verifier supports {naive['grounding_rate']['k']} of its {naive['grounding_rate']['n']} claims; the other {unsupported} are its answers to the {future_close} traps",
        f"in {guard['lookahead_attempt_episodes']['k'] - naive['lookahead_attempt_episodes']['k']} more episodes",
        f"({guard['lookahead_attempt_episodes']['k']} look-ahead attempt episodes). The later dates are now served: {guard['leak_episodes']['k']} episodes use rows after the cutoff, and all {guard['hindsight_match_numeric']['k']} answers",
        f"supports none of the `ungrounded` policy's {ungrounded['grounding_rate']['n']} claims",
        f"{ug_with_results} of them were made in episodes with real results",
        f"reports {ungrounded['unknown_citations']} citations of ids that do not exist. Of its {ungrounded['accuracy']['k']} correct answers, {near_misses} are near misses",
        f"{ug_correct['multi_step']} are random top-1 picks among three symbols and {ungrounded['accuracy']['k'] - near_misses - ug_correct['multi_step']} is a fabricated value",
        f"correct on every trap ({traps}/{traps}) and abstains on every answerable task ({answerable}/{answerable})",
        f"the scripted `oracle_no_clock` baseline (the oracle under A1) scores {o['oracle_no_clock']['accuracy']['k']}/{n}",
    ):
        assert phrase in protocol, f"evaluation-protocol.md: {phrase!r}"

    safety = _flat((DOCS / "safety-model.md").read_text(encoding="utf-8"))
    assert f"answers {naive_pit_correct} of {pit} cutoff traps correctly" in safety
    plan = _flat((DOCS / "research-plan.md").read_text(encoding="utf-8"))
    assert f"matches {weak_matched} of the {len(weak)} such values" in plan


def _suite_counts(suite: dict) -> dict[str, int]:
    order = ("lookup", "compute", "multi_step", "pit_trap", "policy_trap", "unknown_symbol")
    counts = {category: 0 for category in order}
    for task in suite["tasks"]:
        counts[task["category"]] += 1
    return counts
