"""Render and write the planted-alpha benchmark summary (JSON + Markdown)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


from ..jsonutil import write_json


HEADLINE_HEADING = "## Headline: planted markets (mean ± sd across complete runs)"
RECOVERY_HEADING = "## Recovery of each planted signal (complete runs)"
NULL_HEADING = "## Null markets (snr = 0): false selections"
NULL_RECALL_HEADING = "## Null markets (snr = 0): recall of the planted expressions"
LONG_SHORT_HEADING = "## Composite long-short on the test window (synthetic)"


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.{digits}f}"


def _mean_sd(stats: Mapping[str, Any], digits: int = 3) -> str:
    if not stats or stats.get("mean") is None:
        return "n/a"
    mean = _fmt(stats["mean"], digits)
    sd = stats.get("std")
    return mean if sd is None else f"{mean} ± {_fmt(sd, digits)}"


def _span(values: Sequence[Any], digits: int) -> str:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return "n/a"
    low, high = min(finite), max(finite)
    return _fmt(low, digits) if _fmt(low, digits) == _fmt(high, digits) else f"{_fmt(low, digits)} to {_fmt(high, digits)}"


def headline_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    """Cells of the headline table (shared with the paper-table renderer)."""

    rows = []
    for row in summary["aggregate"]:
        metrics = row["metrics"]
        rows.append(
            [
                _fmt(row["snr"], 2),
                row["arm"],
                f"{row['n_complete']}/{row['n_runs']}",
                _mean_sd(metrics["recovery_rate"], 2),
                _mean_sd(metrics["recall_rate"], 2),
                _mean_sd(metrics["n_selected"], 1),
                _mean_sd(metrics["fdp_true"], 2),
                _mean_sd(metrics["test_nonsignificant_rate"], 2),
                _mean_sd(metrics["composite_test_ic"], 4),
                _mean_sd(metrics["composite_test_icir"], 3),
                _mean_sd(metrics["behavioural_novelty_mean"], 2),
                _mean_sd(metrics["structural_novelty_mean"], 2),
            ]
        )
    return rows


def null_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    rows = []
    runs = [run for run in summary["runs"] if float(run["snr"]) == 0 and run.get("complete")]
    for row in summary.get("null_aggregate", []):
        metrics = row["metrics"]
        mine = [run for run in runs if run["arm"] == row["arm"]]
        selecting = sum(1 for run in mine if run.get("n_selected"))
        composite_t = [
            (run.get("composite_test") or {}).get("t_nw") for run in mine if run.get("n_selected")
        ]
        rows.append(
            [
                row["arm"],
                f"{row['n_complete']}/{row['n_runs']}",
                f"{selecting}/{row['n_complete']}",
                _mean_sd(metrics["n_selected"], 1),
                _mean_sd(metrics["n_screen_survivors"], 1),
                _mean_sd(metrics["n_confirmed"], 1),
                _span(composite_t, 2),
            ]
        )
    return rows


def _table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


HEADLINE_HEADER = (
    "SNR",
    "arm",
    "complete runs",
    "recovery",
    "recall",
    "selected",
    "true FDP",
    "test non-sig. rate",
    "composite test IC",
    "composite test ICIR",
    "behavioural novelty",
    "structural novelty",
)
NULL_HEADER = (
    "arm",
    "complete runs",
    "runs selecting ≥ 1 factor",
    "selected",
    "screen survivors",
    "confirmed on validation",
    "composite test t (runs that selected)",
)


def render_markdown(summary: Mapping[str, Any], *, command: str) -> str:
    """Human-readable report; every number is copied from ``summary``."""

    config = summary["config"]
    planted = summary["planted"]
    markets = summary["markets"]
    planted_markets = [market for market in markets if float(market["snr"]) > 0]
    lines = [
        "# Synthetic planted-alpha benchmark",
        "",
        f"> **{summary['banner']}.** Prices are simulated with known planted signals to test "
        "whether the search protocol can recover them, and how often it selects factors when nothing "
        "is planted. These results say nothing about the profitability of any factor in real markets.",
        "",
        "## Reproduce",
        "",
        "```bash",
        command,
        "```",
        "",
        f"Total runtime: {summary['runtime_seconds']} s. Software: "
        + ", ".join(f"{key} {value}" for key, value in summary["software"].items())
        + ".",
        "",
        "## Setup",
        "",
        f"- Markets: {config['n_symbols']} symbols x {config['n_dates']} sessions; planted SNR levels "
        f"{', '.join(_fmt(value, 2) for value in config['snr_levels'])} with market seeds "
        f"{', '.join(str(seed) for seed in config['seeds'])}; null markets (snr = 0) with seeds "
        f"{', '.join(str(seed) for seed in config['null_seeds'])}. Proposer seed = "
        f"{summary['proposer_seed_offset']} + market seed, whatever the SNR: for market seed k the "
        "feedback-free random arm evaluates the same expression stream on the planted and null markets of "
        "seed k, and genetic programming starts from the same first batch (common random numbers), so "
        "these cells are not independent samples of the proposer.",
        f"- Splits: formation / validation / test fractions {config['fractions']}, embargo "
        f"= lag + horizon sessions; execution lag {config['metric']['lag']}, horizon "
        f"{config['metric']['horizon']}.",
        f"- Budget: {config['budget']} trials per run unless the arm says `name@budget` (invalid and "
        f"degenerate proposals count; exact repeats do not), batches of {config['batch_size']}. A trial "
        f"is degenerate when fewer than max({config['min_ic_dates']}, {config['min_ic_fraction']} x "
        "scorable) formation dates have a defined IC.",
        f"- Selection: formation screen {config['fdr_method'].upper()} at {config['fdr_alpha']} over "
        "behavioural classes (m = budget minus behavioural duplicates); validation confirmation "
        f"{config['confirm_method'].upper()} at {config['confirm_alpha']} on one-sided validation "
        "p-values (m = candidates carried forward); rank by validation ICIR; greedy decorrelation at "
        f"|rho| < {config['max_abs_corr']}; top {config['top_k']}; one sealed test reveal.",
        f"- Recovery: a planted signal is recovered if a selected (oriented) factor's test-window values "
        f"have mean cross-sectional rank correlation rho >= {config['recovery_threshold']} with it. "
        "Recall (\"found\") applies |rho| to every evaluated trial on the formation window: it says that the "
        "proposer produced a close copy of a planted expression, not that search detected a signal, and it is "
        "about as high on null markets, where nothing is planted (see the null-market recall table).",
        f"- True FDP: share of selected factors whose oriented rank correlation with the planted "
        f"composite (the true expected-return signal) on the test window is below "
        f"{config['true_discovery_threshold']}; on null markets every selected factor is false.",
        f"- Test non-significance rate: share of selected factors whose one-sided test IC fails BH at "
        f"{config['test_alpha']} within the selected set (p-values against t(n - 1)). It measures test "
        "power, not falsity.",
        "- Behavioural novelty: 1 - max |mean per-date rank correlation| of a selected factor with every "
        "reference-library entry on the formation window (value-based; 0 = a re-spelled library factor). "
        "Structural novelty: 1 - max subtree Jaccard similarity with the library (syntax only; secondary).",
        "",
        "Planted signals (equal weights, each entered as `cs_zscore(expression)`); behavioural novelty "
        "against the reference library on the formation window, range over planted markets:",
        "",
    ]
    novelty_rows = []
    for item in planted:
        values = [market["planted_behavioural_novelty"][item["name"]]["novelty"] for market in planted_markets]
        nearest = sorted({market["planted_behavioural_novelty"][item["name"]]["nearest"] for market in planted_markets} - {None})
        novelty_rows.append(
            [item["name"], f"`{item['expression']}`", item["family"], _span(values, 2), ", ".join(nearest) or "n/a"]
        )
    lines += _table(("name", "expression", "family", "behavioural novelty", "nearest library entry"), novelty_rows)
    lines += ["", "Arms:", ""]
    lines += [f"- `{name}`: {state}" for name, state in summary["proposers"].items()]
    lines += ["", HEADLINE_HEADING, ""]
    lines += _table(HEADLINE_HEADER, headline_rows(summary))
    lines += [
        "",
        "Recovery, recall and selected average all complete runs. True FDP, the test non-significance "
        "rate, composite test IC/ICIR and the novelty scores describe selected factors, so they average "
        "only the runs that selected at least one factor (see the per-run funnel); a value without ± comes "
        "from a single run, and `n/a` means no run selected anything.",
        "",
        RECOVERY_HEADING,
        "",
    ]
    recovery_rows = []
    for row in summary["aggregate"]:
        n = row["n_complete"]
        cells = [
            f"{row['recovered_counts'].get(item['name'], 0)}/{n} selected, "
            f"{row['recalled_counts'].get(item['name'], 0)}/{n} found"
            for item in planted
        ]
        recovery_rows.append([_fmt(row["snr"], 2), row["arm"], *cells])
    lines += _table(("SNR", "arm", *(item["name"] for item in planted)), recovery_rows)
    lines += ["", NULL_HEADING, ""]
    lines += _table(NULL_HEADER, null_rows(summary))
    lines += [
        "",
        NULL_RECALL_HEADING,
        "",
        "Recall on the null markets, where the planted expressions carry no return: a high value here shows "
        "that recall measures whether a proposer writes such expressions, not whether it detects a signal.",
        "",
    ]
    null_recall_rows = []
    for row in summary.get("null_aggregate", []):
        n = row["n_complete"]
        null_recall_rows.append(
            [
                row["arm"],
                f"{n}/{row['n_runs']}",
                _mean_sd(row["metrics"]["recall_rate"], 2),
                *(f"{row['recalled_counts'].get(item['name'], 0)}/{n} found" for item in planted),
            ]
        )
    lines += _table(("arm", "complete runs", "recall", *(item["name"] for item in planted)), null_recall_rows)
    lines += [
        "",
        "## Oracle reference (planted signals scored on the test window)",
        "",
        "Power = share of planted signals whose one-sided test IC passes BH at "
        f"{config['test_alpha']} within the planted set (p-values against t(n - 1)).",
        "",
    ]
    oracle_rows = []
    for market in planted_markets:
        oracle = market["oracle_test"]
        cells = [
            f"IC {_fmt(oracle[item['name']]['ic_mean'], 4)}, t {_fmt(oracle[item['name']]['t_nw'], 2)}"
            for item in planted
        ]
        composite = oracle["composite"]
        cells.append(f"IC {_fmt(composite['ic_mean'], 4)}, t {_fmt(composite['t_nw'], 2)}")
        cells.append(_fmt(oracle.get("power"), 2))
        oracle_rows.append([_fmt(market["snr"], 2), str(market["seed"]), *cells])
    lines += _table(("SNR", "seed", *(item["name"] for item in planted), "planted composite", "power"), oracle_rows)
    lines += [
        "",
        LONG_SHORT_HEADING,
        "",
        "Annualized net Sharpe ratio of the composite's daily overlapping top-minus-bottom quintile "
        f"portfolio at {config['metric']['cost_bps']:g} bps per unit of one-way turnover, from "
        "the test window only. It is not a strategy estimate.",
        "",
    ]
    long_short_rows = []
    for row in summary["aggregate"]:
        stats = row["metrics"]["composite_test_ls_net_sharpe"]
        long_short_rows.append(
            [
                _fmt(row["snr"], 2),
                row["arm"],
                str(stats["n"]),
                _mean_sd(stats, 2),
                _span([stats["min"], stats["max"]], 2),
            ]
        )
    for snr in sorted({float(market["snr"]) for market in planted_markets}):
        values = [market["oracle_test"]["composite"]["ls_net_sharpe"] for market in planted_markets if float(market["snr"]) == snr]
        long_short_rows.append([_fmt(snr, 2), "oracle (planted composite)", str(len(values)), "n/a", _span(values, 2)])
    lines += _table(("SNR", "arm", "runs with a selection", "net Sharpe (mean ± sd)", "range"), long_short_rows)
    lines += ["", "## Per-run selection funnel", ""]
    funnel_rows = []
    for run in summary["runs"]:
        if not run.get("complete"):
            continue
        composite = run.get("composite_test") or {}
        funnel_rows.append(
            [
                _fmt(run["snr"], 2),
                str(run["seed"]),
                run["arm"],
                str(run["n_trials"]),
                str(run["n_evaluated"]),
                str(run["n_degenerate"]),
                str(run["n_behaviour_duplicates"]),
                str(run["n_screen_survivors"]),
                str(run["n_confirmed"]),
                str(run["n_rejected_correlation"]),
                str(run["n_selected"]),
                _fmt(run["recovery_rate"], 2),
                _fmt(run["fdp_true"], 2),
                _fmt(composite.get("ic_mean"), 4),
                f"`{run['ledger_head'][:12]}`",
            ]
        )
    lines += _table(
        (
            "SNR",
            "seed",
            "arm",
            "trials",
            "evaluated",
            "degenerate",
            "behav. duplicates",
            "screen survivors",
            "confirmed",
            "rejected (corr.)",
            "selected",
            "recovery",
            "true FDP",
            "composite test IC",
            "ledger head",
        ),
        funnel_rows,
    )
    lines += ["", "## Incomplete or aborted runs", ""]
    incomplete = summary.get("incomplete_runs", [])
    if incomplete:
        lines += _table(
            ("arm", "SNR", "seed", "stop reason", "trials", "aborted"),
            [
                [run["arm"], _fmt(run["snr"], 2), str(run["seed"]), str(run["stop_reason"]), str(run["n_trials"]), _fmt(bool(run["aborted"]))]
                for run in incomplete
            ],
        )
    else:
        lines.append("None: every run used its full budget.")
    lines += [
        "",
        "## Caveats",
        "",
        "- Synthetic data: the data-generating process, the planted signals and the SNR levels were "
        "chosen by the authors of this benchmark (the `drawn` signal by a random draw whose procedure was "
        "fixed in code before it was run - self-attested, not externally registered; see "
        "`results/planted_signal_draw.json`). A proposer's ranking here need not transfer to real markets.",
        "- The two `textbook` signals favour any proposer with prior knowledge of classic factors (for "
        "example an LLM); `library_family` is a documented idea whose Kakushadze (2016) formula is in the "
        "reference library; `drawn` was sampled from the random baseline's own grammar distribution, "
        "which if anything favours that baseline.",
        f"- {len(config['seeds'])} seeds per planted cell and {len(config['null_seeds'])} null seeds give "
        "coarse estimates; standard deviations are across runs.",
        "- Budgets of a few hundred trials are small for genetic programming; results for GP hold at this "
        "budget only (arms `name@budget` run larger budgets under the same trial accounting).",
        "- Ledger heads are SHA-256 hashes of each run's trial ledger (timestamps disabled), so a "
        "rerun on the same software stack can be checked record by record.",
        "",
    ]
    return "\n".join(lines)


def write_summary(summary: Mapping[str, Any], out_dir: str | Path, *, command: str) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``summary.md`` into ``out_dir``."""

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    payload = {**summary, "command": command}
    json_path = target / "summary.json"
    md_path = target / "summary.md"
    write_json(json_path, payload)
    # Render from the written (key-sorted) JSON rather than the in-memory payload,
    # so summary.md is exactly render_markdown(summary.json) on a fresh run too.
    written = json.loads(json_path.read_text(encoding="utf-8"))
    md_path.write_text(render_markdown(written, command=command), encoding="utf-8")
    return json_path, md_path


__all__ = [
    "HEADLINE_HEADER",
    "HEADLINE_HEADING",
    "LONG_SHORT_HEADING",
    "NULL_HEADER",
    "NULL_HEADING",
    "NULL_RECALL_HEADING",
    "RECOVERY_HEADING",
    "headline_rows",
    "null_rows",
    "render_markdown",
    "write_summary",
]
