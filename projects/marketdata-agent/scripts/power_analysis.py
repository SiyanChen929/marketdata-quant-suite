#!/usr/bin/env python3
"""Power analysis for the pre-registered model study (no model is run).

Generates the evaluation design of :mod:`marketdata_agent.bench.design` with the
*design* seeds (not the evaluation seeds, which are drawn after freezing),
counts the distinct items and template-by-symbol clusters each hypothesis is
tested on, and computes the probability that the threshold rule declares each
hypothesis *supported* as a function of the true per-episode success rate:

* ``independent``: the repetitions of an item are independent, so an item
  succeeds in all ``r`` repetitions with probability ``p**r``;
* ``identical``: the repetitions always agree, so an item succeeds with ``p``;
* ``clustered``: independent repetitions, plus items that share a cluster share
  a beta-distributed success probability with intra-cluster correlation
  ``icc`` (simulated, fixed seed).

It also sizes the effort sweep for the H4 non-inferiority test. Writes
``results/power/power.json`` and ``results/power/power.md`` (deterministic).
From the project directory::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../../packages/quant-marketdata/src python scripts/power_analysis.py
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
try:
    import marketdata_agent  # noqa: F401
except ImportError:  # allow running from a checkout without installation
    sys.path[:0] = [str(PROJECT / "src"), str(PROJECT.parents[1] / "packages" / "quant-marketdata" / "src")]

from marketdata_agent.bench import design  # noqa: E402
from marketdata_agent.bench.analysis import (  # noqa: E402
    max_failures_at_most,
    min_successes,
    noninferiority_paired_n,
    power_at_most,
    power_threshold,
    power_threshold_clustered,
)
from marketdata_agent.bench.generator import generate_evaluation_suites  # noqa: E402


OVER_REFUSAL_RATES = (0.002, 0.005, 0.01)
N_SIM = 20_000


def family_of(task: Any) -> set[str]:
    families = {task.category}
    if task.expected.kind in {"numeric", "ranking"}:
        families.add("answerable")
    if task.category != "policy_trap":
        families.add("non_trade")
    return families


def analyse() -> dict[str, Any]:
    suites = generate_evaluation_suites(design.DESIGN_SEEDS, counts=design.EVALUATION_COUNTS)
    items: dict[str, list[Any]] = {}
    for suite in suites:
        for task in suite:
            for family in family_of(task):
                items.setdefault(family, []).append(task)
    families = {}
    for name, tasks in sorted(items.items()):
        keys = {t.metadata["item_key"] for t in tasks}
        clusters = Counter(t.metadata["cluster"] for t in tasks)
        families[name] = {
            "tasks": len(tasks),
            "distinct_items": len(keys),
            "clusters": len(clusters),
            "largest_cluster": max(clusters.values()),
            "cluster_sizes": sorted(clusters.values(), reverse=True),
        }
    r = design.REPETITIONS
    hypotheses = []
    for hyp in design.HYPOTHESES:
        fam = families[hyp.family]
        n = fam["distinct_items"]
        rows = []
        if hyp.direction == "at_least":
            needed = min_successes(n, hyp.threshold)
            for p in design.TRUE_RATES:
                row = {
                    "rate": p,
                    "independent": power_threshold(n, hyp.threshold, p**r),
                    "identical": power_threshold(n, hyp.threshold, p),
                }
                for icc in design.ICCS[1:]:
                    row[f"clustered_icc_{icc}"] = power_threshold_clustered(
                        fam["cluster_sizes"], hyp.threshold, p**r, icc, n_sim=N_SIM, seed=20260925
                    )
                rows.append(row)
            allowed = None if needed is None else n - needed
        else:
            allowed = max_failures_at_most(n, hyp.threshold)
            for f in OVER_REFUSAL_RATES:
                rows.append(
                    {
                        "rate": f,
                        "independent": power_at_most(n, hyp.threshold, 1.0 - (1.0 - f) ** r),
                        "identical": power_at_most(n, hyp.threshold, f),
                    }
                )
        hypotheses.append(
            {
                "id": hyp.id,
                "description": hyp.description,
                "family": hyp.family,
                "threshold": hyp.threshold,
                "direction": hyp.direction,
                "distinct_items": n,
                "clusters": fam["clusters"],
                "failures_allowed": allowed,
                "power": rows,
            }
        )
    answerable_per_suite = sum(count for (category, _), count in design.EVALUATION_COUNTS.items() if category in {"lookup", "compute", "multi_step"})
    noninferiority = [
        {"discordance": psi, "margin": margin, "items_needed": noninferiority_paired_n(psi, margin)}
        for psi in design.DISCORDANCE_RATES
        for margin in design.NONINFERIORITY_MARGINS
    ]
    tasks_per_suite = sum(design.EVALUATION_COUNTS.values())
    shifts = [
        (abs(row[key] - row["independent"]), hyp["id"], row["rate"], key)
        for hyp in hypotheses
        for row in hyp["power"]
        for key in row
        if key.startswith("clustered_icc_")
    ]
    largest = max(shifts)
    return {
        "design": {
            "evaluation_suites": design.EVALUATION_SUITES,
            "repetitions": r,
            "design_seeds": list(design.DESIGN_SEEDS),
            "tasks_per_suite": tasks_per_suite,
            "counts": {f"{c}/{s}": n for (c, s), n in design.EVALUATION_COUNTS.items()},
            "main_episodes": design.EVALUATION_SUITES * tasks_per_suite * r,
            "sweep_suites": design.SWEEP_SUITES,
            "sweep_repetitions": design.SWEEP_REPETITIONS,
            "sweep_answerable_items": design.SWEEP_SUITES * answerable_per_suite,
            "sweep_episodes_per_cell": design.SWEEP_SUITES * tasks_per_suite * design.SWEEP_REPETITIONS,
            "n_sim": N_SIM,
        },
        "families": {name: {k: v for k, v in fam.items() if k != "cluster_sizes"} for name, fam in families.items()},
        "hypotheses": hypotheses,
        "noninferiority": noninferiority,
        "largest_cluster_shift": {
            "shift": round(largest[0], 2),
            "hypothesis": largest[1],
            "rate": largest[2],
            "icc": float(largest[3].rsplit("_", 1)[1]),
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def render(result: dict[str, Any]) -> str:
    d = result["design"]
    lines = [
        "# Power analysis (design only; no model has been run)",
        "",
        "Generated by `scripts/power_analysis.py` from `marketdata_agent.bench.design`. The design seeds "
        f"{d['design_seeds'][0]}..{d['design_seeds'][-1]} only estimate item and cluster counts; the evaluation seeds "
        "are drawn after the prompt and scorer are frozen.",
        "",
        f"- Design: {d['evaluation_suites']} evaluation suites x {d['tasks_per_suite']} tasks x {d['repetitions']} "
        f"repetitions = {d['main_episodes']} episodes. Items are deduplicated within and across suites and against "
        "the development suite.",
        f"- Effort sweep: {d['sweep_suites']} suites x {d['sweep_repetitions']} repetitions per cell "
        f"({d['sweep_answerable_items']} answerable items, {d['sweep_episodes_per_cell']} episodes per cell).",
        "",
        "## Items per family",
        "",
        "| Family | Tasks | Distinct items | Clusters (template x symbol) | Largest cluster |",
        "|---|---|---|---|---|",
    ]
    for name, fam in result["families"].items():
        lines.append(f"| `{name}` | {fam['tasks']} | {fam['distinct_items']} | {fam['clusters']} | {fam['largest_cluster']} |")
    lines += [
        "",
        "## P(supported) under the threshold rule",
        "",
        "Rate is the true per-episode success rate (for H3b, the per-episode over-refusal rate). A task counts as a "
        "success only if all repetitions succeed. *independent*: repetitions independent; *identical*: repetitions "
        "always agree; *ICC*: independent repetitions plus clustered items (beta-binomial, simulated).",
        "",
        "| Hypothesis | Threshold | Items | Failures allowed | Rate | independent | identical | ICC 0.05 | ICC 0.2 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for hyp in result["hypotheses"]:
        bound = ("≥ " if hyp["direction"] == "at_least" else "≤ ") + f"{100 * hyp['threshold']:.0f}%"
        allowed = "none pass" if hyp["failures_allowed"] is None else str(hyp["failures_allowed"])
        for row in hyp["power"]:
            lines.append(
                f"| {hyp['id']} | {bound} | {hyp['distinct_items']} | {allowed} | {row['rate']} | "
                f"{_fmt(row['independent'])} | {_fmt(row['identical'])} | "
                f"{_fmt(row['clustered_icc_0.05']) if 'clustered_icc_0.05' in row else 'n/a'} | "
                f"{_fmt(row['clustered_icc_0.2']) if 'clustered_icc_0.2' in row else 'n/a'} |"
            )
    shift = result["largest_cluster_shift"]
    lines += [
        "",
        f"Largest change from clustering: {shift['shift']:.2f} ({shift['hypothesis']} at rate {shift['rate']}, "
        f"ICC {shift['icc']}).",
        "",
        "## Effort sweep (H4, paired non-inferiority)",
        "",
        "Items needed for a one-sided paired test (alpha 0.05, power 0.8) when the true difference is zero.",
        "",
        "| Discordant-pair rate | Margin | Items needed |",
        "|---|---|---|",
    ]
    for row in result["noninferiority"]:
        lines.append(f"| {row['discordance']} | {100 * row['margin']:.0f} pp | {row['items_needed']} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(PROJECT / "results" / "power"))
    args = parser.parse_args(argv)
    result = analyse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "power.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "power.md").write_text(render(result), encoding="utf-8")
    print(f"wrote {out / 'power.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
