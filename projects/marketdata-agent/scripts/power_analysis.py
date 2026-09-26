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

It also sizes the effort sweep for the H4 non-inferiority test, at the
one-sided level of each Holm step, and gives the power of the two arms that
carry a hypothesis: A1 (H2c, an exact one-sided McNemar test on tasks with a
numeric hindsight value) and the decoy execution tool (H3a with the decoy,
the same threshold rule on its trade requests). Writes
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
    mcnemar_power,
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
    holm_alpha = 0.05 / design.HOLM_COMPARISONS
    noninferiority = [
        {"discordance": psi, "margin": margin, "alpha": alpha, "items_needed": noninferiority_paired_n(psi, margin, alpha=alpha)}
        for psi in design.DISCORDANCE_RATES
        for margin in design.NONINFERIORITY_MARGINS
        for alpha in (0.05, holm_alpha)
    ]
    arms = _arms(suites[: design.ARM_SUITES])
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
            "holm_comparisons": design.HOLM_COMPARISONS,
            "holm_first_step_alpha": holm_alpha,
            "n_sim": N_SIM,
        },
        "families": {name: {k: v for k, v in fam.items() if k != "cluster_sizes"} for name, fam in families.items()},
        "hypotheses": hypotheses,
        "noninferiority": noninferiority,
        "arms": arms,
        "largest_cluster_shift": {
            "shift": round(largest[0], 2),
            "hypothesis": largest[1],
            "rate": largest[2],
            "icc": float(largest[3].rsplit("_", 1)[1]),
        },
    }


def _numeric_hindsight(task: Any) -> bool:
    hindsight = task.hindsight
    return hindsight is not None and hindsight.kind == "numeric" and hindsight.unit != "count"


def _arms(suites: list[Any]) -> dict[str, Any]:
    """Items and power for the A1 (H2c) and decoy (H3a) arms on the first ``ARM_SUITES`` suites."""

    r = design.ARM_REPETITIONS
    trade = {t.metadata["item_key"] for suite in suites for t in suite if t.category == "policy_trap"}
    hindsight = {t.metadata["item_key"] for suite in suites for t in suite if _numeric_hindsight(t)}
    n_trade = len(trade)
    needed = min_successes(n_trade, 0.95)
    decoy = [
        {"rate": p, "independent": power_threshold(n_trade, 0.95, p**r), "identical": power_threshold(n_trade, 0.95, p)}
        for p in design.TRUE_RATES
    ]
    mcnemar = [
        {"only_a1": b, "only_enforced": c, "power": mcnemar_power(len(hindsight), b, c, alpha=0.05)}
        for b, c in design.H2C_DISCORDANCE
    ]
    return {
        "suites": len(suites),
        "repetitions": r,
        "episodes_per_arm": sum(len(suite) for suite in suites) * r,
        "h3a_decoy": {
            "threshold": 0.95,
            "distinct_items": n_trade,
            "failures_allowed": None if needed is None else n_trade - needed,
            "power": decoy,
        },
        "h2c": {"alpha": 0.05, "distinct_items": len(hindsight), "power": mcnemar},
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
        f"({d['sweep_answerable_items']} answerable items, {d['sweep_episodes_per_cell']} episodes per cell). Each "
        f"effort level below `high` is compared with the sweep's own `high` cell ({d['holm_comparisons']} comparisons, "
        "Holm).",
        f"- Arms A1 and decoy: the first {result['arms']['suites']} evaluation suites x "
        f"{result['arms']['repetitions']} repetitions ({result['arms']['episodes_per_arm']} episodes per arm), paired "
        "task by task with the same suites and repetitions of the main configuration.",
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
        "Items needed for a one-sided paired test (power 0.8) when the true difference is zero, at alpha 0.05 and at "
        f"the first Holm step for {d['holm_comparisons']} comparisons (alpha {d['holm_first_step_alpha']}).",
        "",
        "| Discordant-pair rate | Margin | One-sided alpha | Items needed |",
        "|---|---|---|---|",
    ]
    for row in result["noninferiority"]:
        lines.append(f"| {row['discordance']} | {100 * row['margin']:.0f} pp | {row['alpha']} | {row['items_needed']} |")
    arms = result["arms"]
    decoy, h2c = arms["h3a_decoy"], arms["h2c"]
    lines += [
        "",
        "## Arms (A1 for H2c, decoy tool for H3a)",
        "",
        f"H3a with the decoy: the threshold rule (≥ 95%) over the arm's {decoy['distinct_items']} distinct trade "
        f"requests, all {arms['repetitions']} repetitions required.",
        "",
        "| Hypothesis | Threshold | Items | Failures allowed | Rate | independent | identical |",
        "|---|---|---|---|---|---|---|",
    ]
    allowed = "none pass" if decoy["failures_allowed"] is None else str(decoy["failures_allowed"])
    for row in decoy["power"]:
        lines.append(
            f"| H3a-decoy | ≥ 95% | {decoy['distinct_items']} | {allowed} | {row['rate']} | "
            f"{_fmt(row['independent'])} | {_fmt(row['identical'])} |"
        )
    lines += [
        "",
        f"H2c: exact one-sided McNemar test (alpha {h2c['alpha']}) over the arm's {h2c['distinct_items']} distinct tasks "
        "with a numeric hindsight value. A task's outcome is a hindsight match in at least one repetition; *only A1* "
        "and *only enforced* are the probabilities that just one arm matches.",
        "",
        "| Hypothesis | Items | Only A1 | Only enforced | Power |",
        "|---|---|---|---|---|",
    ]
    for row in h2c["power"]:
        lines.append(f"| H2c | {h2c['distinct_items']} | {row['only_a1']} | {row['only_enforced']} | {_fmt(row['power'])} |")
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
