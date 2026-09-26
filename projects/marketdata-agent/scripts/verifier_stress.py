#!/usr/bin/env python3
"""Stress-test the grounding verifier with fabricated claims next to real tool results.

Two measurements, both on the synthetic development suite and both deterministic:

1. **Per-mode specificity of the committed ``ungrounded`` run**: for each
   ``ungrounded_mode``, how many claims the verifier accepted, and whether the
   episodes had real results to match against.
2. **Injection into oracle episodes**: the oracle policy is rerun in process,
   so every episode has its real tool results. Into each episode's answer
   context the script injects fabricated claims and records how often the
   verifier *accepts* one (false acceptance):

   * ``random_uncited`` / ``random_cited``: a plausible value of the task's
     quantity, drawn at random and shown with 0 to 3 decimals, uncited (it may
     match any output of the episode) or cited to a real result;
   * ``near_miss``: the true value one unit off in the last shown decimal
     (two units when one unit would still be a correct rounding of a tie);
   * ``sign_flip``: the true value with the opposite sign (unsigned when the
     truth is negative);
   * ``wrong_row`` / ``wrong_field``: for daily-bar lookups, another row's
     close, or the period high, reported as the close;
   * ``range_last_close`` / ``range_first_close``: for questions about the
     close on a named date, the ``last_close`` of a range that starts on that
     date and ends at the latest session, or the ``first_close`` of a range
     that ends on it, reported as the close on the named date (a close from
     another session).

Random values are drawn from fixed ranges (see ``SAMPLING`` and the report):
the false-acceptance rates are chance rates for values drawn from those
ranges, not bounds on a model's errors, which cluster near true or related
outputs (the near-miss and wrong-session rows are closer to that error model).

Rates are false-acceptance rates with Wilson 95% intervals. These are
properties of the verifier on synthetic answers, not of any model. Writes
``results/verifier/verifier_stress.json`` and ``.md``. From the project
directory::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:../../packages/quant-marketdata/src python scripts/verifier_stress.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT = Path(__file__).resolve().parents[1]
try:
    import marketdata_agent  # noqa: F401
except ImportError:  # allow running from a checkout without installation
    sys.path[:0] = [str(PROJECT / "src"), str(PROJECT.parents[1] / "packages" / "quant-marketdata" / "src")]

import numpy as np  # noqa: E402

from marketdata_agent import Copilot, FrameBarSource, ToolRuntime  # noqa: E402
from marketdata_agent.bench import DEFAULT_SEED, generate_suite, ungrounded_mode  # noqa: E402
from marketdata_agent.bench.baselines import baseline_backend, near_miss  # noqa: E402
from marketdata_agent.bench.generator import dataset_frame  # noqa: E402
from marketdata_agent.bench.scoring import rate  # noqa: E402
from marketdata_agent.grounding import verify_grounding  # noqa: E402


SAMPLES = 20
DECIMALS = {"percent": (0, 1, 2), "usd": (0, 1, 2), "ratio": (1, 2, 3), "count": (0,)}
DISPLAY_STEP = {"percent": 1e-4, "usd": 1e-2, "ratio": 1e-3, "count": 1.0}
SEED = 20260925
SAMPLING = {
    "annualized_volatility": "uniform(12%, 45%)",
    "max_drawdown": "-uniform(4%, 30%)",
    "correlation": "uniform(-0.3, 0.8)",
    "last_close, reference_close": "uniform($20, $300)",
    "count": "integer from 7 to 14",
    "simple_return (and other keys)": "normal(mean 2%, sd 8%)",
}


def _rng(*parts: object) -> np.random.Generator:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def _show(value: float, unit: str, decimals: int, *, signed_plus: bool = False) -> str:
    if unit == "percent":
        text = f"{value * 100:.{decimals}f}%"
    elif unit == "usd":
        text = f"${value:.{decimals}f}" if value >= 0 else f"-${-value:.{decimals}f}"
    elif unit == "count":
        text = f"{int(round(value))}"
    else:
        text = f"{value:.{decimals}f}"
    return ("+" + text) if signed_plus and value >= 0 else text


def _plausible(key: str, rng: np.random.Generator) -> tuple[float, str]:
    if key == "annualized_volatility":
        return float(rng.uniform(0.12, 0.45)), "percent"
    if key == "max_drawdown":
        return -float(rng.uniform(0.04, 0.30)), "percent"
    if key == "correlation":
        return float(rng.uniform(-0.3, 0.8)), "ratio"
    if key in {"last_close", "reference_close"}:
        return float(rng.uniform(20.0, 300.0)), "usd"
    if key == "count":
        return float(rng.integers(7, 15)), "count"
    return float(rng.normal(0.02, 0.08)), "percent"


def _outputs_bin(count: int) -> str:
    return "1-5" if count <= 5 else ("6-20" if count <= 20 else "21+")


def ungrounded_by_mode(results_dir: Path, suite: Any) -> list[dict[str, Any]]:
    tasks = {task.id: task for task in suite}
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    path = results_dir / "ungrounded" / "episodes.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        task = tasks[record["task"]["id"]]
        mode = ungrounded_mode(task)
        grounding = record["episode"]["grounding"]
        entry = stats[mode]
        entry["episodes"] += 1
        entry["episodes_with_results"] += int(bool(record["episode"]["result_ids"]))
        entry["claims"] += grounding["n_claims"]
        entry["accepted"] += grounding["n_supported"]
        entry["correct"] += int(record["score"]["correct"])
    rows = []
    for mode in sorted(stats):
        entry = stats[mode]
        rows.append({"mode": mode, **dict(entry), "false_acceptance": rate(entry["accepted"], entry["claims"])})
    return rows


def oracle_injections(suite: Any) -> dict[str, Any]:
    source = FrameBarSource(dataset_frame(suite.dataset))
    by_cell: dict[tuple[str, str, int], list[int]] = defaultdict(lambda: [0, 0])
    by_outputs: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    episodes = 0
    for task in suite:
        episode = Copilot(baseline_backend("oracle", task), source=source, as_of=task.as_of).run(task.question, episode_id=task.id)
        if not episode.results:
            continue
        episodes += 1
        results = episode.results
        rids = sorted(results)
        n_outputs = sum(len(results[r].provenance.outputs) for r in rids)
        form = task.answer_form
        key = form.key if form else "reference_close"
        truth_rid, truth = None, None
        if form is not None and task.expected.kind == "numeric":
            for rid in rids:
                if key in results[rid].provenance.outputs:
                    truth_rid, truth = rid, results[rid].provenance.outputs[key]
        rng = _rng(SEED, task.id)

        def record(mode: str, unit: str, decimals: int, text: str, rid: str | None, pool: Any = None) -> None:
            cite = f" [r:{rid}]" if rid else ""
            report = verify_grounding(f"ANSWER: {text}{cite}", pool or results, question=task.question)
            accepted = int(any(check.supported for check in report.checks))
            cell = by_cell[(mode, unit, decimals)]
            cell[0] += accepted
            cell[1] += 1
            if mode == "random_uncited":
                bucket = by_outputs[(mode, _outputs_bin(n_outputs))]
                bucket[0] += accepted
                bucket[1] += 1

        for _ in range(SAMPLES):
            value, unit = _plausible(key, rng)
            for decimals in DECIMALS[unit]:
                record("random_uncited", unit, decimals, _show(value, unit, decimals), None)
                record("random_cited", unit, decimals, _show(value, unit, decimals), rids[int(rng.integers(0, len(rids)))])
        if truth is None or truth_rid is None:
            continue
        unit = form.unit or "percent"
        top = DECIMALS[unit][-1]
        step = DISPLAY_STEP[unit]
        for direction in (-1.0, 1.0):
            record("near_miss", unit, top, _show(near_miss(truth, step, direction), unit, top), truth_rid)
        if unit in {"percent", "ratio"} and abs(truth) >= step:
            record("sign_flip", unit, top, _show(-truth, unit, top, signed_plus=key == "max_drawdown"), truth_rid)
        if form.tool == "get_daily_bars":
            payload = results[truth_rid].payload
            rows = list(payload.get("rows") or [])
            for row in rows[:-1]:
                if abs(float(row["close"]) - truth) >= 0.005:
                    record("wrong_row", "usd", 2, _show(float(row["close"]), "usd", 2), truth_rid)
            high = float((payload.get("summary") or {}).get("max_high", truth))
            if abs(high - truth) >= 0.005:
                record("wrong_field", "usd", 2, _show(high, "usd", 2), truth_rid)
            args = dict(results[truth_rid].provenance.args)
            if args.get("start") == args.get("end"):  # the question names one date: inject closes of other sessions
                day = str(args["start"])
                runtime = ToolRuntime.for_source(source, task.as_of)
                earlier = (np.datetime64(day) - np.timedelta64(30, "D")).astype(str)
                ranges = (
                    ("range_last_close", "last_close", "last_date", {"start": day, "end": "latest"}),
                    ("range_first_close", "first_close", "first_date", {"start": earlier, "end": day}),
                )
                for mode, key, date_key, bounds in ranges:
                    extra = runtime.call_tool("get_daily_bars", {"symbol": args["symbol"], **bounds}).result
                    if extra is None or extra.payload[date_key] == day:
                        continue
                    value = float(extra.payload["summary"][key])
                    if abs(value - truth) >= 0.005:
                        pool = {**results, extra.result_id: extra}
                        record(mode, "usd", 2, _show(value, "usd", 2), extra.result_id, pool)
                        record(mode + "_uncited", "usd", 2, _show(value, "usd", 2), None, pool)
    cells = [
        {"mode": mode, "unit": unit, "decimals": decimals, **rate(accepted, total)}
        for (mode, unit, decimals), (accepted, total) in sorted(by_cell.items())
    ]
    outputs = [
        {"mode": mode, "outputs_in_episode": bucket, **rate(accepted, total)}
        for (mode, bucket), (accepted, total) in sorted(by_outputs.items())
    ]
    return {
        "episodes_with_results": episodes,
        "samples_per_episode": SAMPLES,
        "sampling": SAMPLING,
        "cells": cells,
        "by_outputs": outputs,
    }


def _pct(entry: dict[str, Any]) -> str:
    if entry.get("rate") is None:
        return "n/a"
    low, high = entry["ci95"]
    return f"{100 * entry['rate']:.1f}% [{100 * low:.1f}, {100 * high:.1f}]"


def render(result: dict[str, Any]) -> str:
    lines = [
        "# Grounding-verifier stress test (synthetic answers; not model results)",
        "",
        "Generated by `scripts/verifier_stress.py`. False acceptance is the share of fabricated or wrong claims that "
        "the verifier marks as supported; intervals are Wilson 95%.",
        "",
        "## Committed `ungrounded` baseline, by mode",
        "",
        "| Mode | Episodes | With tool results | Claims | Accepted | False acceptance | Scored correct |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["ungrounded_by_mode"]:
        lines.append(
            f"| `{row['mode']}` | {row['episodes']} | {row['episodes_with_results']} | {row['claims']} | {row['accepted']} | "
            f"{_pct(row['false_acceptance'])} | {row['correct']} |"
        )
    inj = result["oracle_injections"]
    lines += [
        "",
        f"## Claims injected next to real results ({inj['episodes_with_results']} oracle episodes)",
        "",
        "| Mode | Unit | Decimals shown | Accepted / injected | False acceptance |",
        "|---|---|---|---|---|",
    ]
    for cell in inj["cells"]:
        lines.append(f"| `{cell['mode']}` | {cell['unit']} | {cell['decimals']} | {cell['k']}/{cell['n']} | {_pct(cell)} |")
    lines += [
        "",
        "Random values (`random_uncited`, `random_cited`) are drawn from these ranges, so their false-acceptance "
        "rates are chance rates for such draws, not bounds on how often a model's wrong numbers are accepted: a "
        "model's errors cluster near true or related outputs, which the `near_miss`, `sign_flip`, `wrong_row`, "
        "`wrong_field` and `range_*` rows probe instead.",
        "",
        "| Quantity asked about | Random value drawn from |",
        "|---|---|",
    ]
    for key, text in inj["sampling"].items():
        lines.append(f"| `{key}` | {text} |")
    lines += [
        "",
        "## Uncited random claims by number of numeric outputs in the episode",
        "",
        "| Outputs in episode | Accepted / injected | False acceptance |",
        "|---|---|---|",
    ]
    for row in inj["by_outputs"]:
        lines.append(f"| {row['outputs_in_episode']} | {row['k']}/{row['n']} | {_pct(row)} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", default=str(PROJECT / "results" / "benchmark"), help="committed benchmark results")
    parser.add_argument("--out", default=str(PROJECT / "results" / "verifier"))
    args = parser.parse_args(argv)
    suite = generate_suite(DEFAULT_SEED)
    result = {
        "suite_sha256": suite.sha256(),
        "ungrounded_by_mode": ungrounded_by_mode(Path(args.results), suite),
        "oracle_injections": oracle_injections(suite),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verifier_stress.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "verifier_stress.md").write_text(render(result), encoding="utf-8")
    print(f"wrote {out / 'verifier_stress.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
