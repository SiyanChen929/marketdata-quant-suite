#!/usr/bin/env python3
"""Run the offline harness-validation baselines on the generated benchmark suite.

Writes ``results/benchmark/`` (relative to the project directory unless ``--out`` is given):

* ``summary.json`` / ``summary.md``: combined headline and per-category metrics;
* ``suite.json``: every task with its ground truth (fingerprinted);
* ``<agent>/episodes.jsonl``, ``<agent>/summary.json``, ``<agent>/summary.md``;
* ``<agent>/audit/``: hash-chained audit logs (git-ignored). Scripted runs use a
  fixed audit timestamp, so rerunning regenerates each log byte for byte, and
  its head must equal the head recorded in the committed ``summary.json``.

The six agents are scripted baselines, not language models. They validate
the harness on synthetic data. ``oracle_no_clock`` is the oracle under the
no-clock ablation (A1) and must match ``oracle`` exactly. LLM results require an API key and are run
separately (``marketdata-agent bench run --agent anthropic``).

From the suite root::

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/quant-marketdata/src:projects/marketdata-agent/src \\
      python projects/marketdata-agent/scripts/run_benchmark.py
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


PROJECT = Path(__file__).resolve().parents[1]
try:
    import marketdata_agent  # noqa: F401
except ImportError:  # allow running from a checkout without installation
    sys.path[:0] = [str(PROJECT / "src"), str(PROJECT.parents[1] / "packages" / "quant-marketdata" / "src")]

from marketdata_agent.bench import BASELINE_NAMES, DEFAULT_SEED, baseline_agent, generate_suite, run_suite  # noqa: E402


REPRODUCE = (
    "PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/quant-marketdata/src:projects/marketdata-agent/src "
    "python projects/marketdata-agent/scripts/run_benchmark.py"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(PROJECT / "results" / "benchmark"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--tasks", type=int, default=None, help="stratified subset (default: the full suite)")
    parser.add_argument("--agents", default=",".join(BASELINE_NAMES))
    args = parser.parse_args(argv)

    suite = generate_suite(args.seed)
    if args.tasks is not None:
        suite = suite.subset(args.tasks)
    names = [name.strip() for name in args.agents.split(",") if name.strip()]
    extra = [f"--seed {args.seed}"] if args.seed != DEFAULT_SEED else []
    extra += [f"--tasks {args.tasks}"] if args.tasks is not None else []
    extra += [f"--agents {','.join(names)}"] if names != list(BASELINE_NAMES) else []
    command = " ".join([REPRODUCE, *extra])

    out = Path(args.out)
    started = time.perf_counter()
    combined = run_suite([baseline_agent(name) for name in names], suite, out, command=command, overwrite=True)
    suite.write_json(out / "suite.json")
    elapsed = time.perf_counter() - started

    print(combined["banner"])
    print(
        f"suite: {len(suite)} tasks ({len(set(suite.item_keys()))} distinct items), sha256 {suite.sha256()[:16]}, "
        f"seed {args.seed}"
    )
    for name, agent in combined["agents"].items():
        o = agent["overall"]
        print(
            f"{name:16s} accuracy {o['accuracy']['k']:3d}/{o['accuracy']['n']} "
            f"grounding {o['grounding_rate']['k']}/{o['grounding_rate']['n']} "
            f"look-ahead-episodes {o['lookahead_attempt_episodes']['k']} leak-episodes {o['leak_episodes']['k']} "
            f"denied-call-episodes {o['denied_call_episodes']['k']} audit {'ok' if agent['audit']['ok'] else 'FAILED'} "
            f"head {agent['audit']['head_hash'][:16]}"
        )
    print(f"wrote {out / 'summary.md'} in {elapsed:.1f}s")
    return 0 if combined["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
