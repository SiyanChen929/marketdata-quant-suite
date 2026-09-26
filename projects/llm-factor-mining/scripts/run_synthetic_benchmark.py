#!/usr/bin/env python3
"""Run the synthetic planted-alpha benchmark (random vs evolutionary, optional LLM).

Writes ``results/synthetic_benchmark/summary.json`` and ``summary.md`` inside
this project (or ``--out``); per-run ledgers go to a fresh, time-stamped
subdirectory of ``runs/synthetic_benchmark`` (gitignored) on every
invocation.  The grid covers planted markets (SNR 0.05 and 0.15, seeds 0-2)
and null markets (snr = 0, seeds 0-9).  The LLM arm runs only with
``--backend anthropic`` (credentials required; responses are recorded to a
fresh ``llm_responses.jsonl`` next to the summary) or ``--backend replay
--replay-file PATH``; without ``--backend`` it is recorded as not run, and a
requested backend that cannot run stops the script before anything runs.

Example (from the repository root)::

    PYTHONPATH=packages/quant-marketdata/src:projects/llm-factor-mining/src \
        python projects/llm-factor-mining/scripts/run_synthetic_benchmark.py

All results are synthetic-data evidence only.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import shlex
import sys


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(PROJECT / "src"))

from llm_factor_mining.cli import build_parser, run_benchmark_from_args  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    defaults: list[str] = []
    if not any(part == "--out" or part.startswith("--out=") for part in raw):
        defaults += ["--out", str(PROJECT / "results" / "synthetic_benchmark")]
    if not any(part == "--runs-dir" or part.startswith("--runs-dir=") for part in raw):
        defaults += ["--runs-dir", str(PROJECT / "runs" / "synthetic_benchmark")]
    args = build_parser().parse_args(["benchmark", *raw, *defaults])
    command = (
        "PYTHONPATH=packages/quant-marketdata/src:projects/llm-factor-mining/src "
        "python projects/llm-factor-mining/scripts/run_synthetic_benchmark.py"
    )
    if raw:
        command += " " + shlex.join(raw)
    json_path, md_path, summary, runs_path = run_benchmark_from_args(args, command=command)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    print(f"per-run ledgers in {runs_path}")
    print(f"total runtime {summary['runtime_seconds']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
