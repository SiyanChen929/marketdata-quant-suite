#!/usr/bin/env python3
"""Re-run the pre-registered draw of the benchmark's "drawn" planted signal.

The procedure and its acceptance criteria are documented in
``llm_factor_mining.benchmark.draw``.  This script writes every candidate it
examined, with the reasons for each rejection, to
``results/planted_signal_draw.json`` (or ``--out``); the accepted expression
must equal ``llm_factor_mining.benchmark.synthetic.DRAWN_PLANTED``.

Usage (from the repository root)::

    PYTHONPATH=packages/quant-marketdata/src:projects/llm-factor-mining/src \\
        python projects/llm-factor-mining/scripts/draw_planted_signal.py

Synthetic data only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(PROJECT / "src"))

from llm_factor_mining.benchmark.draw import draw_planted_signal  # noqa: E402
from llm_factor_mining.benchmark.synthetic import DRAWN_PLANTED, FIXED_PLANTED  # noqa: E402
from llm_factor_mining.jsonutil import write_json  # noqa: E402


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=PROJECT / "results" / "planted_signal_draw.json")
    args = parser.parse_args(argv)
    record = draw_planted_signal(FIXED_PLANTED)
    record["note"] = "SYNTHETIC DATA - NOT EVIDENCE ABOUT REAL MARKETS. Pre-registered draw; see benchmark/draw.py."
    write_json(args.out, record)
    accepted = record["accepted"]
    print(f"accepted candidate k={accepted['k']}: {accepted['expression']} ({len(record['rejected'])} rejected)")
    if accepted["expression"] != DRAWN_PLANTED.expression:
        print("the accepted expression differs from DRAWN_PLANTED", file=sys.stderr)
        return 1
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
