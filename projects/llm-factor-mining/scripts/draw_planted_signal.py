#!/usr/bin/env python3
"""Re-run the draw of the benchmark's "drawn" planted signal.

The procedure and its acceptance criteria are fixed in code in
``llm_factor_mining.benchmark.draw`` (self-attested; not externally
registered).  The record lists every candidate examined, with the reasons for
each rejection; the accepted expression must equal
``llm_factor_mining.benchmark.synthetic.DRAWN_PLANTED``.

By default the script only *verifies*: it re-runs the draw and compares the
record with the committed ``results/planted_signal_draw.json`` without
writing anything.  ``--out PATH`` writes the record to ``PATH`` instead.

Usage (from the repository root)::

    PYTHONPATH=packages/quant-marketdata/src:projects/llm-factor-mining/src \\
        python projects/llm-factor-mining/scripts/draw_planted_signal.py

Synthetic data only.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT / "src") not in sys.path:  # allow running without an editable install
    sys.path.insert(0, str(PROJECT / "src"))

from llm_factor_mining.benchmark.draw import draw_planted_signal  # noqa: E402
from llm_factor_mining.benchmark.synthetic import DRAWN_PLANTED, FIXED_PLANTED  # noqa: E402
from llm_factor_mining.jsonutil import json_safe, write_json  # noqa: E402


COMMITTED = PROJECT / "results" / "planted_signal_draw.json"
NOTE = (
    "SYNTHETIC DATA - NOT EVIDENCE ABOUT REAL MARKETS. Draw procedure fixed in code (benchmark/draw.py) "
    "before the draw was run; self-attested, not externally registered."
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out", type=Path, default=None, help="write the record here (default: only compare with the committed record)"
    )
    args = parser.parse_args(argv)
    record = draw_planted_signal(FIXED_PLANTED)
    record["note"] = NOTE
    accepted = record["accepted"]
    print(f"accepted candidate k={accepted['k']}: {accepted['expression']} ({len(record['rejected'])} rejected)")
    if accepted["expression"] != DRAWN_PLANTED.expression:
        print("the accepted expression differs from DRAWN_PLANTED", file=sys.stderr)
        return 1
    if args.out is not None:
        write_json(args.out, record)
        print(f"wrote {args.out}")
        return 0
    committed = json.loads(COMMITTED.read_text(encoding="utf-8"))
    if json.loads(json.dumps(json_safe(record))) != committed:
        print(f"the re-run draw differs from the committed {COMMITTED}", file=sys.stderr)
        return 1
    print(f"matches the committed {COMMITTED} (nothing written)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
