#!/usr/bin/env python3
"""Validate a legacy bar file and optionally import it into the shared store."""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import pandas as pd

from quant_marketdata import DataContractError, MarketDataStore, normalize_bars


PROVISIONAL_MARKERS = ("partial", "provisional", "intraday", "with_partial")


def load_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError("input must be CSV or Parquet")


def prepare(
    frame: pd.DataFrame,
    *,
    finality: str,
    source: str,
    attest_confirmed: bool = False,
) -> pd.DataFrame:
    bars = frame.copy()
    if "source" not in bars:
        bars["source"] = source
    if "finality" not in bars:
        if finality == "confirmed" and not attest_confirmed:
            raise DataContractError(
                "unlabeled input cannot be promoted to confirmed without "
                "--attest-completed-sessions"
            )
        bars["finality"] = finality
    return normalize_bars(bars, finality=finality)


def source_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--data-home", type=Path, default=None)
    parser.add_argument("--finality", choices=("confirmed", "provisional"), required=True)
    parser.add_argument("--source", default="marketdata.app")
    parser.add_argument(
        "--attest-completed-sessions",
        action="store_true",
        help="Required to label an input lacking a finality column as confirmed.",
    )
    parser.add_argument("--write", action="store_true", help="Write after validation; default is dry-run.")
    args = parser.parse_args()

    path = args.input.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if args.finality == "confirmed" and any(marker in path.name.lower() for marker in PROVISIONAL_MARKERS):
        raise ValueError(
            "the filename looks provisional; import it with --finality provisional or use a confirmed source"
        )
    bars = prepare(
        load_input(path),
        finality=args.finality,
        source=args.source,
        attest_confirmed=args.attest_completed_sessions,
    )
    print(f"source: {path}")
    print(f"sha256: {source_digest(path)}")
    print(f"rows: {len(bars)}")
    print(f"symbols: {bars['symbol'].nunique() if not bars.empty else 0}")
    print(f"date range: {bars['date'].min() if not bars.empty else None} -> {bars['date'].max() if not bars.empty else None}")
    print(f"finality: {args.finality}")
    if not args.write:
        print("dry-run: validated only; pass --write to import")
        return

    data_home = args.data_home or (Path(os.environ["QUANT_DATA_HOME"]) if os.getenv("QUANT_DATA_HOME") else None)
    store = MarketDataStore(root=data_home)
    store.write_bars(bars, finality=args.finality)
    print(f"imported into: {store.root}")


if __name__ == "__main__":
    main()
