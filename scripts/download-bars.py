#!/usr/bin/env python3
"""Download one canonical MarketData daily panel into the shared external store."""

from __future__ import annotations

import argparse
from pathlib import Path

from quant_marketdata import MarketDataClient, MarketDataStore


def parse_symbols(values: list[str], symbols_file: Path | None) -> list[str]:
    symbols: list[str] = []
    for value in values:
        symbols.extend(part.strip() for part in value.split(","))
    if symbols_file:
        symbols.extend(
            line.strip()
            for line in symbols_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    normalized = sorted({symbol.upper() for symbol in symbols if symbol})
    if not normalized:
        raise ValueError("provide --symbols or --symbols-file")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", action="append", default=[], help="Comma-separated symbols; repeatable.")
    parser.add_argument("--symbols-file", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--finality", choices=("confirmed", "provisional"), default="confirmed")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    symbols = parse_symbols(args.symbols, args.symbols_file)
    store = MarketDataStore()
    client = MarketDataClient(store=store)
    bars = client.get_bulk_daily_bars(
        symbols,
        start=args.start,
        end=args.end,
        finality=args.finality,
        refresh=args.refresh,
    )
    print(f"provider: MarketData")
    print(f"store: {store.root}")
    print(f"finality: {args.finality}")
    print(f"rows: {len(bars)}")
    print(f"symbols: {bars['symbol'].nunique() if not bars.empty else 0}/{len(symbols)}")
    print(f"date range: {bars['date'].min() if not bars.empty else None} -> {bars['date'].max() if not bars.empty else None}")


if __name__ == "__main__":
    main()
