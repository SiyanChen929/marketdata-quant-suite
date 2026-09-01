"""Command-line entry point for the equity-pairs research pipeline."""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

from .config import PipelineConfig, load_config, validate_config
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen current S&P 500 sector pairs, run a held-out cost-aware "
            "backtest, and build the complete research package."
        )
    )
    parser.add_argument(
        "--as-of",
        default=str(date.today() - timedelta(days=1)),
        help="Last included market date (YYYY-MM-DD); defaults to yesterday.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.json"),
        help="JSON configuration path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory; defaults to outputs/pairs_research_<as-of>.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="Override cointegration-screen worker count.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Refresh constituent, confirmed MarketData, and FRED exact-request caches.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute cointegration screens even if processed caches exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.workers is not None:
        config = PipelineConfig(
            research=replace(config.research, workers=args.workers),
            strategy=config.strategy,
            macro=config.macro,
        )
        validate_config(config)
    output = args.output_dir or Path("outputs") / f"pairs_research_{args.as_of}"
    report = run_pipeline(
        as_of=args.as_of,
        output_dir=output,
        config=config,
        refresh_data=args.refresh_data,
        reuse_computations=not args.recompute,
    )
    print(report)


if __name__ == "__main__":
    main()
