#!/usr/bin/env python3
"""Run the event-study plumbing with fictional symbols and in-memory bars."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from index_rebalance_event_study.backtest import (
    aggregate_date_portfolio,
    build_daily_event_ledger,
    performance_metrics,
)
from index_rebalance_event_study.events import load_event_csv


ROOT = Path(__file__).resolve().parents[1]


def synthetic_confirmed_bars(events: pd.DataFrame) -> pd.DataFrame:
    """Construct deterministic canonical candles; no provider is contacted."""

    rows: list[dict[str, object]] = []
    for index, event in enumerate(events.itertuples(index=False)):
        announcement = pd.Timestamp(event.announcement_date)
        effective = pd.Timestamp(event.effective_close_date)
        base = 80.0 + 10.0 * index
        first_close = base * 1.001
        favorable = event.source_id == "synthetic_review_001"
        signed_move = 0.02 if favorable else -0.01
        underlying_move = signed_move if event.action == "add" else -signed_move
        effective_open = base * (1.0 + underlying_move / 2.0)
        effective_close = base * (1.0 + underlying_move)
        next_open = effective_close * (1.0 - underlying_move / 4.0)
        for date, open_, close in (
            (announcement + pd.Timedelta(days=1), base, first_close),
            (effective, effective_open, effective_close),
            (effective + pd.Timedelta(days=3), next_open, next_open * 1.001),
        ):
            rows.append(
                {
                    "date": date,
                    "symbol": event.symbol,
                    "open": open_,
                    "high": max(open_, close) * 1.005,
                    "low": min(open_, close) * 0.995,
                    "close": close,
                    "volume": 100_000 + index,
                    "source": "synthetic_demo",
                    "finality": "confirmed",
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    events = load_event_csv(ROOT / "data" / "sample" / "synthetic_events.csv")
    ledger, failures = build_daily_event_ledger(events, synthetic_confirmed_bars(events))
    portfolio = aggregate_date_portfolio(
        ledger,
        "announcement_to_effective_return_gross",
        cost_bps=5.0,
    )
    metrics = performance_metrics(portfolio)
    print(
        json.dumps(
            {
                "synthetic": True,
                "network_used": False,
                "event_rows": len(events),
                "ledger_rows": len(ledger),
                "failure_rows": len(failures),
                "independent_event_dates": metrics.get("event_dates", 0),
                "synthetic_total_return": metrics.get("total_return"),
                "warning": "Synthetic plumbing test only; no investment evidence.",
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
