"""Provider-neutral index rebalance event-study toolkit."""

from .backtest import aggregate_date_portfolio, build_daily_event_ledger, performance_metrics
from .events import load_event_csv, validate_event_frame
from .marketdata import ConfirmedDailyBars

__all__ = [
    "ConfirmedDailyBars",
    "aggregate_date_portfolio",
    "build_daily_event_ledger",
    "load_event_csv",
    "performance_metrics",
    "validate_event_frame",
]

__version__ = "0.1.0"
