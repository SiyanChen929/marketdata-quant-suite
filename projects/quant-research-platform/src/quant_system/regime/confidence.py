"""Regime and limited-history confidence adjustments."""

from __future__ import annotations


def limited_history_multiplier(history_days: int, min_history_days: int = 252) -> float:
    """Discount new listings/restructured stocks with limited history."""

    if history_days >= min_history_days:
        return 1.0
    if history_days >= 120:
        return 0.5
    return 0.3
