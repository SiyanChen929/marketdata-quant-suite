"""Position sizing helpers."""

from __future__ import annotations


def shares_for_target_weight(equity: float, target_weight: float, price: float) -> float:
    """Convert target weight to share quantity."""

    if price <= 0:
        return 0.0
    return equity * target_weight / price
