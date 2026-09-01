"""Walk-forward split utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    """One train/test window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_walk_forward_splits(
    dates: pd.Series,
    train_years: int = 3,
    test_months: int = 6,
    expanding: bool = False,
) -> list[WalkForwardSplit]:
    """Create rolling or expanding train/test windows."""

    unique = pd.Series(pd.to_datetime(dates).dropna().sort_values().unique())
    if unique.empty:
        return []
    start = unique.iloc[0]
    end = unique.iloc[-1]
    splits: list[WalkForwardSplit] = []
    train_start = start
    train_end = start + pd.DateOffset(years=train_years)
    while train_end < end:
        test_start = train_end + pd.Timedelta(days=1)
        test_end = min(test_start + pd.DateOffset(months=test_months) - pd.Timedelta(days=1), end)
        actual_train_start = start if expanding else train_start
        splits.append(WalkForwardSplit(actual_train_start, train_end, test_start, test_end))
        train_end = test_end
        if not expanding:
            train_start = train_end - pd.DateOffset(years=train_years)
    return splits
