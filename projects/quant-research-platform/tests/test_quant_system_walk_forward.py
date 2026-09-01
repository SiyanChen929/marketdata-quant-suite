from __future__ import annotations

import pandas as pd

from quant_system.optimization.walk_forward import make_walk_forward_splits


def test_walk_forward_split_order() -> None:
    dates = pd.Series(pd.bdate_range("2018-01-01", "2022-12-31"))
    splits = make_walk_forward_splits(dates, train_years=2, test_months=6)
    assert splits
    first = splits[0]
    assert first.train_start < first.train_end < first.test_start < first.test_end
