from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from equity_pairs.macro import derive_macro_variables


def test_monthly_derivations_ignore_daily_union_rows():
    dates = pd.date_range("2023-01-01", "2024-02-01", freq="D")
    frame = pd.DataFrame(index=dates, columns=["DFF", "CPIAUCSL", "UNRATE"], dtype=float)
    frame["DFF"] = 5.0
    monthly_dates = pd.date_range("2023-01-01", "2024-02-01", freq="MS")
    frame.loc[monthly_dates, "CPIAUCSL"] = np.arange(len(monthly_dates), dtype=float) + 100.0
    frame.loc[monthly_dates, "UNRATE"] = np.arange(len(monthly_dates), dtype=float) / 10.0 + 4.0

    result = derive_macro_variables(frame)

    expected_cpi = (112.0 / 100.0 - 1.0) * 100.0
    assert result.loc[pd.Timestamp("2024-01-01"), "cpi_yoy"] == expected_cpi
    assert result.loc[pd.Timestamp("2024-01-01"), "unemployment_12m_change"] == pytest.approx(1.2)
    assert result["cpi_yoy"].notna().sum() == 2
