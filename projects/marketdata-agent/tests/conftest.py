from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from marketdata_agent import AuditLog, FrameBarSource, ToolRuntime, synthetic_bars


AS_OF = date(2023, 6, 30)  # a Friday
LATE_SYMBOL = "SYN06"
LATE_LISTING = "2023-03-01"


@pytest.fixture(scope="session")
def panel() -> pd.DataFrame:
    """Six synthetic symbols; SYN06 lists on 2023-03-01."""

    return synthetic_bars(6, start="2021-01-04", end="2023-12-29", seed=20240601, listings={LATE_SYMBOL: LATE_LISTING})


@pytest.fixture
def source(panel: pd.DataFrame) -> FrameBarSource:
    return FrameBarSource(panel)


@pytest.fixture
def runtime(source: FrameBarSource) -> ToolRuntime:
    return ToolRuntime.for_source(source, AS_OF)


@pytest.fixture
def audit(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "episode.audit.jsonl", fsync=False)


@pytest.fixture
def audited_runtime(source: FrameBarSource, audit: AuditLog) -> ToolRuntime:
    return ToolRuntime.for_source(source, AS_OF, audit=audit, episode_id="ep-test")


def closes(panel: pd.DataFrame, symbol: str, start: str | None = None, end: str | None = None) -> pd.Series:
    """Independent close-series selection straight from the panel (no package code)."""

    rows = panel.loc[panel["symbol"] == symbol]
    if start is not None:
        rows = rows.loc[rows["date"] >= pd.Timestamp(start)]
    if end is not None:
        rows = rows.loc[rows["date"] <= pd.Timestamp(end)]
    return rows.set_index("date")["close"].sort_index()


def brute_force_drawdown(values: np.ndarray) -> float:
    """O(n^2) reference: worst later/earlier price ratio minus one (0 if never down)."""

    worst = 0.0
    for i in range(len(values)):
        later = values[i:]
        worst = min(worst, float(later.min() / values[i] - 1.0))
    return worst
