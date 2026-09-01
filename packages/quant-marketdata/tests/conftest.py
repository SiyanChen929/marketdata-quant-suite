from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd
import pytest


class FakeResponse:
    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


class FakeSession:
    def __init__(self, responses: Iterable[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append({"url": url, **kwargs})
        if not self.responses:
            raise AssertionError("unexpected network call")
        return self.responses.pop(0)


def candle_payload(*dates: str) -> dict[str, Any]:
    timestamps = [
        int(pd.Timestamp(f"{date} 05:00:00Z").timestamp()) for date in dates
    ]
    count = len(timestamps)
    return {
        "s": "ok",
        "t": timestamps,
        "o": [100.0 + index for index in range(count)],
        "h": [102.0 + index for index in range(count)],
        "l": [99.0 + index for index in range(count)],
        "c": [101.0 + index for index in range(count)],
        "v": [1_000_000 + index for index in range(count)],
    }


@pytest.fixture
def canonical_bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-12-29", "2024-01-02"]),
            "symbol": ["AAPL", "AAPL"],
            "open": [99.0, 100.0],
            "high": [102.0, 103.0],
            "low": [98.0, 99.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
            "source": ["synthetic", "synthetic"],
            "finality": ["confirmed", "confirmed"],
        }
    )
