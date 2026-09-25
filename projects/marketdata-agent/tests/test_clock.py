from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime

import pandas as pd
import pytest

from marketdata_agent import AsOfClock, Code, InsufficientDataError, LookaheadViolation, ToolInputError, parse_date


def test_parse_date_accepts_iso_strings_and_date_objects():
    assert parse_date("2023-06-30") == date(2023, 6, 30)
    assert parse_date(date(2023, 6, 30)) == date(2023, 6, 30)
    assert parse_date(pd.Timestamp("2023-06-30")) == date(2023, 6, 30)
    assert parse_date(datetime(2023, 6, 30)) == date(2023, 6, 30)


@pytest.mark.parametrize(
    "value",
    ["2023/06/30", "2023-06-30T10:00:00", "2023-02-30", "30-06-2023", "", 20230630, None, pd.Timestamp("2023-06-30 15:59")],
)
def test_parse_date_rejects_malformed_or_intraday_values(value):
    with pytest.raises(ToolInputError):
        parse_date(value)


def test_clock_is_immutable_and_coerces_strings():
    clock = AsOfClock("2023-06-30")
    assert clock.as_of == date(2023, 6, 30)
    with pytest.raises(FrozenInstanceError):
        clock.as_of = date(2024, 1, 1)  # type: ignore[misc]


def test_dates_after_as_of_raise_an_observable_violation():
    clock = AsOfClock(date(2023, 6, 30))
    assert clock.check(date(2023, 6, 30), field="end") == date(2023, 6, 30)
    with pytest.raises(LookaheadViolation) as caught:
        clock.validate("2023-07-03", field="end")
    violation = caught.value
    assert violation.code == Code.LOOKAHEAD
    assert violation.field == "end"
    assert violation.requested == date(2023, 7, 3)
    assert violation.as_of == date(2023, 6, 30)


def test_latest_resolves_to_last_confirmed_session_on_or_before_as_of():
    sessions = [date(2023, 6, 28), date(2023, 6, 29), date(2023, 6, 30), date(2023, 7, 3)]
    assert AsOfClock(date(2023, 6, 30)).resolve("latest", field="end", sessions=sessions) == date(2023, 6, 30)
    # Weekend as-of date: "latest" is the prior Friday, never the next Monday.
    assert AsOfClock(date(2023, 7, 2)).resolve("LATEST", field="end", sessions=sessions) == date(2023, 6, 30)


def test_latest_is_not_accepted_for_start_dates_and_needs_sessions():
    clock = AsOfClock(date(2023, 6, 30))
    with pytest.raises(ToolInputError):
        clock.validate("latest", field="start")
    with pytest.raises(ToolInputError):
        clock.resolve("latest", field="end", sessions=None)
    with pytest.raises(InsufficientDataError):
        clock.resolve_latest([date(2023, 7, 3)])
