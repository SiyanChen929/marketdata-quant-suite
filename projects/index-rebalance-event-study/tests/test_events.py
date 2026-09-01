from pathlib import Path

import pandas as pd
import pytest

from index_rebalance_event_study.events import (
    EventSchemaError,
    ParsedReview,
    build_primary_event_frame,
    load_event_csv,
    name_similarity,
    normalize_company_name,
    validate_event_frame,
)


SAMPLE = Path(__file__).parents[1] / "data" / "sample" / "synthetic_events.csv"


def test_normalize_company_name_removes_corporate_suffixes() -> None:
    assert normalize_company_name("Northstar Synthetic Holdings, Inc. Class A") == "NORTHSTAR SYNTHETIC"


def test_name_similarity_handles_fictional_abbreviation() -> None:
    assert name_similarity("BLUE MESA FICTIONAL INDS", "Blue Mesa Fictional Industries, Inc.") > 0.8


def test_primary_review_classifies_secondary_migration() -> None:
    effective = pd.Timestamp("2035-01-12")
    reviews = [
        ParsedReview(
            announcement_date=pd.Timestamp("2035-01-08"),
            effective_close_date=effective,
            segment="primary",
            source_id="synthetic_primary",
            additions=("Northstar Synthetic Labs",),
            deletions=(),
        ),
        ParsedReview(
            announcement_date=pd.Timestamp("2035-01-08"),
            effective_close_date=effective,
            segment="secondary",
            source_id="synthetic_secondary",
            additions=(),
            deletions=("Northstar Synthetic Labs Class A",),
        ),
    ]
    events = build_primary_event_frame(reviews)
    assert events.loc[0, "change_type"] == "up_from_secondary"


def test_included_fixture_is_explicitly_synthetic() -> None:
    events = load_event_csv(SAMPLE)
    assert events["symbol"].str.startswith("ZZZ").all()
    assert events["is_synthetic"].astype(bool).all()


def test_announcement_after_effective_date_fails_closed() -> None:
    events = pd.read_csv(SAMPLE).head(1)
    events["announcement_date"] = "2035-02-01"
    with pytest.raises(EventSchemaError, match="announcement_date"):
        validate_event_frame(events)
