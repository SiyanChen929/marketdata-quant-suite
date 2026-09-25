from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from quant_marketdata import CANONICAL_COLUMNS, MarketDataStore, normalize_bars

from marketdata_agent import (
    AsOfClock,
    FrameBarSource,
    LookaheadViolation,
    PointInTimeBars,
    ProvisionalDataError,
    SourceContractError,
    StoreBarSource,
    UnknownSymbolError,
    hash_bars,
    synthetic_bars,
)
from marketdata_agent.synthetic import SyntheticPanelSpec, synthetic_symbols

from conftest import AS_OF, LATE_LISTING, LATE_SYMBOL


def test_synthetic_panel_is_canonical_confirmed_and_deterministic():
    first = synthetic_bars(3, start="2022-01-03", end="2022-12-30", seed=11)
    second = synthetic_bars(3, start="2022-01-03", end="2022-12-30", seed=11)
    other = synthetic_bars(3, start="2022-01-03", end="2022-12-30", seed=12)
    pd.testing.assert_frame_equal(first, second)
    assert hash_bars(first) == hash_bars(second) != hash_bars(other)
    assert tuple(first.columns) == CANONICAL_COLUMNS
    pd.testing.assert_frame_equal(normalize_bars(first), first)
    assert set(first["finality"]) == {"confirmed"}
    assert set(first["source"]) == {"synthetic"}
    assert first["date"].dt.dayofweek.max() <= 4
    assert (first["high"] >= first[["open", "close"]].max(axis=1)).all()
    assert (first["low"] <= first[["open", "close"]].min(axis=1)).all()
    assert synthetic_symbols(3) == ["SYN01", "SYN02", "SYN03"]


def test_synthetic_listings_drop_pre_listing_rows(panel):
    late = panel.loc[panel["symbol"] == LATE_SYMBOL, "date"]
    assert late.min() >= pd.Timestamp(LATE_LISTING)
    spec = SyntheticPanelSpec(symbols=("A", "B"), listings=(("B", "2022-01-03"),))
    assert spec.to_dict()["listings"] == {"B": "2022-01-03"}
    with pytest.raises(ValueError):
        synthetic_bars(["A"], listings={"Z": "2022-01-03"})


def test_frame_source_rejects_provisional_and_unlabelled_rows(panel):
    provisional = panel.head(10).copy()
    provisional.loc[provisional.index[-1], "finality"] = "provisional"
    with pytest.raises(ProvisionalDataError):
        FrameBarSource(provisional)
    with pytest.raises(SourceContractError):
        FrameBarSource(panel.drop(columns="finality").head(10))


def test_frame_source_filters_inclusively(source):
    rows = source.read_bars(["syn01"], date(2023, 6, 1), date(2023, 6, 30))
    assert set(rows["symbol"]) == {"SYN01"}
    assert rows["date"].min() == pd.Timestamp("2023-06-01")
    assert rows["date"].max() == pd.Timestamp("2023-06-30")
    assert source.list_symbols() == ["SYN01", "SYN02", "SYN03", "SYN04", "SYN05", "SYN06"]
    assert source.snapshot_id().startswith("frame:")


def _store_with_provisional_tail(tmp_path, panel) -> MarketDataStore:
    store = MarketDataStore(tmp_path / "data-home")
    confirmed = panel.loc[panel["symbol"].isin(["SYN01", "SYN02"]) & (panel["date"] <= pd.Timestamp("2023-06-30"))]
    provisional = panel.loc[
        panel["symbol"].isin(["SYN01", "SYN02"])
        & (panel["date"] > pd.Timestamp("2023-06-30"))
        & (panel["date"] <= pd.Timestamp("2023-07-07"))
    ].assign(finality="provisional")
    store.write_bars(confirmed, finality="confirmed")
    store.write_bars(provisional, finality="provisional")
    return store


def test_store_source_reads_only_the_confirmed_lake(tmp_path, panel):
    store = _store_with_provisional_tail(tmp_path, panel)
    assert not store.read_bars(["SYN01"], finality="provisional").empty  # the staging area really has rows
    source = StoreBarSource(store)
    rows = source.read_bars(["SYN01", "SYN02"], None, date(2023, 12, 29))
    assert set(rows["finality"]) == {"confirmed"}
    assert rows["date"].max() == pd.Timestamp("2023-06-30")
    assert source.list_symbols() == ["SYN01", "SYN02"]
    assert source.snapshot_id().startswith("store-manifest:")
    restricted = StoreBarSource(store, universe=["syn02"])
    assert restricted.list_symbols() == ["SYN02"]
    assert set(restricted.read_bars(["SYN01", "SYN02"], None, None)["symbol"]) == {"SYN02"}


def test_store_source_never_requests_provisional_finality(tmp_path, panel):
    store = _store_with_provisional_tail(tmp_path, panel)
    calls: list[str] = []
    original = store.read_bars

    def spy(*args, **kwargs):
        calls.append(kwargs.get("finality"))
        return original(*args, **kwargs)

    store.read_bars = spy  # type: ignore[method-assign]
    source = StoreBarSource(store)
    view = PointInTimeBars(source, AsOfClock(AS_OF))
    view.available_symbols()
    view.read(["SYN01"], date(2023, 1, 3), date(2023, 6, 30))
    assert calls and set(calls) == {"confirmed"}


def test_view_refuses_dates_after_as_of(source):
    view = PointInTimeBars(source, AsOfClock(AS_OF))
    with pytest.raises(LookaheadViolation):
        view.read(["SYN01"], date(2023, 6, 1), date(2023, 7, 3))
    with pytest.raises(LookaheadViolation):
        view.read(["SYN01"], date(2023, 7, 3), None)
    capped = view.read(["SYN01"], date(2023, 6, 1), None)
    assert capped["date"].max() == pd.Timestamp(AS_OF)


class _RogueSource:
    """Returns whatever frame it was given, ignoring the requested range."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def read_bars(self, symbols, start, end):
        return self.frame

    def list_symbols(self):
        return sorted(self.frame["symbol"].unique())


def test_view_rejects_sources_that_leak_future_or_provisional_rows(panel):
    clock = AsOfClock(AS_OF)
    leaky = PointInTimeBars(_RogueSource(panel.loc[panel["symbol"] == "SYN01"]), clock)
    with pytest.raises(SourceContractError, match="after the requested end"):
        leaky.read(["SYN01"], date(2023, 6, 1), date(2023, 6, 30))
    provisional = panel.loc[(panel["symbol"] == "SYN01") & (panel["date"] <= pd.Timestamp(AS_OF))].assign(
        finality="provisional"
    )
    with pytest.raises(ProvisionalDataError):
        PointInTimeBars(_RogueSource(provisional), clock).read(["SYN01"], None, None)
    extra = panel.loc[panel["date"] <= pd.Timestamp(AS_OF)]
    with pytest.raises(SourceContractError, match="unrequested symbols"):
        PointInTimeBars(_RogueSource(extra), clock).read(["SYN01"], None, None)
    with pytest.raises(TypeError):
        PointInTimeBars(object(), clock)  # type: ignore[arg-type]


def test_view_universe_and_calendar_exclude_the_future(source):
    early = PointInTimeBars(source, AsOfClock(date(2023, 2, 15)))
    assert LATE_SYMBOL not in early.available_symbols()
    with pytest.raises(UnknownSymbolError):
        early.require_symbols([LATE_SYMBOL])
    view = PointInTimeBars(source, AsOfClock(date(2023, 7, 2)))  # Sunday
    assert LATE_SYMBOL in view.available_symbols()
    assert view.sessions()[-1] == date(2023, 6, 30)
    assert view.require_symbols(["syn01", "SYN01"]) == ["SYN01"]
