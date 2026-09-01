"""Earnings calendar alignment."""

from __future__ import annotations

import pandas as pd


def align_earnings_to_dates(prices: pd.DataFrame, earnings: pd.DataFrame, post_event_window_days: int = 2) -> pd.DataFrame:
    """Compute days to next known earnings date for each price date."""

    base = prices[["date", "symbol"]].drop_duplicates().copy()
    base["date"] = pd.to_datetime(base["date"]).dt.normalize()
    if earnings.empty:
        base["days_to_earnings"] = pd.NA
        base["event_risk_score"] = 0.0
        return base
    events = earnings.copy()
    events["event_date"] = pd.to_datetime(events["event_date"]).dt.normalize()
    events["known_date"] = pd.to_datetime(events.get("known_date", events["event_date"])).dt.normalize()
    rows = []
    for symbol, dates in base.groupby("symbol"):
        ev = events[events["symbol"] == symbol].sort_values("event_date")
        for _, row in dates.iterrows():
            known = ev[(ev["known_date"] <= row["date"]) & (ev["event_date"] >= row["date"])]
            if not known.empty:
                nxt = known.iloc[0]
                rows.append(_event_row(row["date"], symbol, nxt, int((nxt["event_date"] - row["date"]).days)))
                continue
            recent = ev[
                (ev["known_date"] <= row["date"])
                & (ev["event_date"] < row["date"])
                & ((row["date"] - ev["event_date"]).dt.days <= int(post_event_window_days))
            ].sort_values("event_date", ascending=False)
            if recent.empty:
                rows.append({"date": row["date"], "symbol": symbol, "days_to_earnings": pd.NA, "event_risk_score": 0.0})
            else:
                last = recent.iloc[0]
                rows.append(_event_row(row["date"], symbol, last, -int((row["date"] - last["event_date"]).days)))
    return pd.DataFrame(rows)


def _event_row(date: pd.Timestamp, symbol: str, event: pd.Series, days_to_earnings: int) -> dict:
    row = {
        "date": date,
        "symbol": symbol,
        "days_to_earnings": days_to_earnings,
        "next_earnings_date": event.get("event_date"),
        "event_known_date": event.get("known_date"),
        "event_risk_score": float(event.get("event_risk_score", 50.0)),
    }
    for column in (
        "expected_move",
        "report_time",
        "surprise_eps_pct",
        "historical_earnings_gap_volatility",
        "recent_guidance_change",
        "dilution_event",
        "secondary_offering",
        "regulatory_event",
        "FDA_event",
        "lockup_expiration",
        "debt_maturity",
        "macro_event_exposure",
    ):
        if column in event.index:
            row[column] = event.get(column)
    return row
