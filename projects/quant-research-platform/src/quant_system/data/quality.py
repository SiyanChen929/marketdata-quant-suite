"""Data reconciliation helpers for imperfect vendor feeds."""

from __future__ import annotations

import pandas as pd


def reconcile_daily_ohlcv_with_intraday(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    *,
    close_mismatch_threshold: float = 0.01,
    max_dates: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use completed intraday bars to repair stale latest daily closes.

    Some vendors update intraday bars before their end-of-day candle is fully
    corrected. If the daily close differs materially from the final intraday
    close for the same symbol/date, the close and adjusted close are replaced
    with the intraday value. Only the latest ``max_dates`` overlapping sessions
    are considered so historical runs do not become slow or accidentally depend
    on a full intraday archive.
    """

    if daily.empty or intraday.empty:
        return daily.copy(), _empty_report()
    if not {"symbol", "date", "close"}.issubset(daily.columns):
        return daily.copy(), _empty_report()
    if not {"symbol", "date", "datetime", "close"}.issubset(intraday.columns):
        return daily.copy(), _empty_report()

    daily_out = daily.copy()
    daily_out["symbol"] = daily_out["symbol"].astype(str).str.upper()
    daily_out["date"] = pd.to_datetime(daily_out["date"], format="mixed", errors="coerce").dt.normalize()

    minute = intraday.copy()
    minute["symbol"] = minute["symbol"].astype(str).str.upper()
    minute["date"] = pd.to_datetime(minute["date"], format="mixed", errors="coerce").dt.normalize()
    minute["datetime"] = pd.to_datetime(minute["datetime"], format="mixed", errors="coerce")
    minute["close"] = pd.to_numeric(minute["close"], errors="coerce")
    minute = minute.dropna(subset=["symbol", "date", "datetime", "close"])
    if minute.empty:
        return daily_out, _empty_report()

    latest_dates = sorted(minute["date"].dropna().unique())[-max(1, int(max_dates)) :]
    minute = minute[minute["date"].isin(latest_dates)].copy()
    minute = minute.sort_values(["symbol", "date", "datetime"])
    last_bars = minute.groupby(["symbol", "date"], as_index=False).tail(1)
    last_columns = ["symbol", "date", "datetime", "close"]
    if "source" in last_bars:
        last_columns.append("source")
    last_bars = last_bars[last_columns].rename(
        columns={"datetime": "intraday_last_datetime", "close": "intraday_close", "source": "intraday_source"}
    )

    merged = daily_out.reset_index(names="_daily_index").merge(last_bars, on=["symbol", "date"], how="inner")
    if merged.empty:
        return daily_out, _empty_report()
    merged["daily_close"] = pd.to_numeric(merged["close"], errors="coerce")
    merged["close_mismatch_pct"] = (merged["intraday_close"] / merged["daily_close"].replace(0, pd.NA) - 1.0).abs()
    bad = merged[
        merged["daily_close"].notna()
        & merged["intraday_close"].notna()
        & merged["close_mismatch_pct"].gt(float(close_mismatch_threshold))
    ].copy()
    if bad.empty:
        return daily_out, _empty_report()

    for _, row in bad.iterrows():
        idx = row["_daily_index"]
        daily_out.loc[idx, "close"] = float(row["intraday_close"])
        if "adj_close" in daily_out.columns:
            daily_out.loc[idx, "adj_close"] = float(row["intraday_close"])
        daily_out.loc[idx, "source"] = row.get("intraday_source", "intraday-derived")
        daily_out.loc[idx, "finality"] = "provisional"

    report = bad[
        [
            "symbol",
            "date",
            "daily_close",
            "intraday_close",
            "close_mismatch_pct",
            "intraday_last_datetime",
        ]
    ].copy()
    report["reconciliation_reason"] = "daily_close_replaced_by_intraday_last_bar"
    report["replacement_source"] = bad.get("intraday_source", "intraday-derived")
    report["replacement_finality"] = "provisional"
    return daily_out, report.sort_values(["date", "symbol"]).reset_index(drop=True)


def _empty_report() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "symbol",
            "date",
            "daily_close",
            "intraday_close",
            "close_mismatch_pct",
            "intraday_last_datetime",
            "reconciliation_reason",
            "replacement_source",
            "replacement_finality",
        ]
    )
