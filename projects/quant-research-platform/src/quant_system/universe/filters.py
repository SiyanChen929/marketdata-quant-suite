"""Tradable universe filters."""

from __future__ import annotations

import pandas as pd

from quant_system.config import UniverseConfig


def apply_price_quality_adjustments(prices: pd.DataFrame, config: UniverseConfig) -> pd.DataFrame:
    """Quarantine stale/corporate-action-contaminated history without dropping symbols.

    Large unadjusted corporate-action jumps can poison rolling signals, ATR stops,
    and position sizing. If a symbol has enough clean history after the last
    suspicious jump/stale run, keep only the clean segment and mark it for a
    position-size haircut instead of removing the ticker entirely.
    """

    if prices.empty:
        return prices.copy()
    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    adjusted_frames: list[pd.DataFrame] = []
    for symbol, frame in prices.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        clean = frame.copy()
        clean["quality_adjusted"] = False
        clean["quality_adjustment_reason"] = ""
        clean["quality_position_multiplier"] = 1.0
        clean["quality_history_start"] = pd.NaT
        close = pd.to_numeric(clean["adj_close"] if "adj_close" in clean else clean["close"], errors="coerce")
        abs_returns = close.pct_change().abs()
        jump_dates = clean.loc[abs_returns > config.max_unhandled_corporate_action_return, "date"]
        stale_end = _last_stale_run_end(clean, config.max_stale_price_days)
        cut_date = pd.NaT
        reasons: list[str] = []
        if not jump_dates.empty:
            cut_date = pd.Timestamp(jump_dates.max())
            reasons.append("post_corporate_action_quarantine")
        if stale_end is not None and (pd.isna(cut_date) or stale_end > cut_date):
            cut_date = stale_end
            reasons.append("post_stale_price_quarantine")
        if pd.notna(cut_date):
            trimmed = clean[clean["date"] > cut_date].copy()
            if len(trimmed) >= config.min_quality_adjusted_history_days:
                trimmed["quality_adjusted"] = True
                trimmed["quality_adjustment_reason"] = ",".join(reasons)
                trimmed["quality_position_multiplier"] = float(config.quality_adjusted_position_multiplier)
                trimmed["quality_history_start"] = trimmed["date"].min()
                clean = trimmed
        adjusted_frames.append(clean)
    return pd.concat(adjusted_frames, ignore_index=True) if adjusted_frames else prices.copy()


def build_tradable_universe(prices: pd.DataFrame, config: UniverseConfig) -> pd.DataFrame:
    """Return the latest as-of tradable universe snapshot."""

    if prices.empty:
        return pd.DataFrame(columns=["symbol", "tradable", "limited_history_flag", "filter_reason"])
    panel = build_asof_tradable_universe(prices, config)
    if panel.empty:
        return pd.DataFrame(columns=["symbol", "tradable", "limited_history_flag", "filter_reason"])
    return panel.sort_values(["symbol", "date"]).groupby("symbol", as_index=False).tail(1).reset_index(drop=True)


def build_asof_tradable_universe(prices: pd.DataFrame, config: UniverseConfig) -> pd.DataFrame:
    """Build a daily point-in-time tradable/candidate universe mask.

    Each row is evaluated with only information available by that row's close.
    This avoids using the sample end date's liquidity, listing age, stale-price
    status, or post-hoc corporate-action cleanups for earlier history.
    """

    if prices.empty:
        return pd.DataFrame(columns=["date", "symbol", "tradable", "limited_history_flag", "filter_reason"])
    data = prices.sort_values(["symbol", "date"]).copy()
    data["date"] = pd.to_datetime(data["date"]).dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data["volume"] = pd.to_numeric(data["volume"], errors="coerce")
    data["dollar_volume"] = data["close"] * data["volume"]
    grouped = data.groupby("symbol", group_keys=False)
    data["history_days"] = grouped.cumcount() + 1
    data["avg_volume_20d"] = grouped["volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    data["avg_dollar_volume_20d"] = grouped["dollar_volume"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    legacy_quality_cols = [
        "quality_adjusted",
        "quality_adjustment_reason",
        "quality_position_multiplier",
        "quality_history_start",
    ]
    legacy_quality = data[[col for col in legacy_quality_cols if col in data.columns]].copy()
    data = data.drop(columns=[col for col in legacy_quality_cols if col in data.columns])
    quality_input_cols = [col for col in ["date", "adj_close", "close"] if col in data.columns]
    quality = (
        data.groupby("symbol", group_keys=False)[quality_input_cols]
        .apply(lambda frame: _quality_asof_for_symbol(frame, config))
        .reset_index(level=0, drop=True)
    )
    if not legacy_quality.empty and "quality_adjusted" in legacy_quality:
        legacy_adjusted = legacy_quality["quality_adjusted"].fillna(False).astype(bool)
        quality["quality_adjusted"] = quality["quality_adjusted"].astype(bool) | legacy_adjusted
        if "quality_adjustment_reason" in legacy_quality:
            quality["quality_adjustment_reason"] = legacy_quality["quality_adjustment_reason"].fillna(quality["quality_adjustment_reason"])
        if "quality_position_multiplier" in legacy_quality:
            quality.loc[legacy_adjusted, "quality_position_multiplier"] = legacy_quality.loc[legacy_adjusted, "quality_position_multiplier"].fillna(
                config.quality_adjusted_position_multiplier
            )
        if "quality_history_start" in legacy_quality:
            quality.loc[legacy_adjusted, "quality_history_start"] = legacy_quality.loc[legacy_adjusted, "quality_history_start"]
    data = pd.concat([data, quality], axis=1)
    rows = []
    for _, row in data.iterrows():
        reasons = _asof_filter_reasons(row, config)
        limited_history = bool(row["history_days"] < config.min_history_days)
        tradable = (
            len(reasons) == 0
            or reasons == ["quality_adjusted_history"]
            or (limited_history and config.allow_speculative_bucket and reasons == ["limited_history"])
        )
        rows.append(
            {
                "date": row["date"],
                "symbol": row["symbol"],
                "tradable": bool(tradable),
                "limited_history_flag": limited_history,
                "avg_dollar_volume_20d": row["avg_dollar_volume_20d"],
                "avg_volume_20d": row["avg_volume_20d"],
                "last_price": row["close"],
                "history_days": int(row["history_days"]),
                "max_stale_price_run": row["max_stale_price_run"],
                "max_abs_daily_return": row["max_abs_daily_return"],
                "unhandled_corporate_action_count": row["unhandled_corporate_action_count"],
                "quality_adjusted": bool(row["quality_adjusted"]),
                "quality_adjustment_reason": row["quality_adjustment_reason"],
                "quality_position_multiplier": float(row["quality_position_multiplier"]),
                "quality_history_start": row["quality_history_start"],
                "filter_reason": ",".join(reasons) if reasons else "passed",
            }
        )
    out = pd.DataFrame(rows)
    out["candidate_reason"] = out.apply(_candidate_reason, axis=1)
    return out


def filter_prices_to_tradable(prices: pd.DataFrame, tradable: pd.DataFrame) -> pd.DataFrame:
    """Keep prices for tradable symbols or as-of tradable rows."""

    if {"date", "symbol", "tradable"}.issubset(tradable.columns):
        mask = tradable.loc[tradable["tradable"], ["date", "symbol"]].copy()
        mask["date"] = pd.to_datetime(mask["date"]).dt.normalize()
        out = prices.copy()
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out.merge(mask.drop_duplicates(), on=["date", "symbol"], how="inner")
    symbols = set(tradable.loc[tradable["tradable"], "symbol"])
    return prices[prices["symbol"].isin(symbols)].copy()


def _quality_asof_for_symbol(frame: pd.DataFrame, config: UniverseConfig) -> pd.DataFrame:
    ordered = frame.sort_values("date").copy()
    close = pd.to_numeric(ordered["adj_close"] if "adj_close" in ordered else ordered["close"], errors="coerce")
    abs_returns = close.pct_change().abs()
    unchanged = close.eq(close.shift())
    groups = unchanged.ne(unchanged.shift()).cumsum()
    stale_run = unchanged.groupby(groups).cumcount() + 1
    stale_run = stale_run.where(unchanged, 0)
    issue = abs_returns.gt(config.max_unhandled_corporate_action_return)
    if config.max_stale_price_days > 0:
        issue = issue | stale_run.gt(config.max_stale_price_days)
    issue_reason = pd.Series("", index=ordered.index, dtype=object)
    issue_reason.loc[abs_returns.gt(config.max_unhandled_corporate_action_return)] = "post_corporate_action_quarantine"
    issue_reason.loc[stale_run.gt(config.max_stale_price_days)] = issue_reason.loc[stale_run.gt(config.max_stale_price_days)].mask(
        issue_reason.loc[stale_run.gt(config.max_stale_price_days)].eq(""),
        "post_stale_price_quarantine",
    )
    issue_id = issue.cumsum()
    clean_days = ordered.groupby(issue_id).cumcount()
    clean_days = clean_days.where(issue_id.gt(0), pd.NA)
    reason_values: list[str] = []
    last_reason = ""
    for is_issue, reason in zip(issue.to_numpy(), issue_reason.to_numpy()):
        if bool(is_issue) and str(reason):
            last_reason = str(reason)
        reason_values.append(last_reason)
    quality_adjusted = clean_days.ge(config.min_quality_adjusted_history_days).fillna(False)
    returns = abs_returns.fillna(0.0)
    return pd.DataFrame(
        {
            "max_stale_price_run": stale_run.expanding(min_periods=1).max().astype(float).to_numpy(),
            "max_abs_daily_return": returns.expanding(min_periods=1).max().astype(float).to_numpy(),
            "unhandled_corporate_action_count": abs_returns.gt(config.max_unhandled_corporate_action_return).cumsum().astype(float).to_numpy(),
            "current_stale_price_run": stale_run.astype(float).to_numpy(),
            "current_unhandled_jump": abs_returns.gt(config.max_unhandled_corporate_action_return).fillna(False).to_numpy(),
            "quality_adjusted": quality_adjusted.astype(bool).to_numpy(),
            "quality_adjustment_reason": reason_values,
            "quality_position_multiplier": quality_adjusted.map(
                {True: float(config.quality_adjusted_position_multiplier), False: 1.0}
            ).to_numpy(),
            "quality_history_start": ordered["date"].where(quality_adjusted).groupby(issue_id).transform("min").to_numpy(),
        },
        index=ordered.index,
    )


def _asof_filter_reasons(row: pd.Series, config: UniverseConfig) -> list[str]:
    reasons: list[str] = []
    if pd.isna(row["close"]) or float(row["close"]) < config.min_price:
        reasons.append("price_below_min")
    if pd.isna(row["avg_dollar_volume_20d"]) or float(row["avg_dollar_volume_20d"]) < config.min_avg_dollar_volume_20d:
        reasons.append("dollar_volume_below_min")
    if pd.isna(row["avg_volume_20d"]) or float(row["avg_volume_20d"]) < config.min_avg_volume_20d:
        reasons.append("volume_below_min")
    quality_history_ok = bool(row.get("quality_adjusted", False))
    if int(row["history_days"]) < config.min_history_days and not quality_history_ok:
        reasons.append("limited_history")
    if float(row.get("current_stale_price_run", 0.0) or 0.0) > config.max_stale_price_days:
        reasons.append("stale_price")
    if bool(row.get("current_unhandled_jump", False)) and not quality_history_ok:
        reasons.append("possible_unhandled_corporate_action")
    if quality_history_ok and not reasons:
        reasons.append("quality_adjusted_history")
    return reasons


def _candidate_reason(row: pd.Series) -> str:
    if bool(row.get("quality_adjusted", False)) and bool(row.get("tradable", False)):
        start = row.get("quality_history_start")
        start_text = pd.Timestamp(start).date() if pd.notna(start) else "unknown"
        return f"quality-adjusted clean segment from {start_text}; position multiplier={float(row.get('quality_position_multiplier', 1.0)):.2f}"
    if bool(row.get("tradable", False)):
        return "passed as-of tradable filters"
    return f"not tradable as of date: {row.get('filter_reason', '')}"


def _last_stale_run_end(frame: pd.DataFrame, max_stale_days: int) -> pd.Timestamp | None:
    """Return the end date of the last stale close run above the threshold."""

    if max_stale_days <= 0 or frame.empty:
        return None
    ordered = frame.sort_values("date")
    close = pd.to_numeric(ordered["adj_close"] if "adj_close" in ordered else ordered["close"], errors="coerce")
    unchanged = close.eq(close.shift())
    groups = unchanged.ne(unchanged.shift()).cumsum()
    stale_groups = unchanged.groupby(groups).sum()
    oversized = stale_groups[stale_groups > max_stale_days]
    if oversized.empty:
        return None
    group_id = oversized.index[-1]
    return pd.Timestamp(ordered.loc[groups == group_id, "date"].max())


def _price_quality_flags(frame: pd.DataFrame, config: UniverseConfig) -> dict[str, float]:
    """Return simple full-series data quality flags for one symbol.

    These are data hygiene filters, not alpha filters: a long flat close series
    often means a stale/delisted feed, and very large adjusted-close jumps often
    mean a corporate action was not handled cleanly enough for stop/ATR logic.
    """

    ordered = frame.sort_values("date").copy()
    close = pd.to_numeric(ordered["adj_close"] if "adj_close" in ordered else ordered["close"], errors="coerce")
    close = close.replace([float("inf"), float("-inf")], pd.NA).dropna()
    if close.empty:
        return {
            "max_stale_price_run": 0.0,
            "max_abs_daily_return": 0.0,
            "unhandled_corporate_action_count": 0.0,
        }
    unchanged = close.eq(close.shift())
    groups = unchanged.ne(unchanged.shift()).cumsum()
    stale_runs = unchanged.groupby(groups).sum()
    max_stale = int(stale_runs.max()) if not stale_runs.empty else 0
    returns = close.pct_change().replace([float("inf"), float("-inf")], pd.NA).dropna()
    abs_returns = returns.abs()
    max_abs_return = float(abs_returns.max()) if not abs_returns.empty else 0.0
    corporate_action_count = int((abs_returns > config.max_unhandled_corporate_action_return).sum())
    return {
        "max_stale_price_run": float(max_stale),
        "max_abs_daily_return": max_abs_return,
        "unhandled_corporate_action_count": float(corporate_action_count),
    }
