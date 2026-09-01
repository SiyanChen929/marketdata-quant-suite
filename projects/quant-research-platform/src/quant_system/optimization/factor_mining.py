"""Cross-sectional factor mining workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from quant_system.config import AppConfig
from quant_system.strategies.base import StrategyContext


DEFAULT_HORIZONS = (1, 5, 21)


def mine_factors(
    features: pd.DataFrame,
    context: StrategyContext,
    config: AppConfig,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    max_factors: int = 80,
    min_names_per_date: int = 20,
) -> dict[str, pd.DataFrame]:
    """Mine point-in-time factors with IC, quantile spreads, and stability flags."""

    panel = build_factor_mining_panel(features, context, horizons=horizons)
    if panel.empty:
        empty = pd.DataFrame()
        return {"panel": empty, "summary": empty, "ic_by_period": empty, "quantiles": empty, "correlation": empty, "selected": empty}
    factors = candidate_factor_columns(panel, max_factors=max_factors)
    ic_by_period = factor_ic_by_period(
        panel,
        factors,
        config,
        horizons=horizons,
        min_names_per_date=min_names_per_date,
    )
    quantiles = factor_quantile_spreads(panel, factors, horizons=horizons, min_names_per_date=min_names_per_date)
    corr = factor_correlation(panel, factors)
    summary = summarize_factor_candidates(ic_by_period, corr)
    selected = summary[summary["candidate_status"].eq("candidate")].copy() if not summary.empty else pd.DataFrame()
    selected_decorrelated = decorrelate_factor_candidates(selected, corr)
    return {
        "panel": panel,
        "summary": summary,
        "ic_by_period": ic_by_period,
        "quantiles": quantiles,
        "correlation": corr,
        "selected": selected,
        "selected_decorrelated": selected_decorrelated,
    }


def add_activation_diagnostics(
    tables: dict[str, pd.DataFrame],
    signals: pd.DataFrame,
    config: AppConfig,
    *,
    horizon: int = 5,
    top_n: int = 20,
    min_names_per_date: int = 10,
) -> dict[str, pd.DataFrame]:
    """Attach activation diagnostics that compare factor top names to current signal top names.

    The diagnostic is deliberately read-only. It helps reject sparse or no-op
    research ideas before spending a full backtest cycle on a new overlay.
    """

    out = dict(tables)
    panel = out.get("panel", pd.DataFrame())
    selected = out.get("selected_decorrelated", pd.DataFrame())
    out["activation"] = factor_activation_diagnostics(
        panel,
        signals,
        selected,
        config,
        horizon=horizon,
        top_n=top_n,
        min_names_per_date=min_names_per_date,
    )
    return out


def factor_activation_diagnostics(
    panel: pd.DataFrame,
    signals: pd.DataFrame,
    selected_factors: pd.DataFrame,
    config: AppConfig,
    *,
    horizon: int = 5,
    top_n: int = 20,
    min_names_per_date: int = 10,
) -> pd.DataFrame:
    """Measure whether candidate factors would actually change selected names.

    The baseline universe is the current strategy signal table for each date,
    not the full tradable universe. This answers the self-evolution question:
    "Would this factor change the names the current strategy is already
    considering, and did those changed names help out-of-sample?"
    """

    ret_col = f"fwd_{int(horizon)}d"
    if panel.empty or signals.empty or selected_factors.empty or ret_col not in panel:
        return pd.DataFrame()
    if not {"date", "symbol", "final_score"}.issubset(signals.columns):
        return pd.DataFrame()
    factors = _activation_factor_rows(selected_factors)
    if factors.empty:
        return pd.DataFrame()
    signal_cols = ["date", "symbol", "final_score"]
    signal_frame = signals[signal_cols].copy()
    signal_frame["date"] = pd.to_datetime(signal_frame["date"], errors="coerce").dt.normalize()
    signal_frame["symbol"] = signal_frame["symbol"].astype(str).str.upper()
    signal_frame["baseline_score"] = pd.to_numeric(signal_frame["final_score"], errors="coerce")
    keep = ["date", "symbol", ret_col, *factors["factor"].astype(str).tolist()]
    data = panel[[col for col in keep if col in panel.columns]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    merged = signal_frame.merge(data, on=["date", "symbol"], how="left").dropna(subset=["date", "symbol", "baseline_score"])
    if merged.empty:
        return pd.DataFrame()
    periods = [
        ("full", None, None, False),
        ("train", config.regime.train_start, config.regime.train_end, True),
        ("validation", config.regime.validation_start, config.regime.validation_end, False),
        ("forward", config.regime.forward_start, None, False),
    ]
    rows: list[dict] = []
    for _, factor_row in factors.iterrows():
        factor = str(factor_row["factor"])
        if factor not in merged:
            continue
        direction = str(factor_row.get("direction", "high_is_good"))
        for period, start, end, used_for_selection in periods:
            period_frame = _date_slice(merged, start, end)
            summary = _factor_activation_for_period(
                period_frame,
                factor,
                ret_col,
                direction=direction,
                top_n=top_n,
                min_names_per_date=min_names_per_date,
            )
            if not summary:
                continue
            summary.update(
                {
                    "period": period,
                    "factor": factor,
                    "direction": direction,
                    "horizon": f"{int(horizon)}d",
                    "horizon_days": int(horizon),
                    "used_for_selection": used_for_selection,
                    "selection_score": factor_row.get("selection_score", np.nan),
                    "overfit_flags": factor_row.get("overfit_flags", ""),
                    "activation_note": "Forward/current is diagnostic only; do not select parameters from it.",
                }
            )
            rows.append(summary)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["period", "activation_score"], ascending=[True, False]).reset_index(drop=True)


def build_factor_mining_panel(
    features: pd.DataFrame,
    context: StrategyContext,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """Build a point-in-time factor panel and forward returns."""

    if features.empty or not {"date", "symbol", "adj_close"}.issubset(features.columns):
        return pd.DataFrame()
    data = features.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data["symbol"] = data["symbol"].astype(str).str.upper()
    data = data.sort_values(["symbol", "date"]).dropna(subset=["date", "symbol", "adj_close"])
    data = _add_derived_factors(data)
    data = _merge_context_factors(data, context)
    price = data[["date", "symbol", "adj_close"]].drop_duplicates(["date", "symbol"]).sort_values(["symbol", "date"])
    wide = price.pivot(index="date", columns="symbol", values="adj_close").sort_index()
    for horizon in horizons:
        h = int(horizon)
        forward_matrix = wide.shift(-h) / wide - 1.0
        try:
            forward = forward_matrix.stack(future_stack=True).dropna().rename(f"fwd_{h}d").reset_index()
        except TypeError:
            forward = forward_matrix.stack(dropna=True).rename(f"fwd_{h}d").reset_index()
        forward["symbol"] = forward["symbol"].astype(str).str.upper()
        data = data.merge(forward, on=["date", "symbol"], how="left")
    return data


def candidate_factor_columns(panel: pd.DataFrame, *, max_factors: int = 80) -> list[str]:
    """Return numeric columns suitable for cross-sectional factor research."""

    include_prefixes = (
        "ret_",
        "rs_",
        "vol_",
        "rsi_",
        "zscore_",
        "distance_",
        "volume_",
        "price_vs_",
        "ma_spread_",
        "mom_",
        "atr_",
        "theme_",
        "fundamental_",
        "growth_",
        "quality_",
        "balance_sheet_",
        "valuation_",
        "revision_",
        "event_",
        "gap_",
        "liquidity_",
    )
    exclude = {
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "date",
        "symbol",
        "return_1d",
    }
    factors: list[str] = []
    numeric = panel.select_dtypes(include=[np.number]).columns
    for column in numeric:
        if column in exclude or str(column).startswith("fwd_"):
            continue
        if str(column).startswith(include_prefixes):
            non_null = int(panel[column].notna().sum())
            unique = int(panel[column].nunique(dropna=True))
            if non_null >= 500 and unique >= 10:
                factors.append(str(column))
    coverage = {factor: float(panel[factor].notna().mean()) for factor in factors}
    factors = sorted(factors, key=lambda factor: (coverage[factor], factor), reverse=True)
    return factors[: int(max_factors)]


def factor_ic_by_period(
    panel: pd.DataFrame,
    factors: Iterable[str],
    config: AppConfig,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    min_names_per_date: int = 20,
) -> pd.DataFrame:
    """Compute rank IC summaries for train, validation, forward, and full periods."""

    periods = [
        ("full", None, None, True),
        ("train", config.regime.train_start, config.regime.train_end, True),
        ("validation", config.regime.validation_start, config.regime.validation_end, True),
        ("forward", config.regime.forward_start, None, False),
    ]
    rows: list[dict] = []
    for period, start, end, used_for_selection in periods:
        sliced = _date_slice(panel, start, end)
        if sliced.empty:
            continue
        for factor in factors:
            for horizon in horizons:
                ret_col = f"fwd_{int(horizon)}d"
                if ret_col not in sliced:
                    continue
                daily = _daily_rank_ic(sliced, factor, ret_col, min_names_per_date=min_names_per_date)
                if len(daily) < 20:
                    continue
                summary = _summarize_daily_ic(daily)
                summary.update(
                    {
                        "period": period,
                        "factor": factor,
                        "horizon": f"{int(horizon)}d",
                        "horizon_days": int(horizon),
                        "used_for_selection": used_for_selection,
                    }
                )
                rows.append(summary)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["period", "horizon_days", "mean_rank_ic"], ascending=[True, True, False]).reset_index(drop=True)


def factor_quantile_spreads(
    panel: pd.DataFrame,
    factors: Iterable[str],
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    min_names_per_date: int = 20,
    quantiles: int = 5,
) -> pd.DataFrame:
    """Compute top-minus-bottom forward-return spreads by factor."""

    rows: list[dict] = []
    for factor in factors:
        for horizon in horizons:
            ret_col = f"fwd_{int(horizon)}d"
            if ret_col not in panel:
                continue
            spreads: list[float] = []
            top_returns: list[float] = []
            bottom_returns: list[float] = []
            for date, day in panel[["date", factor, ret_col]].dropna().groupby("date", sort=True):
                if len(day) < min_names_per_date or day[factor].nunique() < quantiles:
                    continue
                ranks = day[factor].rank(method="first")
                try:
                    bucket = pd.qcut(ranks, quantiles, labels=False) + 1
                except ValueError:
                    continue
                top = day.loc[bucket.eq(quantiles), ret_col].mean()
                bottom = day.loc[bucket.eq(1), ret_col].mean()
                if pd.notna(top) and pd.notna(bottom):
                    spreads.append(float(top - bottom))
                    top_returns.append(float(top))
                    bottom_returns.append(float(bottom))
            if len(spreads) < 20:
                continue
            spread = pd.Series(spreads, dtype=float)
            rows.append(
                {
                    "factor": factor,
                    "horizon": f"{int(horizon)}d",
                    "horizon_days": int(horizon),
                    "mean_quantile_spread": float(spread.mean()),
                    "median_quantile_spread": float(spread.median()),
                    "spread_t_stat": float(spread.mean() / (spread.std(ddof=1) / np.sqrt(len(spread)))) if spread.std(ddof=1) > 0 else np.nan,
                    "positive_spread_rate": float((spread > 0).mean()),
                    "mean_top_quantile_return": float(np.mean(top_returns)),
                    "mean_bottom_quantile_return": float(np.mean(bottom_returns)),
                    "n_days": int(len(spread)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["horizon_days", "mean_quantile_spread"], ascending=[True, False]).reset_index(drop=True)


def factor_correlation(panel: pd.DataFrame, factors: Iterable[str]) -> pd.DataFrame:
    """Compute factor rank correlation and max-correlation diagnostics."""

    available = [factor for factor in factors if factor in panel and panel[factor].notna().sum() >= 500]
    if len(available) < 2:
        return pd.DataFrame(columns=["factor_a", "factor_b", "rank_correlation", "abs_correlation"])
    ranked = panel[available].rank()
    corr = ranked.corr(method="pearson")
    rows = []
    for idx, factor_a in enumerate(available):
        for factor_b in available[idx + 1 :]:
            value = corr.loc[factor_a, factor_b]
            if pd.notna(value):
                rows.append({"factor_a": factor_a, "factor_b": factor_b, "rank_correlation": float(value), "abs_correlation": abs(float(value))})
    return pd.DataFrame(rows).sort_values("abs_correlation", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def summarize_factor_candidates(ic_by_period: pd.DataFrame, corr: pd.DataFrame) -> pd.DataFrame:
    """Create a candidate ranking without using forward period for selection."""

    if ic_by_period.empty:
        return pd.DataFrame()
    one = ic_by_period[ic_by_period["horizon_days"].eq(5)].copy()
    if one.empty:
        one = ic_by_period.copy()
    pivot = one.pivot_table(index="factor", columns="period", values="mean_rank_ic", aggfunc="mean")
    pos = one.pivot_table(index="factor", columns="period", values="positive_ic_rate", aggfunc="mean")
    tstat = one.pivot_table(index="factor", columns="period", values="ic_t_stat", aggfunc="mean")
    rows: list[dict] = []
    max_corr = _max_abs_corr(corr)
    for factor, values in pivot.iterrows():
        train = float(values.get("train", np.nan))
        validation = float(values.get("validation", np.nan))
        forward = float(values.get("forward", np.nan))
        direction = 1.0 if (pd.isna(train) or train >= 0) else -1.0
        train_aligned = direction * train if pd.notna(train) else np.nan
        validation_aligned = direction * validation if pd.notna(validation) else np.nan
        forward_aligned = direction * forward if pd.notna(forward) else np.nan
        validation_pos = float(pos.loc[factor].get("validation", np.nan)) if factor in pos.index else np.nan
        train_t = float(tstat.loc[factor].get("train", np.nan)) if factor in tstat.index else np.nan
        validation_t = float(tstat.loc[factor].get("validation", np.nan)) if factor in tstat.index else np.nan
        corr_penalty = max(0.0, float(max_corr.get(factor, 0.0)) - 0.80) * 0.05
        selection_score = (
            0.45 * _nan_to_zero(validation_aligned)
            + 0.30 * _nan_to_zero(train_aligned)
            + 0.15 * max(0.0, _nan_to_zero(validation_pos) - 0.50)
            + 0.10 * min(abs(_nan_to_zero(validation_t)) / 10.0, 1.0)
            - corr_penalty
        )
        flags = []
        if pd.notna(train_aligned) and pd.notna(validation_aligned) and train_aligned > 0.03 and validation_aligned < 0.005:
            flags.append("train_only_alpha")
        if pd.notna(train) and pd.notna(validation) and np.sign(train) != np.sign(validation):
            flags.append("sign_flip_train_validation")
        if max_corr.get(factor, 0.0) > 0.90:
            flags.append("high_collinearity")
        if pd.notna(validation_t) and abs(validation_t) < 1.0:
            flags.append("weak_validation_tstat")
        status = "candidate" if selection_score > 0 and "sign_flip_train_validation" not in flags and "weak_validation_tstat" not in flags else "reject"
        rows.append(
            {
                "factor": factor,
                "direction": "high_is_good" if direction > 0 else "low_is_good",
                "selection_score": float(selection_score),
                "candidate_status": status,
                "train_mean_ic_aligned": train_aligned,
                "validation_mean_ic_aligned": validation_aligned,
                "forward_mean_ic_aligned_not_used": forward_aligned,
                "validation_positive_ic_rate": validation_pos,
                "train_t_stat": train_t,
                "validation_t_stat": validation_t,
                "max_abs_correlation": float(max_corr.get(factor, np.nan)),
                "overfit_flags": "|".join(flags) if flags else "",
                "selection_note": "Forward/current period is diagnostic only and is not used in selection_score.",
            }
        )
    return pd.DataFrame(rows).sort_values("selection_score", ascending=False).reset_index(drop=True)


def decorrelate_factor_candidates(selected: pd.DataFrame, corr: pd.DataFrame, *, max_abs_corr: float = 0.80) -> pd.DataFrame:
    """Greedily keep high-scoring factors that are not near-duplicates."""

    if selected.empty:
        return selected
    corr_map: dict[tuple[str, str], float] = {}
    if not corr.empty:
        for _, row in corr.iterrows():
            a = str(row.get("factor_a", ""))
            b = str(row.get("factor_b", ""))
            value = float(row.get("abs_correlation", 0.0) or 0.0)
            corr_map[(a, b)] = value
            corr_map[(b, a)] = value
    kept: list[dict] = []
    dropped: list[dict] = []
    for _, row in selected.sort_values("selection_score", ascending=False).iterrows():
        factor = str(row["factor"])
        duplicate_of = ""
        duplicate_corr = 0.0
        for kept_row in kept:
            other = str(kept_row["factor"])
            value = corr_map.get((factor, other), 0.0)
            if value > duplicate_corr:
                duplicate_corr = value
                duplicate_of = other
        record = row.to_dict()
        if duplicate_corr >= max_abs_corr:
            record["decorrelation_status"] = "drop_collinear"
            record["representative_factor"] = duplicate_of
            record["representative_abs_corr"] = duplicate_corr
            dropped.append(record)
            continue
        record["decorrelation_status"] = "keep"
        record["representative_factor"] = factor
        record["representative_abs_corr"] = duplicate_corr
        kept.append(record)
    out = pd.DataFrame([*kept, *dropped])
    if out.empty:
        return out
    return out.sort_values(["decorrelation_status", "selection_score"], ascending=[False, False]).reset_index(drop=True)


def save_factor_mining_report(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Write factor mining artifacts and a small HTML report."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        if name == "panel":
            frame.head(200_000).to_csv(out / "factor_mining_panel_sample.csv", index=False)
        else:
            frame.to_csv(out / f"{name}.csv", index=False)
    html = _html_report(tables)
    (out / "factor_mining_report.html").write_text(html, encoding="utf-8")


def _add_derived_factors(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    grouped = out.groupby("symbol", group_keys=False)
    for window, skip in [(21, 0), (21, 5), (63, 5), (126, 5), (252, 21)]:
        col = f"mom_{window}_skip{skip}"
        out[col] = grouped["adj_close"].transform(lambda s, w=window, k=skip: s.shift(k) / s.shift(w + k) - 1.0)
        vol_col = f"vol_{min(window, 63)}d"
        if vol_col in out:
            out[f"{col}_voladj"] = out[col] / out[vol_col].replace(0, np.nan)
    for window in (20, 50, 200):
        ma = f"ma_{window}"
        if ma in out:
            out[f"price_vs_ma_{window}"] = out["adj_close"] / out[ma].replace(0, np.nan) - 1.0
    if {"ma_20", "ma_50"}.issubset(out.columns):
        out["ma_spread_20_50"] = out["ma_20"] / out["ma_50"].replace(0, np.nan) - 1.0
    if {"ma_50", "ma_200"}.issubset(out.columns):
        out["ma_spread_50_200"] = out["ma_50"] / out["ma_200"].replace(0, np.nan) - 1.0
    if {"rs_21d", "rs_63d"}.issubset(out.columns):
        out["rs_acceleration_21_63"] = out["rs_21d"] - out["rs_63d"]
    if {"atr_14", "adj_close"}.issubset(out.columns):
        out["atr_14_pct"] = out["atr_14"] / out["adj_close"].replace(0, np.nan)
    if {"volume", "volume_ma_20"}.issubset(out.columns):
        vol_std = grouped["volume"].transform(lambda s: s.rolling(20, min_periods=10).std())
        out["volume_zscore_20"] = (out["volume"] - out["volume_ma_20"]) / vol_std.replace(0, np.nan)
    if {"volume", "adj_close"}.issubset(out.columns):
        dollar_volume = out["volume"] * out["adj_close"]
        out["liquidity_log_dollar_volume"] = np.log1p(dollar_volume.clip(lower=0))
    if {"low_252", "adj_close"}.issubset(out.columns):
        out["distance_from_low_252"] = out["adj_close"] / out["low_252"].replace(0, np.nan) - 1.0
    return out


def _activation_factor_rows(selected_factors: pd.DataFrame, *, max_factors: int = 20) -> pd.DataFrame:
    data = selected_factors.copy()
    if "decorrelation_status" in data:
        data = data[data["decorrelation_status"].eq("keep")]
    if "candidate_status" in data:
        data = data[data["candidate_status"].eq("candidate")]
    if "factor" not in data:
        return pd.DataFrame()
    if "selection_score" in data:
        data = data.sort_values("selection_score", ascending=False)
    return data.head(int(max_factors)).reset_index(drop=True)


def _factor_activation_for_period(
    data: pd.DataFrame,
    factor: str,
    ret_col: str,
    *,
    direction: str,
    top_n: int,
    min_names_per_date: int,
) -> dict[str, float] | None:
    daily_rows: list[dict[str, float]] = []
    sign = -1.0 if direction == "low_is_good" else 1.0
    subset = data[["date", "symbol", "baseline_score", factor, ret_col]].dropna()
    for date, day in subset.groupby("date", sort=True):
        if len(day) < int(min_names_per_date) or day[factor].nunique() < 5:
            continue
        n = min(int(top_n), len(day))
        if n <= 0:
            continue
        baseline = day.sort_values("baseline_score", ascending=False).head(n)
        factor_sorted = day.assign(_activation_factor_score=sign * pd.to_numeric(day[factor], errors="coerce")).sort_values(
            "_activation_factor_score", ascending=False
        )
        factor_top = factor_sorted.head(n)
        baseline_symbols = set(baseline["symbol"].astype(str))
        factor_symbols = set(factor_top["symbol"].astype(str))
        overlap = len(baseline_symbols & factor_symbols)
        changed = n - overlap
        if n == 0:
            continue
        daily_rows.append(
            {
                "date": pd.Timestamp(date),
                "top_n": float(n),
                "overlap_rate": float(overlap / n),
                "changed_names": float(changed),
                "changed_name_rate": float(changed / n),
                "baseline_top_forward_return": float(pd.to_numeric(baseline[ret_col], errors="coerce").mean()),
                "factor_top_forward_return": float(pd.to_numeric(factor_top[ret_col], errors="coerce").mean()),
            }
        )
    if len(daily_rows) < 10:
        return None
    daily = pd.DataFrame(daily_rows)
    lift = daily["factor_top_forward_return"] - daily["baseline_top_forward_return"]
    activation_score = float(lift.mean() * max(float(daily["changed_name_rate"].mean()), 0.0))
    return {
        "activation_score": activation_score,
        "mean_forward_return_lift": float(lift.mean()),
        "median_forward_return_lift": float(lift.median()),
        "positive_lift_rate": float((lift > 0).mean()),
        "mean_overlap_rate": float(daily["overlap_rate"].mean()),
        "mean_changed_names": float(daily["changed_names"].mean()),
        "mean_changed_name_rate": float(daily["changed_name_rate"].mean()),
        "mean_baseline_top_forward_return": float(daily["baseline_top_forward_return"].mean()),
        "mean_factor_top_forward_return": float(daily["factor_top_forward_return"].mean()),
        "n_days": int(len(daily)),
    }


def _merge_context_factors(data: pd.DataFrame, context: StrategyContext) -> pd.DataFrame:
    out = data.copy()
    for frame in [context.theme_scores, context.fundamentals, context.events]:
        if frame is None or frame.empty or not {"date", "symbol"}.issubset(frame.columns):
            continue
        temp = frame.copy()
        temp["date"] = pd.to_datetime(temp["date"], errors="coerce").dt.normalize()
        temp["symbol"] = temp["symbol"].astype(str).str.upper()
        merge_cols = ["date", "symbol"] + [
            col
            for col in temp.columns
            if col not in {"date", "symbol"} and (col not in out.columns or col in {"theme_score", "theme_active"})
        ]
        out = out.merge(temp[merge_cols].drop_duplicates(["date", "symbol"]), on=["date", "symbol"], how="left", suffixes=("", "_ctx"))
    return out


def _daily_rank_ic(panel: pd.DataFrame, factor: str, ret_col: str, *, min_names_per_date: int) -> pd.Series:
    values: dict[pd.Timestamp, float] = {}
    data = panel[["date", factor, ret_col]].dropna()
    for date, day in data.groupby("date", sort=True):
        if len(day) < min_names_per_date or day[factor].nunique() < 5 or day[ret_col].nunique() < 5:
            continue
        ic = day[factor].rank().corr(day[ret_col].rank())
        if pd.notna(ic):
            values[pd.Timestamp(date)] = float(ic)
    return pd.Series(values, dtype=float).sort_index()


def _summarize_daily_ic(daily: pd.Series) -> dict:
    mean = float(daily.mean())
    std = float(daily.std(ddof=1))
    return {
        "mean_rank_ic": mean,
        "median_rank_ic": float(daily.median()),
        "ic_t_stat": mean / (std / np.sqrt(len(daily))) if std > 0 else np.nan,
        "positive_ic_rate": float((daily > 0).mean()),
        "n_days": int(len(daily)),
    }


def _date_slice(panel: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    data = panel
    if start:
        data = data[data["date"].ge(pd.to_datetime(start))]
    if end:
        data = data[data["date"].le(pd.to_datetime(end))]
    return data


def _max_abs_corr(corr: pd.DataFrame) -> dict[str, float]:
    if corr.empty:
        return {}
    rows: dict[str, float] = {}
    for _, row in corr.iterrows():
        value = float(row.get("abs_correlation", 0.0) or 0.0)
        for col in ("factor_a", "factor_b"):
            factor = str(row.get(col, ""))
            rows[factor] = max(rows.get(factor, 0.0), value)
    return rows


def _nan_to_zero(value: float) -> float:
    return 0.0 if pd.isna(value) else float(value)


def _html_report(tables: dict[str, pd.DataFrame]) -> str:
    selected = tables.get("selected", pd.DataFrame())
    selected_decorrelated = tables.get("selected_decorrelated", pd.DataFrame())
    summary = tables.get("summary", pd.DataFrame())
    ic = tables.get("ic_by_period", pd.DataFrame())
    quantiles = tables.get("quantiles", pd.DataFrame())
    corr = tables.get("correlation", pd.DataFrame())
    activation = tables.get("activation", pd.DataFrame())
    selected_cols = [
        "factor",
        "direction",
        "selection_score",
        "validation_mean_ic_aligned",
        "forward_mean_ic_aligned_not_used",
        "validation_positive_ic_rate",
        "max_abs_correlation",
        "overfit_flags",
    ]
    selected_html = _format_table(selected, selected_cols, 30)
    decorrelated_cols = [
        "factor",
        "direction",
        "selection_score",
        "decorrelation_status",
        "representative_factor",
        "representative_abs_corr",
        "validation_mean_ic_aligned",
        "overfit_flags",
    ]
    decorrelated_html = _format_table(selected_decorrelated, decorrelated_cols, 40)
    summary_html = _format_table(summary, selected_cols, 50)
    ic_html = _format_table(ic, ["period", "factor", "horizon", "mean_rank_ic", "ic_t_stat", "positive_ic_rate", "n_days"], 80)
    quantile_html = _format_table(quantiles, ["factor", "horizon", "mean_quantile_spread", "spread_t_stat", "positive_spread_rate", "n_days"], 50)
    corr_html = _format_table(corr, ["factor_a", "factor_b", "rank_correlation", "abs_correlation"], 50)
    activation_html = _format_table(
        activation,
        [
            "period",
            "factor",
            "direction",
            "activation_score",
            "mean_forward_return_lift",
            "positive_lift_rate",
            "mean_changed_names",
            "mean_overlap_rate",
            "n_days",
        ],
        80,
    )
    return f"""<!doctype html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>Factor Mining Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 28px; color: #172033; }}
    .notice {{ background: #fff7e6; border: 1px solid #ffd58a; padding: 12px 14px; border-radius: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-bottom: 24px; }}
    th, td {{ border-bottom: 1px solid #e4e8f0; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>因子挖掘报告</h1>
  <p class="notice">候选因子的 selection_score 只使用 train + validation；forward/current 只用于诊断，不能反向调参。</p>
  <h2>去共线候选清单</h2>{decorrelated_html}
  <h2>因子激活诊断</h2>{activation_html}
  <h2>可候选因子 Top</h2>{selected_html}
  <h2>全部因子摘要</h2>{summary_html}
  <h2>IC 分段明细</h2>{ic_html}
  <h2>分位数组合 Spread</h2>{quantile_html}
  <h2>高相关因子</h2>{corr_html}
</body>
</html>"""


def _format_table(data: pd.DataFrame, columns: list[str], n: int) -> str:
    if data.empty:
        return "<p>无足够数据。</p>"
    cols = [col for col in columns if col in data.columns]
    table = data[cols].head(n).copy()
    for column in table.columns:
        if table[column].dtype.kind in "fc":
            table[column] = pd.to_numeric(table[column], errors="coerce").map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    return table.to_html(index=False, escape=False)
