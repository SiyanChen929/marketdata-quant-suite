"""Live trading data-quality gate for target weights."""

from __future__ import annotations

import pandas as pd

from quant_system.config import AppConfig


def apply_latest_data_quality_gate_to_targets(
    targets: pd.DataFrame,
    data_quality_detail: pd.DataFrame,
    config: AppConfig,
) -> tuple[pd.DataFrame, list[str]]:
    """Set latest target weights to zero for symbols blocked by live data checks.

    The gate is intentionally live-only. Daily data-quality checks normally use
    the newest downloaded/broker-reconciled state, so applying them to older
    training or validation dates would leak current knowledge into history.
    """

    warnings: list[str] = []
    if targets.empty or data_quality_detail.empty:
        return targets, warnings
    if not {"date", "symbol", "target_weight"}.issubset(targets.columns):
        return targets, warnings
    if not _blocking_enabled(config):
        return targets, warnings
    if "blocked_from_trading" not in data_quality_detail:
        return targets, warnings

    out = targets.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["symbol"] = out["symbol"].astype(str).str.upper()
    latest_target_date = out["date"].max()
    quality = data_quality_detail.copy()
    quality["symbol"] = quality["symbol"].astype(str).str.upper()
    if "latest_date" in quality:
        quality["latest_date"] = pd.to_datetime(quality["latest_date"], errors="coerce").dt.normalize()
        quality_latest_date = quality["latest_date"].max()
        if pd.notna(quality_latest_date) and pd.notna(latest_target_date) and latest_target_date < quality_latest_date:
            warnings.append(
                "Skipped live data-quality target blocking for a historical window; "
                "latest quality checks are only used for the current trading plan."
            )
            return out, warnings

    blocked = quality[quality["blocked_from_trading"].fillna(False).astype(bool)].copy()
    if blocked.empty:
        return out, warnings
    blocked_symbols = set(blocked["symbol"])
    latest_mask = out["date"].eq(latest_target_date)
    block_mask = latest_mask & out["symbol"].isin(blocked_symbols)
    if not block_mask.any():
        return out, warnings

    quality_map = blocked.set_index("symbol")
    out["data_quality_blocked"] = out.get("data_quality_blocked", False)
    out["data_quality_block_reason"] = out.get("data_quality_block_reason", "")
    out["data_quality_status"] = out.get("data_quality_status", "")
    out.loc[block_mask, "target_weight_before_data_quality_gate"] = out.loc[block_mask, "target_weight"]
    out.loc[block_mask, "target_weight"] = 0.0
    for symbol in sorted(out.loc[block_mask, "symbol"].unique()):
        row_mask = block_mask & out["symbol"].eq(symbol)
        if symbol in quality_map.index:
            out.loc[row_mask, "data_quality_blocked"] = True
            out.loc[row_mask, "data_quality_block_reason"] = str(quality_map.at[symbol, "block_reason"]) if "block_reason" in quality_map else ""
            out.loc[row_mask, "data_quality_status"] = str(quality_map.at[symbol, "status"]) if "status" in quality_map else ""
    warnings.append(
        "Live data-quality gate set latest target weights to zero for "
        f"{int(block_mask.sum())} blocked symbol rows: {', '.join(sorted(blocked_symbols.intersection(set(out.loc[block_mask, 'symbol']))))}."
    )
    return out, warnings


def _blocking_enabled(config: AppConfig) -> bool:
    quality = config.data_quality
    return bool(quality.block_on_stale_price or quality.block_on_abnormal_jump or quality.block_on_broker_price_diff)
