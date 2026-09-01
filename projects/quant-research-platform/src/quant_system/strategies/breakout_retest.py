"""Breakout + retest trend system."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.strategies.base import Strategy, StrategyContext, attach_context_scores, event_entries_allowed, final_score, output_columns


class BreakoutRetestTrendSystem(Strategy):
    """Trade high-volume breakouts and breakdowns with relative strength."""

    name = "breakout"

    def generate_signals(self, features: pd.DataFrame, context: StrategyContext) -> pd.DataFrame:
        params = self.params
        window = int(params.get("breakout_window", 55))
        volume_multiplier = float(params.get("volume_multiplier", 1.5))
        retest_ma = int(params.get("retest_ma", 20))
        max_positions = int(params.get("max_positions", 50))
        out = features.sort_values(["symbol", "date"]).copy()
        high_col = f"high_{window}_prior" if f"high_{window}_prior" in out.columns else "high_252_prior"
        low_col = f"low_{window}_prior" if f"low_{window}_prior" in out.columns else "low_252_prior"
        ma_col = f"ma_{retest_ma}" if f"ma_{retest_ma}" in out.columns else "ma_20"
        rs_col = "rs_63d" if "rs_63d" in out.columns else "ret_63d"
        breakout = (out["adj_close"] >= out[high_col]) & (out["volume_expansion"] >= volume_multiplier) & (out[rs_col] > 0)
        breakdown = (out["adj_close"] <= out[low_col]) & (out["volume_expansion"] >= volume_multiplier) & (out[rs_col] < 0)
        out["retest_distance_ma_pct"] = out["adj_close"] / out[ma_col] - 1.0
        retest_proximity = (1.0 - out["retest_distance_ma_pct"].abs() / 0.03).clip(lower=0.0, upper=1.0)
        retest_long = (
            (out["adj_close"] > out[ma_col])
            & (out["low"] <= out[ma_col] * 1.01)
            & (out[rs_col] > 0)
            & (out["adj_close"] > out["ma_50"])
        )
        retest_short = (
            (out["adj_close"] < out[ma_col])
            & (out["high"] >= out[ma_col] * 0.99)
            & (out[rs_col] < 0)
            & (out["adj_close"] < out["ma_50"])
        )
        out["breakout_component"] = breakout.astype(float) * 100
        out["volume_component"] = out["volume_expansion"].clip(0, 3) / 3 * 100
        out["rs_component"] = out.groupby("date")[rs_col].rank(pct=True) * 100
        out["retest_component"] = np.where(retest_long | retest_short, 60.0 + 40.0 * retest_proximity, 25.0 * retest_proximity)
        out["technical_score"] = (
            0.40 * out["breakout_component"]
            + 0.25 * out["volume_component"]
            + 0.25 * out["rs_component"]
            + 0.10 * out["retest_component"]
        )
        out["relative_strength_score"] = out["rs_component"]
        out = attach_context_scores(out, context)
        out["final_score"] = final_score(out)
        out["signal"] = 0.0
        event_allowed = event_entries_allowed(out, params)
        out.loc[(breakout | retest_long) & event_allowed & (out["fundamental_score"].fillna(50) >= float(params.get("long_min_fundamental_score", 40))), "signal"] = 1.0
        short_quality = out["fundamental_score"].fillna(50) <= float(params.get("short_max_fundamental_score", 50))
        out.loc[(breakdown | retest_short) & event_allowed & short_quality, "signal"] = -1.0
        out["stop_loss_atr"] = float(params.get("stop_loss_atr", 2.5))
        out["trailing_stop_atr"] = float(params.get("trailing_stop_atr", 3.0))
        out["take_profit_r_multiple"] = float(params.get("take_profit_r_multiple", 0.0))
        out["max_holding_days"] = int(params.get("max_holding_days", 60))
        out["score_decomposition"] = out.apply(
            lambda row: (
                f"breakout={row['breakout_component']:.1f}; volume={row['volume_component']:.1f}; "
                f"rs={row['rs_component']:.1f}; retest_score={row['retest_component']:.1f}; "
                f"ma_gap={row.get('retest_distance_ma_pct', 0):.2%}; "
                f"fundamental={row.get('fundamental_score', 50):.1f}; event_risk={row.get('event_risk_score', 0):.1f}"
            ),
            axis=1,
        )
        out["reason_for_entry"] = out.apply(_reason, axis=1)
        signals = out[out["technical_score"].notna()][output_columns(out)].sort_values(["date", "final_score"], ascending=[True, False])
        return signals.groupby("date", group_keys=False).head(max_positions * 2).reset_index(drop=True)


def _reason(row: pd.Series) -> str:
    setup = (
        f"breakout={row.get('breakout_component', 0):.0f}, "
        f"volume={row.get('volume_component', 0):.0f}, "
        f"rs={row.get('rs_component', 0):.0f}, "
        f"retest={row.get('retest_component', 0):.0f}, "
        f"ma_gap={row.get('retest_distance_ma_pct', 0):.1%}, "
        f"event={row.get('event_risk_score', 0):.1f}"
    )
    if row["signal"] > 0:
        return f"Long breakout/retest: price action confirms either prior-high breakout or controlled MA retest; {setup}."
    if row["signal"] < 0:
        return f"Short breakdown/retest: downside break or failed MA reclaim with weak relative strength; {setup}."
    return f"No breakout/retest position: setup incomplete or risk/quality gate failed; {setup}."
