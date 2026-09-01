"""Short-term mean reversion strategy."""

from __future__ import annotations

import pandas as pd

from quant_system.strategies.base import Strategy, StrategyContext, attach_context_scores, event_entries_allowed, final_score, output_columns


class MeanReversionShortTerm(Strategy):
    """Small-weight countertrend strategy gated by broader trend context."""

    name = "mean_reversion"

    def generate_signals(self, features: pd.DataFrame, context: StrategyContext) -> pd.DataFrame:
        params = self.params
        max_positions = int(params.get("max_positions", 30))
        out = features.sort_values(["symbol", "date"]).copy()
        rs_col = "rs_63d" if "rs_63d" in out.columns else "ret_63d"
        uptrend = (out["adj_close"] > out["ma_100"]) & (out["ma_50"] > out["ma_100"])
        downtrend = (out["adj_close"] < out["ma_100"]) & (out["ma_50"] < out["ma_100"])
        oversold = (out["rsi_14"] < float(params.get("long_rsi", 35))) | (out["zscore_20"] < float(params.get("long_zscore", -1.5))) | (out["adj_close"] < out["bb_lower_20"])
        overbought = (out["rsi_14"] > float(params.get("short_rsi", 70))) | (out["zscore_20"] > float(params.get("short_zscore", 1.5))) | (out["adj_close"] > out["bb_upper_20"])
        long_setup = uptrend & oversold & (out[rs_col] > -0.05)
        short_setup = downtrend & overbought & (out[rs_col] < 0.05)
        out["mean_reversion_pressure"] = (50 - out["rsi_14"].fillna(50)).clip(-50, 50)
        out["technical_score"] = (50 + out["mean_reversion_pressure"] - out["zscore_20"].fillna(0) * 10).clip(0, 100)
        out["relative_strength_score"] = out.groupby("date")[rs_col].rank(pct=True) * 100
        out = attach_context_scores(out, context)
        out["final_score"] = final_score(out)
        out["signal"] = 0.0
        event_allowed = event_entries_allowed(out, params)
        out.loc[long_setup & event_allowed & (out["fundamental_score"].fillna(50) >= float(params.get("long_min_fundamental_score", 40))), "signal"] = 1.0
        out.loc[short_setup & event_allowed & (out["fundamental_score"].fillna(50) <= float(params.get("short_max_fundamental_score", 45))), "signal"] = -1.0
        out["stop_loss_atr"] = float(params.get("stop_loss_atr", 1.5))
        out["trailing_stop_atr"] = float(params.get("trailing_stop_atr", 2.0))
        out["take_profit_r_multiple"] = float(params.get("take_profit_r_multiple", 1.5))
        out["max_holding_days"] = int(params.get("max_holding_days", 15))
        out["score_decomposition"] = out.apply(
            lambda row: (
                f"rsi={row.get('rsi_14', 50):.1f}; z20={row.get('zscore_20', 0):.2f}; "
                f"trend_up={bool(uptrend.loc[row.name])}; trend_down={bool(downtrend.loc[row.name])}; "
                f"fundamental={row.get('fundamental_score', 50):.1f}; event_risk={row.get('event_risk_score', 0):.1f}"
            ),
            axis=1,
        )
        out["reason_for_entry"] = out.apply(_reason, axis=1)
        signals = out[out["technical_score"].notna()][output_columns(out)].sort_values(["date", "final_score"], ascending=[True, False])
        return signals.groupby("date", group_keys=False).head(max_positions * 2).reset_index(drop=True)


def _reason(row: pd.Series) -> str:
    setup = (
        f"rsi={row.get('rsi_14', 50):.1f}, "
        f"z20={row.get('zscore_20', 0):.2f}, "
        f"pressure={row.get('mean_reversion_pressure', 0):.1f}, "
        f"fundamental={row.get('fundamental_score', 50):.1f}, "
        f"event={row.get('event_risk_score', 0):.1f}"
    )
    if row["signal"] > 0:
        return f"Long mean reversion: oversold pullback while longer-term trend remains intact; {setup}."
    if row["signal"] < 0:
        return f"Short mean reversion: overbought bounce inside weak trend; {setup}."
    return f"No mean-reversion position: pullback/overbought setup or risk gate not strong enough; {setup}."
