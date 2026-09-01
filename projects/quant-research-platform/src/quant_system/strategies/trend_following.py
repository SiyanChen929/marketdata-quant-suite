"""Trend following with regime filter."""

from __future__ import annotations

import pandas as pd

from quant_system.strategies.base import Strategy, StrategyContext, attach_context_scores, event_entries_allowed, final_score, output_columns


class TrendFollowingWithRegimeFilter(Strategy):
    """Moving-average trend following with benchmark regime awareness."""

    name = "trend"

    def generate_signals(self, features: pd.DataFrame, context: StrategyContext) -> pd.DataFrame:
        params = self.params
        ma_fast = int(params.get("ma_fast", 50))
        ma_slow = int(params.get("ma_slow", 200))
        max_positions = int(params.get("max_positions", 50))
        fast = f"ma_{ma_fast}"
        slow = f"ma_{ma_slow}"
        out = features.sort_values(["symbol", "date"]).copy()
        bench = out[out["symbol"] == context.benchmark.upper()][["date", "adj_close", "ma_50", "ma_200", "vol_20d"]].rename(
            columns={"adj_close": "bench_close", "ma_50": "bench_ma_50", "ma_200": "bench_ma_200", "vol_20d": "bench_vol_20d"}
        )
        out = out.merge(bench, on="date", how="left")
        risk_on = (out["bench_close"] > out["bench_ma_50"]) & (out["bench_close"] > out["bench_ma_200"]) & (out["bench_vol_20d"].fillna(0) < float(params.get("panic_volatility", 0.035)))
        risk_off = (out["bench_close"] < out["bench_ma_200"]) | (out["bench_vol_20d"].fillna(0) >= float(params.get("panic_volatility", 0.035)))
        long_setup = (out["adj_close"] > out[fast]) & (out[fast] > out[slow]) & risk_on
        short_setup = (out["adj_close"] < out[fast]) & (out[fast] < out[slow]) & (risk_off | (out.get("rs_63d", 0) < 0))
        trend_strength = ((out["adj_close"] / out[slow] - 1.0) * 250).clip(-50, 50)
        adx_score = out["adx_14"].fillna(15).clip(0, 50)
        rs_score = out.groupby("date")[out["rs_63d"].name if "rs_63d" in out.columns else "ret_63d"].rank(pct=True) * 100
        out["technical_score"] = (50 + trend_strength + 0.3 * (adx_score - 20)).clip(0, 100)
        out["relative_strength_score"] = rs_score
        out = attach_context_scores(out, context)
        out["final_score"] = final_score(out)
        out["signal"] = 0.0
        event_allowed = event_entries_allowed(out, params)
        out.loc[long_setup & event_allowed & (out["fundamental_score"].fillna(50) >= float(params.get("long_min_fundamental_score", 40))), "signal"] = 1.0
        out.loc[short_setup & event_allowed & (out["fundamental_score"].fillna(50) <= float(params.get("short_max_fundamental_score", 50))), "signal"] = -1.0
        out["stop_loss_atr"] = float(params.get("stop_loss_atr", 3.0))
        out["trailing_stop_atr"] = float(params.get("trailing_stop_atr", 4.0))
        out["take_profit_r_multiple"] = float(params.get("take_profit_r_multiple", 0.0))
        out["max_holding_days"] = int(params.get("max_holding_days", 120))
        out["trend_price_vs_slow_pct"] = out["adj_close"] / out[slow] - 1.0
        out["trend_adx"] = out["adx_14"]
        out["score_decomposition"] = out.apply(
            lambda row: (
                f"price_vs_slow={row.get('trend_price_vs_slow_pct', 0) if pd.notna(row.get('trend_price_vs_slow_pct', 0)) else 0:.2%}; "
                f"adx={row.get('adx_14', 0):.1f}; risk_on={bool(risk_on.loc[row.name])}; "
                f"fundamental={row.get('fundamental_score', 50):.1f}; event_risk={row.get('event_risk_score', 0):.1f}"
            ),
            axis=1,
        )
        out["reason_for_entry"] = out.apply(_reason, axis=1)
        signals = out[out["technical_score"].notna()][output_columns(out)].sort_values(["date", "final_score"], ascending=[True, False])
        return signals.groupby("date", group_keys=False).head(max_positions * 2).reset_index(drop=True)


def _reason(row: pd.Series) -> str:
    setup = (
        f"price_vs_slow={row.get('trend_price_vs_slow_pct', 0):.1%}, "
        f"adx={row.get('trend_adx', 0):.1f}, "
        f"rs={row.get('relative_strength_score', 50):.1f}, "
        f"fundamental={row.get('fundamental_score', 50):.1f}, "
        f"event={row.get('event_risk_score', 0):.1f}"
    )
    if row["signal"] > 0:
        return f"Long trend: price is aligned above fast/slow averages while benchmark regime is risk-on; {setup}."
    if row["signal"] < 0:
        return f"Short trend: price is below trend stack with risk-off or relative weakness confirmation; {setup}."
    return f"No trend position: trend stack, regime, quality, or event gate not strong enough; {setup}."
