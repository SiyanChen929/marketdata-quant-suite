"""Strict daily backtest engine with next-session execution."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from quant_system.backtest.costs import apply_slippage_and_impact, commission, daily_short_borrow_cost, total_execution_cost_bps
from quant_system.config import AppConfig
from quant_system.portfolio.risk import drawdown_exposure_multiplier


@dataclass
class BacktestResult:
    """Backtest result frames."""

    equity_curve: pd.DataFrame
    positions: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    targets: pd.DataFrame | None = None
    prices: pd.DataFrame | None = None
    risk_overlay: pd.DataFrame | None = None


class DailyBacktestEngine:
    """Daily event-style engine.

    Target weights dated t are executed at t+1 open by default. This prevents
    the common error of using close-generated signals at the same close.
    """

    def __init__(self, prices: pd.DataFrame, config: AppConfig, intraday_prices: pd.DataFrame | None = None) -> None:
        self.prices = prices.sort_values(["date", "symbol"]).copy()
        self.prices["date"] = pd.to_datetime(self.prices["date"]).dt.normalize()
        if "adv20_dollars" not in self.prices.columns:
            self.prices["dollar_volume"] = self.prices["close"].astype(float) * self.prices["volume"].astype(float)
            self.prices["adv20_dollars"] = (
                self.prices.groupby("symbol")["dollar_volume"]
                .transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
            )
        self.config = config
        self.price_lookup = self.prices.set_index(["date", "symbol"]).sort_index()
        self.dust_notional = max(1.0, float(self.config.execution.min_trade_notional))
        self.intraday_prices = self._prepare_intraday_prices(intraday_prices)
        self.intraday_by_date = {
            pd.Timestamp(date): frame.sort_values(["datetime", "symbol"])
            for date, frame in self.intraday_prices.groupby("date")
        } if not self.intraday_prices.empty else {}

    def run(self, targets: pd.DataFrame, signals: pd.DataFrame | None = None) -> BacktestResult:
        """Run the backtest."""

        calendar = list(pd.to_datetime(sorted(self.prices["date"].unique())))
        if len(calendar) < 2:
            empty = pd.DataFrame()
            return BacktestResult(empty, empty, empty, signals if signals is not None else empty)
        targets_by_signal_date = {pd.Timestamp(date): frame for date, frame in targets.groupby("date")}
        cash = float(self.config.portfolio.initial_capital)
        equity = cash
        peak = equity
        risk_peak = equity
        current_risk_bucket: str | None = None
        holdings: dict[str, float] = {}
        position_meta: dict[str, dict] = {}
        last_close: dict[str, float] = {}
        rows: list[dict] = []
        trade_rows: list[dict] = []
        position_rows: list[dict] = []
        pending_targets = pd.DataFrame()
        previous_equity = equity
        for idx, date in enumerate(calendar):
            risk_bucket = self._risk_bucket(date)
            if current_risk_bucket is None or risk_bucket != current_risk_bucket:
                current_risk_bucket = risk_bucket
                risk_peak = equity
            previous_close = dict(last_close)
            holdings_start = dict(holdings)
            day = self.prices[self.prices["date"] == date]
            for _, row in day.iterrows():
                last_close[str(row["symbol"])] = float(row["adj_close"])
            day_trades: list[dict] = []
            if idx > 0 and not pending_targets.empty:
                multiplier = drawdown_exposure_multiplier(equity, risk_peak, self.config.risk)
                cash, holdings, position_meta, day_trades = self._execute_targets(date, pending_targets, holdings, position_meta, cash, equity, multiplier)
                trade_rows.extend(trade for trade in day_trades if trade.get("record_trade", True))
            open_equity = self._mark_to_open(date, holdings, cash)
            intraday_trades: list[dict] = []
            cash, holdings, position_meta, intraday_trades = self._apply_intraday_risk_controls(date, holdings, position_meta, cash, open_equity)
            if intraday_trades:
                day_trades.extend(intraday_trades)
                trade_rows.extend(trade for trade in intraday_trades if trade.get("record_trade", True))
            cash, holdings, position_meta = self._remove_close_dust(holdings, position_meta, cash, last_close)
            market_value = sum(qty * last_close.get(symbol, 0.0) for symbol, qty in holdings.items())
            long_value = sum(max(0.0, qty * last_close.get(symbol, 0.0)) for symbol, qty in holdings.items())
            short_value = sum(max(0.0, -qty * last_close.get(symbol, 0.0)) for symbol, qty in holdings.items())
            borrow = daily_short_borrow_cost(short_value, self.config.execution.short_borrow_fee_annual)
            cash -= borrow
            cash_interest = cash * float(self.config.execution.cash_rate_annual) / 252.0
            cash += cash_interest
            equity = cash + market_value
            daily_pnl = equity - previous_equity
            long_pnl, short_pnl = self._split_daily_pnl(previous_close, last_close, holdings_start, day_trades, borrow)
            peak = max(peak, equity)
            risk_peak = max(risk_peak, equity)
            gross = (long_value + short_value) / max(equity, 1e-9)
            net = (long_value - short_value) / max(equity, 1e-9)
            visible_day_trades = [trade for trade in day_trades if trade.get("record_trade", True)]
            turnover = sum(abs(trade["quantity"] * trade["price"]) for trade in visible_day_trades) / max(equity, 1e-9)
            rows.append(
                {
                    "date": date,
                    "equity": equity,
                    "cash": cash,
                    "gross_exposure": gross,
                    "net_exposure": net,
                    "turnover": turnover,
                    "long_pnl": long_pnl,
                    "short_pnl": short_pnl,
                    "borrow_cost": borrow,
                    "cash_interest": cash_interest,
                    "intraday_risk_event_count": len(intraday_trades),
                    "intraday_exit_notional": sum(abs(trade.get("signed_quantity", 0.0) * trade.get("price", 0.0)) for trade in intraday_trades),
                }
            )
            for symbol, qty in holdings.items():
                meta = position_meta.get(symbol, {})
                position_rows.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "quantity": qty,
                        "close": last_close.get(symbol, 0.0),
                        "market_value": qty * last_close.get(symbol, 0.0),
                        "entry_price": meta.get("entry_price"),
                        "holding_days": meta.get("holding_days", 0),
                    }
                )
            previous_equity = equity
            self._update_position_meta_after_close(date, holdings, position_meta, last_close)
            pending_targets = self._merge_pending_targets(
                targets_by_signal_date.get(date, pd.DataFrame()),
                self._stop_exit_targets(date, holdings, position_meta, last_close),
            )
        return BacktestResult(
            equity_curve=pd.DataFrame(rows),
            positions=pd.DataFrame(position_rows),
            trades=pd.DataFrame(trade_rows).drop(columns=["record_trade"], errors="ignore"),
            signals=signals if signals is not None else pd.DataFrame(),
        )

    def _prepare_intraday_prices(self, intraday_prices: pd.DataFrame | None) -> pd.DataFrame:
        """Normalize optional minute/5-minute bars for execution-risk simulation."""

        if intraday_prices is None or intraday_prices.empty:
            return pd.DataFrame()
        required = {"symbol", "datetime", "open", "high", "low", "close"}
        if not required.issubset(intraday_prices.columns):
            return pd.DataFrame()
        out = intraday_prices.copy()
        out["symbol"] = out["symbol"].astype(str).str.upper()
        out["datetime"] = pd.to_datetime(out["datetime"], format="mixed", errors="coerce")
        out["date"] = pd.to_datetime(out.get("date", out["datetime"]), format="mixed", errors="coerce").dt.normalize()
        for column in ("open", "high", "low", "close", "volume"):
            if column not in out:
                out[column] = 0.0
            out[column] = pd.to_numeric(out[column], errors="coerce")
        return (
            out.dropna(subset=["symbol", "datetime", "date", "open", "high", "low", "close"])
            .drop_duplicates(["symbol", "datetime"], keep="last")
            .sort_values(["date", "datetime", "symbol"])
        )

    def _remove_close_dust(
        self,
        holdings: dict[str, float],
        position_meta: dict[str, dict],
        cash: float,
        last_close: dict[str, float],
    ) -> tuple[float, dict[str, float], dict[str, dict]]:
        """Drop sub-dollar residual positions at close without changing equity."""

        new_holdings = dict(holdings)
        for symbol, qty in list(new_holdings.items()):
            market_value = qty * last_close.get(symbol, 0.0)
            if 0 < abs(market_value) < self.dust_notional:
                cash += market_value
                new_holdings.pop(symbol, None)
                position_meta.pop(symbol, None)
        return cash, new_holdings, position_meta

    def _mark_to_open(self, date: pd.Timestamp, holdings: dict[str, float], cash: float) -> float:
        """Mark the book at the session open before intraday controls run."""

        value = cash
        for symbol, qty in holdings.items():
            try:
                price = float(self.price_lookup.loc[(date, symbol), "open"])
            except (KeyError, TypeError, ValueError):
                continue
            value += qty * price
        return value

    def _apply_intraday_risk_controls(
        self,
        date: pd.Timestamp,
        holdings: dict[str, float],
        position_meta: dict[str, dict],
        cash: float,
        open_equity: float,
    ) -> tuple[float, dict[str, float], dict[str, dict], list[dict]]:
        """Use minute bars to exit positions during the same trading day."""

        if not self.config.risk.intraday_enabled or not self.intraday_by_date or not holdings:
            return cash, holdings, position_meta, []
        day = self.intraday_by_date.get(pd.Timestamp(date).normalize())
        if day is None or day.empty:
            return cash, holdings, position_meta, []
        new_holdings = dict(holdings)
        trades: list[dict] = []
        first_open = day.sort_values("datetime").drop_duplicates("symbol").set_index("symbol")["open"].to_dict()
        stopped_symbols: set[str] = set()
        portfolio_reduced = False
        minute_groups = list(day.groupby("datetime", sort=True))
        for group_idx, (timestamp, minute) in enumerate(minute_groups):
            active_symbols = set(new_holdings)
            if not active_symbols:
                break
            minute = minute[minute["symbol"].isin(active_symbols)]
            if minute.empty:
                continue
            execution_timestamp, execution_minute, execution_price_col, execution_label = self._intraday_execution_bar(
                group_idx,
                timestamp,
                minute,
                minute_groups,
                active_symbols,
            )
            if execution_minute.empty:
                continue
            marks = minute.set_index("symbol")["close"].to_dict()
            equity_now = cash + sum(qty * marks.get(symbol, first_open.get(symbol, 0.0)) for symbol, qty in new_holdings.items())
            portfolio_drawdown = equity_now / max(open_equity, 1e-9) - 1.0
            breadth_down = minute.apply(
                lambda row: float(row["close"]) / max(float(first_open.get(row["symbol"], row["open"])), 1e-9) - 1.0 < 0,
                axis=1,
            ).mean()
            if (
                not portfolio_reduced
                and portfolio_drawdown <= -float(self.config.risk.intraday_portfolio_drawdown_limit)
            ):
                fraction = float(self.config.risk.intraday_portfolio_exit_fraction)
                cash, new_holdings, position_meta, new_trades = self._execute_intraday_fractional_exit(
                    pd.Timestamp(execution_timestamp),
                    execution_minute,
                    new_holdings,
                    position_meta,
                    cash,
                    fraction,
                    f"Intraday portfolio drawdown {portfolio_drawdown:.2%} breached limit; reduced exposure {execution_label}.",
                    price_col=execution_price_col,
                )
                trades.extend(new_trades)
                portfolio_reduced = True
                continue
            if (
                not portfolio_reduced
                and breadth_down >= float(self.config.risk.intraday_breadth_down_pct)
                and portfolio_drawdown < 0
            ):
                fraction = float(self.config.risk.intraday_breadth_exit_fraction)
                cash, new_holdings, position_meta, new_trades = self._execute_intraday_fractional_exit(
                    pd.Timestamp(execution_timestamp),
                    execution_minute,
                    new_holdings,
                    position_meta,
                    cash,
                    fraction,
                    f"Intraday breadth stress {breadth_down:.0%} of held names below open; reduced exposure {execution_label}.",
                    price_col=execution_price_col,
                )
                trades.extend(new_trades)
                portfolio_reduced = True
                continue
            stop_pct = float(self.config.risk.intraday_symbol_stop_loss_pct)
            for _, row in minute.iterrows():
                symbol = str(row["symbol"])
                if symbol not in new_holdings or symbol in stopped_symbols:
                    continue
                qty = float(new_holdings[symbol])
                session_open = float(first_open.get(symbol, row["open"]) or row["open"])
                move_from_open = float(row["close"]) / max(session_open, 1e-9) - 1.0
                stop_hit = (qty > 0 and move_from_open <= -stop_pct) or (qty < 0 and move_from_open >= stop_pct)
                if not stop_hit:
                    continue
                exec_row = execution_minute[execution_minute["symbol"].astype(str).eq(symbol)]
                if exec_row.empty:
                    continue
                cash, new_holdings, position_meta, trade = self._execute_intraday_close(
                    pd.Timestamp(execution_timestamp),
                    symbol,
                    float(exec_row.iloc[0][execution_price_col]),
                    new_holdings,
                    position_meta,
                    cash,
                    f"Intraday symbol stop hit: move from open {move_from_open:.2%}; exited {execution_label}.",
                )
                if trade:
                    trades.append(trade)
                    stopped_symbols.add(symbol)
        return cash, new_holdings, position_meta, trades

    def _intraday_execution_bar(
        self,
        group_idx: int,
        timestamp: pd.Timestamp,
        minute: pd.DataFrame,
        minute_groups: list[tuple[pd.Timestamp, pd.DataFrame]],
        active_symbols: set[str],
    ) -> tuple[pd.Timestamp, pd.DataFrame, str, str]:
        """Return the execution bar for intraday risk controls."""

        policy = str(getattr(self.config.risk, "intraday_fill_policy", "same_bar_close") or "same_bar_close").lower()
        if policy in {"next_bar_open", "next_open", "next_bar"}:
            if group_idx + 1 >= len(minute_groups):
                return pd.Timestamp(timestamp), pd.DataFrame(), "open", "next bar open"
            execution_timestamp, execution_minute = minute_groups[group_idx + 1]
            execution_minute = execution_minute[execution_minute["symbol"].isin(active_symbols)]
            return pd.Timestamp(execution_timestamp), execution_minute, "open", "next bar open"
        return pd.Timestamp(timestamp), minute.copy(), "close", "same bar close"

    def _execute_intraday_fractional_exit(
        self,
        timestamp: pd.Timestamp,
        minute: pd.DataFrame,
        holdings: dict[str, float],
        position_meta: dict[str, dict],
        cash: float,
        fraction: float,
        reason: str,
        price_col: str = "close",
    ) -> tuple[float, dict[str, float], dict[str, dict], list[dict]]:
        trades: list[dict] = []
        new_holdings = dict(holdings)
        price_col = price_col if price_col in minute.columns else "close"
        price_map = minute.set_index("symbol")[price_col].to_dict()
        for symbol, current_qty in list(new_holdings.items()):
            price = price_map.get(symbol)
            if price is None or abs(current_qty) <= 1e-12:
                continue
            exit_qty = -current_qty * max(0.0, min(1.0, fraction))
            cash, new_holdings, position_meta, trade = self._execute_intraday_trade(
                timestamp, symbol, float(price), exit_qty, new_holdings, position_meta, cash, reason
            )
            if trade:
                trades.append(trade)
        return cash, new_holdings, position_meta, trades

    def _execute_intraday_close(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        price: float,
        holdings: dict[str, float],
        position_meta: dict[str, dict],
        cash: float,
        reason: str,
    ) -> tuple[float, dict[str, float], dict[str, dict], dict | None]:
        qty = -float(holdings.get(symbol, 0.0))
        return self._execute_intraday_trade(timestamp, symbol, price, qty, holdings, position_meta, cash, reason)

    def _execute_intraday_trade(
        self,
        timestamp: pd.Timestamp,
        symbol: str,
        raw_price: float,
        signed_qty: float,
        holdings: dict[str, float],
        position_meta: dict[str, dict],
        cash: float,
        reason: str,
    ) -> tuple[float, dict[str, float], dict[str, dict], dict | None]:
        if abs(signed_qty) <= 1e-12:
            return cash, holdings, position_meta, None
        fill = apply_slippage_and_impact(
            raw_price,
            signed_qty,
            self.config.execution.slippage_bps,
            abs(signed_qty * raw_price),
            None,
            self.config.execution.market_impact_bps_per_1pct_adv,
            self.config.execution.spread_bps,
            self.config.execution.market_impact_exponent,
            self.config.execution.min_liquidity_cost_bps,
        )
        fee = commission(signed_qty, self.config.execution.commission_per_share)
        before = float(holdings.get(symbol, 0.0))
        cash -= signed_qty * fill + fee
        after = before + signed_qty
        if abs(after * fill) < self.dust_notional:
            cash += after * fill
            after = 0.0
        new_holdings = dict(holdings)
        if abs(after) <= 1e-8:
            new_holdings.pop(symbol, None)
            position_meta.pop(symbol, None)
        else:
            new_holdings[symbol] = after
        side = "buy" if signed_qty > 0 else "sell"
        execution_cost_bps = total_execution_cost_bps(
            self.config.execution.slippage_bps,
            abs(signed_qty * raw_price),
            None,
            self.config.execution.market_impact_bps_per_1pct_adv,
            self.config.execution.spread_bps,
            self.config.execution.market_impact_exponent,
            self.config.execution.min_liquidity_cost_bps,
        )
        trade = {
            "date": pd.Timestamp(timestamp).normalize(),
            "datetime": timestamp,
            "symbol": symbol,
            "side": side,
            "quantity": abs(signed_qty),
            "signed_quantity": signed_qty,
            "price": fill,
            "fee": fee,
            "position_quantity_before": before,
            "position_quantity_after": after,
            "target_weight": 0.0 if abs(after) <= 1e-8 else pd.NA,
            "reason_for_entry": "",
            "reason_for_exit": reason,
            "intraday_risk_exit": True,
            "execution_cost_bps": execution_cost_bps,
            "estimated_liquidity_cost": abs(signed_qty * raw_price) * execution_cost_bps / 10_000.0,
            "record_trade": abs(signed_qty * fill) >= self.dust_notional,
        }
        return cash, new_holdings, position_meta, trade

    def _risk_bucket(self, date: pd.Timestamp) -> str:
        """Return the drawdown-control reset bucket for a date."""

        mode = self.config.risk.drawdown_reset
        if mode == "never":
            return "all"
        if mode == "monthly":
            return pd.Timestamp(date).strftime("%Y-%m")
        if mode == "quarterly":
            ts = pd.Timestamp(date)
            return f"{ts.year}-Q{((ts.month - 1) // 3) + 1}"
        return str(pd.Timestamp(date).year)

    def _execute_targets(
        self,
        execution_date: pd.Timestamp,
        targets: pd.DataFrame,
        holdings: dict[str, float],
        position_meta: dict[str, dict],
        cash: float,
        equity: float,
        exposure_multiplier: float,
    ) -> tuple[float, dict[str, float], dict[str, dict], list[dict]]:
        target_map = {
            str(row["symbol"]): float(row["target_weight"]) * exposure_multiplier
            for _, row in targets.iterrows()
        }
        all_symbols = sorted(set(holdings) | set(target_map))
        trades: list[dict] = []
        new_holdings = dict(holdings)
        used_notional = 0.0
        daily_turnover_budget = (
            equity * float(self.config.execution.daily_turnover_cap)
            if self.config.execution.daily_turnover_cap and self.config.execution.daily_turnover_cap > 0
            else float("inf")
        )
        for symbol in all_symbols:
            try:
                raw_open = float(self.price_lookup.loc[(execution_date, symbol), "open"])
            except KeyError:
                continue
            target_weight = target_map.get(symbol, 0.0)
            source = targets[targets["symbol"] == symbol].iloc[0].to_dict() if symbol in set(targets["symbol"]) else {}
            current_qty = new_holdings.get(symbol, 0.0)
            current_value = current_qty * raw_open
            target_weight = self._apply_min_holding_constraint(symbol, target_weight, current_value, equity, position_meta, source)
            target_value = equity * target_weight
            delta_value = target_value - current_value
            closing_existing_position = abs(target_weight) <= 1e-12 and abs(current_qty) > 1e-12
            if abs(delta_value) < self.dust_notional and not closing_existing_position:
                continue
            delta_value, liquidity = self._cap_delta_value(execution_date, symbol, delta_value, raw_open, equity, daily_turnover_budget - used_notional, closing_existing_position)
            if abs(delta_value) < self.dust_notional and not closing_existing_position:
                continue
            used_notional += abs(delta_value)
            full_close = closing_existing_position and abs(abs(delta_value) - abs(current_value)) < self.dust_notional
            estimated_qty = -current_qty if full_close else delta_value / raw_open
            fill = apply_slippage_and_impact(
                raw_open,
                estimated_qty,
                self.config.execution.slippage_bps,
                abs(delta_value),
                liquidity.get("adv20_dollars"),
                self.config.execution.market_impact_bps_per_1pct_adv,
                self.config.execution.spread_bps,
                self.config.execution.market_impact_exponent,
                self.config.execution.min_liquidity_cost_bps,
            )
            execution_cost_bps = total_execution_cost_bps(
                self.config.execution.slippage_bps,
                abs(delta_value),
                liquidity.get("adv20_dollars"),
                self.config.execution.market_impact_bps_per_1pct_adv,
                self.config.execution.spread_bps,
                self.config.execution.market_impact_exponent,
                self.config.execution.min_liquidity_cost_bps,
            )
            qty = -current_qty if full_close else delta_value / fill
            fee = commission(qty, self.config.execution.commission_per_share)
            cash -= qty * fill + fee
            new_qty = current_qty + qty
            if 0 < abs(new_qty * fill) < self.dust_notional:
                cash += new_qty * fill
                new_qty = 0.0
            if abs(new_qty) < 1e-8:
                new_holdings.pop(symbol, None)
                position_meta.pop(symbol, None)
            else:
                new_holdings[symbol] = new_qty
                reset_meta = symbol not in position_meta or current_qty == 0 or current_qty * new_qty < 0
                self._set_position_meta(symbol, fill, source, position_meta, execution_date, reset_meta)
            side = "buy" if qty > 0 else "sell"
            notional = abs(qty * fill)
            trades.append(
                {
                    "date": execution_date,
                    "symbol": symbol,
                    "side": side,
                    "quantity": abs(qty),
                    "signed_quantity": qty,
                    "price": fill,
                    "fee": fee,
                    "position_quantity_before": current_qty,
                    "position_quantity_after": new_qty,
                    "target_weight": target_weight,
                    "adv20_dollars": liquidity.get("adv20_dollars"),
                    "adv_participation": liquidity.get("adv_participation"),
                    "execution_cost_bps": execution_cost_bps,
                    "estimated_liquidity_cost": abs(delta_value) * execution_cost_bps / 10_000.0,
                    "capacity_limited": liquidity.get("capacity_limited", False),
                    "turnover_limited": liquidity.get("turnover_limited", False),
                    "event_risk_score": source.get("event_risk_score"),
                    "days_to_earnings": source.get("days_to_earnings"),
                    "fundamental_score": source.get("fundamental_score"),
                    "theme_score": source.get("theme_score"),
                    "technical_score": source.get("technical_score"),
                    "final_score": source.get("final_score"),
                    "entry_quality_score": source.get("entry_quality_score"),
                    "reward_risk_estimate": source.get("reward_risk_estimate"),
                    "trade_location_type": source.get("trade_location_type"),
                    "trade_action_intent": source.get("trade_action_intent"),
                    "fresh_entry_allowed": source.get("fresh_entry_allowed"),
                    "trade_location_multiplier": source.get("trade_location_multiplier"),
                    "reason_for_entry": source.get("reason_for_entry", ""),
                    "reason_for_exit": source.get("reason_for_exit", "Target weight reduced or symbol left candidate set" if target_weight == 0 else ""),
                    "record_trade": notional >= self.dust_notional,
                }
            )
        return cash, new_holdings, position_meta, trades

    def _apply_min_holding_constraint(
        self,
        symbol: str,
        target_weight: float,
        current_value: float,
        equity: float,
        position_meta: dict[str, dict],
        source: dict,
    ) -> float:
        """Avoid ordinary churn before minimum holding period expires."""

        min_days = int(self.config.risk.min_holding_days or 0)
        if min_days <= 0 or symbol not in position_meta:
            return target_weight
        if source.get("reason_for_exit"):
            return target_weight
        current_weight = current_value / max(equity, 1e-9)
        if abs(current_weight) <= 1e-12:
            return target_weight
        holding_days = int(position_meta.get(symbol, {}).get("holding_days", 0) or 0)
        if holding_days >= min_days:
            return target_weight
        reducing_or_closing = abs(target_weight) < abs(current_weight) and target_weight * current_weight >= 0
        if reducing_or_closing:
            return current_weight
        return target_weight

    def _cap_delta_value(
        self,
        execution_date: pd.Timestamp,
        symbol: str,
        delta_value: float,
        raw_open: float,
        equity: float,
        remaining_turnover_budget: float,
        closing_existing_position: bool,
    ) -> tuple[float, dict]:
        """Apply ADV participation and daily turnover limits."""

        try:
            adv20 = float(self.price_lookup.loc[(execution_date, symbol), "adv20_dollars"])
        except (KeyError, TypeError, ValueError):
            adv20 = 0.0
        max_by_adv = adv20 * float(self.config.execution.max_adv_participation) if adv20 > 0 and self.config.execution.max_adv_participation > 0 else float("inf")
        max_by_turnover = max(0.0, remaining_turnover_budget)
        max_notional = min(max_by_adv, max_by_turnover)
        original = abs(delta_value)
        if original > max_notional and max_notional >= self.dust_notional:
            delta_value = (1.0 if delta_value > 0 else -1.0) * max_notional
        elif original > max_notional and not closing_existing_position:
            delta_value = 0.0
        trade_notional = abs(delta_value)
        return delta_value, {
            "adv20_dollars": adv20,
            "adv_participation": trade_notional / adv20 if adv20 > 0 else pd.NA,
            "capacity_limited": original > max_by_adv,
            "turnover_limited": original > max_by_turnover,
        }

    def _split_daily_pnl(
        self,
        previous_close: dict[str, float],
        current_close: dict[str, float],
        holdings_start: dict[str, float],
        day_trades: list[dict],
        borrow: float,
    ) -> tuple[float, float]:
        """Split daily PnL by the side that carried the exposure."""

        long_pnl = 0.0
        short_pnl = 0.0
        for symbol, qty in holdings_start.items():
            prev = previous_close.get(symbol)
            close = current_close.get(symbol)
            if prev is None or close is None:
                continue
            pnl = qty * (close - prev)
            if qty >= 0:
                long_pnl += pnl
            else:
                short_pnl += pnl
        for trade in day_trades:
            close = current_close.get(str(trade["symbol"]))
            if close is None:
                continue
            components = self._trade_pnl_components(
                float(trade.get("position_quantity_before", 0.0) or 0.0),
                float(trade.get("position_quantity_after", 0.0) or 0.0),
                float(trade["signed_quantity"]),
            )
            total_component_qty = sum(abs(qty) for _, qty in components) or abs(float(trade["signed_quantity"])) or 1.0
            for side, qty in components:
                pnl = qty * (close - float(trade["price"]))
                pnl -= float(trade["fee"]) * abs(qty) / total_component_qty
                if side == "long":
                    long_pnl += pnl
                else:
                    short_pnl += pnl
        short_pnl -= borrow
        return long_pnl, short_pnl

    @staticmethod
    def _trade_pnl_components(before: float, after: float, qty: float) -> list[tuple[str, float]]:
        """Return signed intraday trade quantities attributed to long/short books."""

        if abs(qty) <= 1e-12:
            return []
        if abs(before) <= 1e-12 or before * qty >= 0:
            side = "long" if (after if abs(after) > 1e-12 else qty) >= 0 else "short"
            return [(side, qty)]
        if before * after >= 0:
            return [("long" if before > 0 else "short", qty)]
        closing_qty = -before
        opening_qty = after
        return [
            ("long" if before > 0 else "short", closing_qty),
            ("long" if after > 0 else "short", opening_qty),
        ]

    @staticmethod
    def _merge_pending_targets(signal_targets: pd.DataFrame, risk_targets: pd.DataFrame) -> pd.DataFrame:
        """Merge strategy targets and risk exits, with risk exits taking priority."""

        if signal_targets.empty:
            return risk_targets
        if risk_targets.empty:
            return signal_targets
        combined = pd.concat([signal_targets, risk_targets], ignore_index=True)
        combined["_risk_priority"] = combined["reason_for_exit"].fillna("").ne("").astype(int) if "reason_for_exit" in combined.columns else 0
        combined = combined.sort_values("_risk_priority").drop_duplicates("symbol", keep="last")
        return combined.drop(columns=["_risk_priority"])

    def _set_position_meta(
        self,
        symbol: str,
        fill: float,
        source: dict,
        position_meta: dict[str, dict],
        execution_date: pd.Timestamp,
        reset_meta: bool,
    ) -> None:
        """Initialize or update stop metadata after a trade."""

        if reset_meta:
            position_meta[symbol] = {
                "entry_price": fill,
                "entry_date": execution_date,
                "high_water": fill,
                "low_water": fill,
                "holding_days": 0,
            }
        meta = position_meta[symbol]
        for key in ("stop_loss_atr", "trailing_stop_atr", "take_profit_r_multiple", "max_holding_days"):
            if key in source and pd.notna(source[key]):
                meta[key] = float(source[key])
        try:
            atr = float(self.price_lookup.loc[(execution_date, symbol), "atr_14"])
        except (KeyError, TypeError):
            atr = 0.0
        meta["entry_atr"] = atr if pd.notna(atr) else 0.0

    def _update_position_meta_after_close(
        self,
        date: pd.Timestamp,
        holdings: dict[str, float],
        position_meta: dict[str, dict],
        last_close: dict[str, float],
    ) -> None:
        for symbol, qty in holdings.items():
            meta = position_meta.setdefault(symbol, {"entry_price": last_close.get(symbol, 0.0), "entry_date": date})
            close = last_close.get(symbol, 0.0)
            meta["high_water"] = max(float(meta.get("high_water", close)), close)
            meta["low_water"] = min(float(meta.get("low_water", close)), close)
            entry_date = pd.Timestamp(meta.get("entry_date", date))
            meta["holding_days"] = max(0, int((pd.Timestamp(date) - entry_date).days))

    def _stop_exit_targets(
        self,
        date: pd.Timestamp,
        holdings: dict[str, float],
        position_meta: dict[str, dict],
        last_close: dict[str, float],
    ) -> pd.DataFrame:
        rows: list[dict] = []
        for symbol, qty in holdings.items():
            meta = position_meta.get(symbol, {})
            close = last_close.get(symbol)
            if close is None:
                continue
            atr = float(meta.get("entry_atr", 0.0) or 0.0)
            if atr <= 0:
                try:
                    raw_atr = float(self.price_lookup.loc[(date, symbol), "atr_14"])
                    atr = raw_atr if pd.notna(raw_atr) else 0.0
                except (KeyError, TypeError):
                    atr = 0.0
            reason = ""
            entry = float(meta.get("entry_price", close))
            stop_mult = float(meta.get("stop_loss_atr", self.config.risk.atr_stop_multiple))
            trail_mult = float(meta.get("trailing_stop_atr", self.config.risk.trailing_stop_atr_multiple))
            take_profit_r = float(meta.get("take_profit_r_multiple", 0.0) or 0.0)
            max_days = int(meta.get("max_holding_days", 0) or 0)
            if qty > 0:
                hard_stop = entry - stop_mult * atr
                trailing_stop = float(meta.get("high_water", close)) - trail_mult * atr
                profit_target = entry + take_profit_r * stop_mult * atr if take_profit_r > 0 else None
                if atr > 0 and close <= hard_stop:
                    reason = "ATR stop loss triggered after close; exit next open."
                elif atr > 0 and close <= trailing_stop:
                    reason = "ATR trailing stop triggered after close; exit next open."
                elif profit_target is not None and close >= profit_target:
                    reason = "Take-profit R multiple reached after close; exit next open."
            else:
                hard_stop = entry + stop_mult * atr
                trailing_stop = float(meta.get("low_water", close)) + trail_mult * atr
                profit_target = entry - take_profit_r * stop_mult * atr if take_profit_r > 0 else None
                if atr > 0 and close >= hard_stop:
                    reason = "Short ATR stop loss triggered after close; cover next open."
                elif atr > 0 and close >= trailing_stop:
                    reason = "Short ATR trailing stop triggered after close; cover next open."
                elif profit_target is not None and close <= profit_target:
                    reason = "Short take-profit R multiple reached after close; cover next open."
            if not reason and max_days > 0 and int(meta.get("holding_days", 0)) >= max_days:
                reason = "Time stop reached; exit next open."
            if reason:
                rows.append({"date": date, "symbol": symbol, "target_weight": 0.0, "reason_for_exit": reason})
        return pd.DataFrame(rows)
