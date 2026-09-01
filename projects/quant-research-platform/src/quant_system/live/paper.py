"""Paper broker used to test the live execution path safely."""

from __future__ import annotations

import pandas as pd

from quant_system.config import LiveTradingConfig
from quant_system.live.broker import BrokerAccount, BrokerAdapter


class PaperBrokerAdapter(BrokerAdapter):
    """In-memory paper broker that accepts order tickets without market impact."""

    def __init__(self, config: LiveTradingConfig) -> None:
        self.config = config
        self._account = BrokerAccount(
            account_id="PAPER",
            equity=float(config.paper_cash),
            cash=float(config.paper_cash),
            buying_power=float(config.paper_cash),
            mode="paper",
        )

    def account(self) -> BrokerAccount:
        return self._account

    def positions(self) -> pd.DataFrame:
        return pd.DataFrame(columns=["symbol", "quantity", "market_value", "weight", "close"])

    def submit_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        if orders.empty:
            return pd.DataFrame()
        data = orders.copy()
        data["broker_order_id"] = data["client_order_id"].astype(str).map(lambda value: f"PAPER-{value}")
        data["broker_status"] = "paper_submitted"
        data["broker_message"] = "Accepted by paper broker; no real order was sent."
        return data[
            [
                "client_order_id",
                "broker_order_id",
                "broker_status",
                "broker_message",
                "symbol",
                "side",
                "quantity",
                "notional",
                "order_type",
                "limit_price",
                "time_in_force",
            ]
        ]
