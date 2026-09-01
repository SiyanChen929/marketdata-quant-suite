"""Broker adapter interfaces for live and paper trading."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BrokerAccount:
    """Normalized account snapshot."""

    account_id: str
    equity: float
    cash: float
    buying_power: float
    mode: str = "paper"


@dataclass(frozen=True)
class BrokerOrderResult:
    """Normalized order submission result."""

    client_order_id: str
    broker_order_id: str
    status: str
    submitted_quantity: float
    submitted_notional: float
    message: str = ""


class BrokerAdapter(ABC):
    """Minimal broker interface used by the live execution layer."""

    @abstractmethod
    def account(self) -> BrokerAccount:
        """Return an account snapshot."""

    @abstractmethod
    def positions(self) -> pd.DataFrame:
        """Return open positions with symbol, quantity, market_value, and weight."""

    @abstractmethod
    def submit_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        """Submit normalized order tickets and return broker statuses."""


class UnconfiguredLiveBroker(BrokerAdapter):
    """Placeholder for a real broker until credentials and endpoints are wired."""

    def __init__(self, broker_name: str) -> None:
        self.broker_name = broker_name

    def account(self) -> BrokerAccount:
        raise RuntimeError(f"Real broker adapter '{self.broker_name}' is not configured. Use broker=paper first.")

    def positions(self) -> pd.DataFrame:
        raise RuntimeError(f"Real broker adapter '{self.broker_name}' is not configured. Use broker=paper first.")

    def submit_orders(self, orders: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError(f"Real broker adapter '{self.broker_name}' is not configured. Use broker=paper first.")
