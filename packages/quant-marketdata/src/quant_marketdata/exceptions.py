"""Typed failures exposed by :mod:`quant_marketdata`."""

from __future__ import annotations


class QuantMarketDataError(RuntimeError):
    """Base exception for the package."""


class CredentialError(QuantMarketDataError):
    """A required environment-only credential is unavailable."""


class DataContractError(QuantMarketDataError, ValueError):
    """Market bars violate the canonical data contract."""


class CacheIntegrityError(QuantMarketDataError):
    """A cached object or canonical partition fails checksum validation."""


class ProviderError(QuantMarketDataError):
    """The upstream provider returned an invalid or unsuccessful response."""


class DataUnavailableError(ProviderError):
    """The requested provider data are not currently available."""
