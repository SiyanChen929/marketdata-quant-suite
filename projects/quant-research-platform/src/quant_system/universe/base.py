"""Base universe construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from quant_system.config import UniverseConfig


@dataclass(frozen=True)
class UniverseResult:
    """Universe symbols and warnings."""

    symbols: list[str]
    warnings: list[str]


def build_base_universe(config: UniverseConfig, provider_symbols: list[str] | None = None) -> UniverseResult:
    """Build the base universe from provider symbols, ETFs, or custom watchlists."""

    warnings: list[str] = []
    provider_symbols = [symbol.upper() for symbol in provider_symbols or []]
    file_symbols = _load_base_symbols(config.base_symbols_path)
    custom = list(config.custom_symbols or config.symbols)
    custom = [symbol.upper() for symbol in custom]
    if config.strict_backtest and config.custom_only:
        raise ValueError(
            "Strict backtest mode forbids custom_only=true because it turns a current watchlist/theme basket into "
            "the historical investment universe. Set universe.custom_only=false and provide universe.base_symbols_path "
            "or a point-in-time security master."
        )
    if config.custom_only:
        symbols = custom
        warnings.append("This backtest uses a custom/thematic universe and may suffer from selection bias.")
    elif file_symbols:
        symbols = sorted(set(file_symbols) | set(custom))
        warnings.append(
            f"Base universe loaded from {config.base_symbols_path}; custom/thematic symbols are treated as overlay, not the sole universe."
        )
        warnings.append("Static base universe file is not a historical constituents database; full-period metrics may still suffer from survivorship bias.")
    elif config.base in {"custom_watchlist", "thematic_baskets"}:
        symbols = sorted(set(provider_symbols) | set(custom))
        warnings.append("Custom/thematic symbols are used only as overlay unless custom_only=true.")
    elif provider_symbols:
        symbols = provider_symbols
    else:
        symbols = custom
        warnings.append("Provider universe unavailable; falling back to configured sample watchlist.")
    if config.base == "us_equities":
        warnings.append("Historical constituents were not supplied; report metrics may suffer from survivorship bias.")
    if config.strict_backtest and not (file_symbols or provider_symbols):
        warnings.append("Strict backtest requested but no provider/security-master base universe was available; custom fallback is exploratory only.")
    if not symbols:
        raise ValueError("Universe is empty. Add CSV data or custom_symbols.")
    return UniverseResult(sorted(set(symbols)), warnings)


def _load_base_symbols(path: str | None) -> list[str]:
    """Load a broad base universe symbol file, one ticker per line or a symbol column CSV."""

    if not path:
        return []
    file_path = Path(path)
    if not file_path.exists():
        return []
    try:
        if file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path)
            column = "symbol" if "symbol" in frame.columns else frame.columns[0]
            values = frame[column].dropna().astype(str)
        else:
            values = pd.Series(file_path.read_text(encoding="utf-8").splitlines())
    except Exception:
        return []
    return sorted({value.strip().upper() for value in values if value.strip() and not value.strip().startswith("#")})
