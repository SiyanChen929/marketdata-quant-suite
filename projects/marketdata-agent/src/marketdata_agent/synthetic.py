"""Deterministic synthetic daily bars for tests and offline benchmarks.

The generator is a one-factor lognormal model on a weekday calendar (no
exchange holidays):

    r[i, t] = mu_i / 252 + beta_i * m[t] + e[i, t],
    m[t] ~ N(0, sigma_m^2 / 252),  e[i, t] ~ N(0, sigma_i^2 / 252).

Opens gap from the prior close, highs/lows bracket the open-close body, and
volumes are lognormal.  Rows are labelled ``source="synthetic"`` and
``finality="confirmed"`` and pass :func:`quant_marketdata.normalize_bars`.
Synthetic data establish plumbing and exact-arithmetic checks only; they carry
no information about real markets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from quant_marketdata import normalize_bars

from .sources import FrameBarSource


@dataclass(frozen=True)
class SyntheticPanelSpec:
    """Parameters that fully determine a synthetic panel."""

    symbols: tuple[str, ...]
    start: str = "2021-01-04"
    end: str = "2023-12-29"
    seed: int = 20240601
    market_vol: float = 0.18
    idio_vol: float = 0.22
    listings: tuple[tuple[str, str], ...] = ()
    source: str = "synthetic"

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe description for run manifests."""

        payload = asdict(self)
        payload["symbols"] = list(self.symbols)
        payload["listings"] = {symbol: first for symbol, first in self.listings}
        return payload


def synthetic_symbols(count: int, prefix: str = "SYN") -> list[str]:
    """Return ``count`` deterministic tickers such as ``SYN01``."""

    if count < 1:
        raise ValueError("count must be positive")
    width = max(2, len(str(count)))
    return [f"{prefix}{index:0{width}d}" for index in range(1, count + 1)]


def synthetic_bars(
    symbols: int | Sequence[str] = 5,
    *,
    start: str = "2021-01-04",
    end: str = "2023-12-29",
    seed: int = 20240601,
    market_vol: float = 0.18,
    idio_vol: float = 0.22,
    listings: Mapping[str, str] | None = None,
    source: str = "synthetic",
) -> pd.DataFrame:
    """Generate a canonical, confirmed, deterministic long-form bar panel.

    ``listings`` maps a symbol to its first trading date; earlier rows are
    dropped, which lets tests exercise late listings (survivorship and
    look-ahead in universe construction).
    """

    spec = SyntheticPanelSpec(
        symbols=tuple(synthetic_symbols(symbols) if isinstance(symbols, int) else [str(s).upper() for s in symbols]),
        start=start,
        end=end,
        seed=seed,
        market_vol=market_vol,
        idio_vol=idio_vol,
        listings=tuple(sorted((str(k).upper(), str(v)) for k, v in (listings or {}).items())),
        source=source,
    )
    return generate_panel(spec)


def generate_panel(spec: SyntheticPanelSpec) -> pd.DataFrame:
    """Generate the panel described by ``spec``."""

    if len(set(spec.symbols)) != len(spec.symbols) or not spec.symbols:
        raise ValueError("synthetic symbols must be unique and non-empty")
    dates = pd.bdate_range(spec.start, spec.end)
    if len(dates) < 2:
        raise ValueError("synthetic panel needs at least two sessions")
    rng = np.random.default_rng(spec.seed)
    n_days, n_symbols = len(dates), len(spec.symbols)

    beta = rng.uniform(0.6, 1.4, n_symbols)
    drift = rng.normal(0.06, 0.05, n_symbols)
    sigma = spec.idio_vol * rng.uniform(0.6, 1.4, n_symbols)
    first_price = rng.uniform(20.0, 300.0, n_symbols)
    volume_scale = rng.uniform(5e5, 5e6, n_symbols)

    market = rng.normal(0.0, spec.market_vol / np.sqrt(252.0), n_days)
    idio = rng.normal(0.0, 1.0, (n_days, n_symbols)) * (sigma / np.sqrt(252.0))
    log_returns = drift / 252.0 + market[:, None] * beta + idio
    close = first_price * np.exp(np.cumsum(log_returns, axis=0))

    daily_vol = np.sqrt((beta * spec.market_vol) ** 2 + sigma**2) / np.sqrt(252.0)
    previous_close = np.vstack([first_price, close[:-1]])
    gap = rng.normal(0.0, 0.25, (n_days, n_symbols)) * daily_vol
    open_ = previous_close * np.exp(gap)
    wick_up = np.abs(rng.normal(0.0, 0.5, (n_days, n_symbols))) * daily_vol
    wick_down = np.abs(rng.normal(0.0, 0.5, (n_days, n_symbols))) * daily_vol
    open_, close = np.round(open_, 4), np.round(close, 4)
    high = np.round(np.maximum(open_, close) * np.exp(wick_up), 4)
    low = np.round(np.minimum(open_, close) * np.exp(-wick_down), 4)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))
    volume = np.round(volume_scale * np.exp(rng.normal(0.0, 0.35, (n_days, n_symbols))))

    frame = pd.DataFrame(
        {
            "date": np.repeat(dates.to_numpy(), n_symbols),
            "symbol": np.tile(np.array(spec.symbols, dtype=object), n_days),
            "open": open_.ravel(),
            "high": high.ravel(),
            "low": low.ravel(),
            "close": close.ravel(),
            "volume": volume.ravel(),
            "source": spec.source,
            "finality": "confirmed",
        }
    )
    for symbol, first_date in spec.listings:
        if symbol not in spec.symbols:
            raise ValueError(f"listing date given for unknown symbol {symbol}")
        frame = frame.loc[~(frame["symbol"].eq(symbol) & frame["date"].lt(pd.Timestamp(first_date)))]
    return normalize_bars(frame, finality="confirmed", source=spec.source)


def synthetic_source(symbols: int | Sequence[str] = 5, **kwargs: object) -> FrameBarSource:
    """Return a :class:`FrameBarSource` over :func:`synthetic_bars`."""

    return FrameBarSource(synthetic_bars(symbols, **kwargs))  # type: ignore[arg-type]
