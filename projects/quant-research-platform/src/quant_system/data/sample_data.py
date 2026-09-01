"""Deterministic sample data generator for first-run demos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_sample_data(
    symbols: list[str],
    start: str,
    end: str | None,
    out_dir: str | Path = "data/sample",
    seed: int = 7,
) -> tuple[Path, Path, Path]:
    """Generate deterministic OHLCV, fundamentals, and events CSV samples."""

    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end or pd.Timestamp.today().normalize())
    if len(dates) < 300:
        dates = pd.bdate_range(end=end or pd.Timestamp.today().normalize(), periods=500)
    ohlcv_rows = []
    fundamental_rows = []
    event_rows = []
    sectors = ["Technology", "Industrials", "Materials", "Utilities"]
    for idx, symbol in enumerate(symbols):
        drift = 0.00015 + idx * 0.00003
        vol = 0.012 + (idx % 4) * 0.002
        shocks = rng.normal(drift, vol, len(dates))
        if symbol in {"NVDA", "AVGO", "VST"}:
            shocks += np.linspace(0.0000, 0.00035, len(dates))
        close = 50 * np.exp(np.cumsum(shocks)) * (1 + idx * 0.03)
        open_ = close * (1 + rng.normal(0, 0.003, len(dates)))
        high = np.maximum(open_, close) * (1 + rng.uniform(0.001, 0.012, len(dates)))
        low = np.minimum(open_, close) * (1 - rng.uniform(0.001, 0.012, len(dates)))
        volume = rng.integers(800_000, 8_000_000, len(dates)) * (1 + idx % 3)
        for date, o, h, l, c, v in zip(dates, open_, high, low, close, volume):
            ohlcv_rows.append(
                {
                    "symbol": symbol,
                    "date": date.date().isoformat(),
                    "open": round(float(o), 4),
                    "high": round(float(h), 4),
                    "low": round(float(l), 4),
                    "close": round(float(c), 4),
                    "adj_close": round(float(c), 4),
                    "volume": int(v),
                }
            )
        for qdate in pd.date_range(dates[0], dates[-1], freq="QS"):
            fundamental_rows.append(
                {
                    "symbol": symbol,
                    "date": qdate.date().isoformat(),
                    "sector": sectors[idx % len(sectors)],
                    "industry": f"Sample Industry {idx % 3}",
                    "revenue_yoy_growth": 0.05 + idx * 0.01 + rng.normal(0, 0.03),
                    "eps_yoy_growth": 0.02 + idx * 0.008 + rng.normal(0, 0.04),
                    "free_cash_flow_growth": 0.03 + rng.normal(0, 0.05),
                    "gross_margin": 0.35 + (idx % 5) * 0.04 + rng.normal(0, 0.02),
                    "operating_margin": 0.12 + (idx % 4) * 0.03 + rng.normal(0, 0.02),
                    "net_debt_to_ebitda": 2.0 - (idx % 4) * 0.25 + rng.normal(0, 0.2),
                    "interest_coverage": 4.0 + (idx % 5) + rng.normal(0, 0.4),
                    "price_to_sales": 2.5 + (idx % 5) * 0.7 + rng.normal(0, 0.3),
                    "ev_to_ebitda": 12 + (idx % 4) * 2 + rng.normal(0, 1.0),
                    "eps_estimate_revision_30d": rng.normal(0.0, 0.03),
                    "revenue_estimate_revision_30d": rng.normal(0.0, 0.02),
                }
            )
        for edate in pd.date_range(dates[60], dates[-1], freq="90D"):
            event_rows.append(
                {
                    "symbol": symbol,
                    "event_date": edate.date().isoformat(),
                    "known_date": (edate - pd.Timedelta(days=30)).date().isoformat(),
                    "event_type": "earnings",
                    "event_risk_score": 35 + (idx % 4) * 10,
                    "expected_move": 0.05 + (idx % 3) * 0.02,
                }
            )
    ohlcv_path = output / "ohlcv.csv"
    fundamentals_path = output / "fundamentals.csv"
    events_path = output / "events.csv"
    pd.DataFrame(ohlcv_rows).to_csv(ohlcv_path, index=False)
    pd.DataFrame(fundamental_rows).to_csv(fundamentals_path, index=False)
    pd.DataFrame(event_rows).to_csv(events_path, index=False)
    return ohlcv_path, fundamentals_path, events_path
