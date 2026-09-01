# Equity Pairs Research

An auditable statistical-arbitrage research module for the unified MarketData
quant suite. It exhaustively screens different-issuer pairs inside each current
S&P 500 sector, controls multiple testing, freezes formation-period parameters,
and evaluates close-to-close trades on a held-out period with turnover and
short-borrow costs.

The portfolio value is the research discipline: every candidate can be traced
from confirmed daily bars through cointegration diagnostics, a causal signal,
stock-leg accounting, a trade ledger, portfolio aggregation, and a rendered
report. The evidence remains honest when the strategy fails.

## What this demonstrates

- Engle-Granger tests in both orientations, Bonferroni correction for the
  orientation choice, and within-sector Benjamini-Hochberg FDR control.
- Integration checks, hedge-ratio constraints, half-life estimation, Hurst
  diagnostics, and explicit fallback labels when strict gates do not fill a
  requested sector quota.
- Five-year formation / three-year held-out evaluation with a one-session signal
  lag, gross-normalized legs, drift rebalancing, transaction costs, borrow carry,
  forced exits, and terminal liquidation.
- Equal-capital and sector-balanced portfolios, cost sensitivity, FRED regime
  attribution, quality checks, SHA-256 manifests, charts, and HTML reporting.
- Additional modules for constrained allocation, rolling pair health, nested
  validation, execution-capacity scenarios, grid search, and exact weighted-book
  accounting.

## Data contract

All production equity-price ingestion goes through the suite package:

```python
from quant_marketdata import MarketDataClient, wide_close

bars = MarketDataClient().get_bulk_daily_bars(
    symbols,
    start,
    end,
    finality="confirmed",
)
closes = wide_close(bars)
```

The research wrapper enforces `finality="confirmed"` again before accepting a
panel. `MARKETDATA_TOKEN` is read by the shared client. `QUANT_DATA_HOME` points
its Parquet store outside the repository, so multiple projects reuse one cache
without committing bulky market data.

FRED is retained only for macro attribution. Its credential is read
from `FRED_API_KEY`; no key-file argument or repository fallback exists. FRED
observations are current-vintage and must not be treated as point-in-time macro
signals without ALFRED release alignment.

## Install and test

From the suite root:

```bash
python -m pip install -e packages/quant-marketdata
python -m pip install -e "projects/equity-pairs-research[dev]"
```

For a source-tree test without installation:

```bash
cd projects/equity-pairs-research
PYTHONPATH=../../packages/quant-marketdata/src:src python -m pytest -q
```

Tests use fakes for MarketData, FRED, and the constituent page; the suite makes
no network request.

## Run

```bash
export MARKETDATA_TOKEN="..."
export QUANT_DATA_HOME="/path/outside/this/repository/marketdata"
export FRED_API_KEY="..."

equity-pairs \
  --as-of 2026-08-26 \
  --config configs/default.json \
  --output-dir outputs/pairs_research_2026-08-26
```

`--refresh-data` refreshes the current constituent snapshot, the shared
MarketData exact-request cache, and FRED. `--recompute` reruns the statistical
screens. Generated data, charts, and reports are ignored by Git.

## Private-study provenance and decision

The figures below are a disclosure from the original private research run, not a
result that can be reproduced or independently verified from this public
repository. Its input snapshots, generated reports, and historical result
artifacts are excluded. The private run dated 2026-08-26 used 503 current
securities, tested 13,917 different-issuer within-sector pairs, and completed
110 held-out pair backtests. Under 5 bps one-way turnover cost and 30 bps annual
short-borrow carry:

- the all-selected portfolio delivered **-0.46% net CAGR** and **-4.62% maximum
  drawdown**;
- only 7 of 110 requested slots passed every strict formation gate;
- a **+12.40%** ex-post winning-ten curve was selection-biased and is not a live
  performance estimate; and
- later production-gate work returned **-1.84%** cash-stripped pairs-alpha CAGR,
  **-3.57%** nested out-of-sample CAGR, and **0/10** eligible pairs at the sample
  end.

Those results support a **research / paper-trade only** decision. Historical
result artifacts and raw data are intentionally excluded from this public
repository; the numbers are disclosed to prevent a polished implementation
from implying a profitable strategy. The offline tests validate research logic
and synthetic fixtures, not these private-run figures.

## Limits

The universe uses current S&P 500 membership and current GICS classifications,
which introduces survivorship and historical-membership bias. Confirmed daily
bars do not provide executable quotes, historical locates, name-specific borrow,
market impact, or capacity. A production evaluation requires point-in-time
constituents, delisted total returns, quote/auction data, borrow and locate
history, release-vintage macro data, rolling re-formation, and paper-trade
reconciliation.

See [`docs/methodology.md`](docs/methodology.md) for equations, assumptions, and
primary references.
