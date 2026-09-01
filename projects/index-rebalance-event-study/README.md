# Index Rebalance Event Study

`index-rebalance-event-study` is a compact, auditable research template for
studying constituent additions and deletions around index implementation dates.
It focuses on the parts that can be published safely:

- a provider-neutral event schema;
- confirmed daily-bar retrieval through the shared `quant_marketdata` package;
- date-balanced event portfolios, implementation costs, and date-level inference;
- pre-close underlying-volatility diagnostics with explicit execution limits; and
- deterministic synthetic fixtures and offline tests.

The repository contains no index-provider notices, constituent histories,
market-data caches, auction records, NBBO data, PDFs, or historical research
outputs. Every included symbol begins with `ZZZ` and is fictional.

## Research boundary

Licensed index announcements, pro-forma weights, permanent identifiers,
closing-auction records, historical NBBO, borrow information, and options data
are private user-supplied research inputs. A private ingestion adapter should
normalize those inputs to the schemas in [docs/data-contract.md](docs/data-contract.md).
They should remain outside this repository.

Daily candles are loaded only through the suite's shared
`quant_marketdata.MarketDataClient`. The public adapter calls
`get_bulk_daily_bars(..., finality="confirmed")` and uses the canonical columns:

```text
date, symbol, open, high, low, close, volume, source, finality
```

The shared cache is external. Set `QUANT_DATA_HOME` to a directory outside the
checkout and provide `MARKETDATA_TOKEN` only through the environment. The
project does not contain an HTTP downloader or token loader.

## Project map

```text
src/index_rebalance_event_study/
  events.py       Event validation, migration classification, name matching
  marketdata.py   Confirmed-bar adapter for quant_marketdata
  backtest.py     Daily event ledger, portfolio aggregation, date-level metrics
  volatility.py   Underlying path and model-sensitivity diagnostics
data/sample/
  synthetic_events.csv
scripts/
  run_synthetic_demo.py
tests/
```

## Run the offline example

From this project directory inside `marketdata-quant-suite`:

```bash
PYTHONPATH=src:../../packages/quant-marketdata/src \
  python -m pytest -q

PYTHONPATH=src:../../packages/quant-marketdata/src \
  python scripts/run_synthetic_demo.py
```

The demo generates confirmed-schema candles in memory for the fictional `ZZZ*`
symbols. It makes no network request and writes no market data into the repo.

## Load production daily bars

```python
from index_rebalance_event_study.marketdata import ConfirmedDailyBars

data = ConfirmedDailyBars.from_environment()
bars = data.get_bulk_daily_bars(
    symbols=["USER_SUPPLIED_SYMBOL"],
    start="2025-01-01",
    end="2025-03-31",
)
```

`ConfirmedDailyBars` fixes `finality="confirmed"`; provisional sessions cannot
enter the formal event study through this adapter.

## Interpretation limits

- A vendor daily close is not an official closing-auction print.
- A signal entered before imbalance dissemination cannot use later NOII or
  auction information.
- The independent sample is the implementation date, not the number of stocks.
- Underlying OHLCV movement is not executable options P&L. Historical option
  claims require point-in-time quotes/trades, contract definitions, sizes,
  fees, and hedge fills.
- Synthetic demo results test plumbing only and have no investment meaning.

This project is research software, not investment advice or an order-routing
system.
