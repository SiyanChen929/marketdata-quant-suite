# Data-source policy

## One source for prices

MarketData is the sole production source for supported US equity, ETF, and listed-option market prices in this suite. The shared `quant_marketdata` package owns all provider calls and produces a canonical long-form OHLCV frame:

```text
date, symbol, open, high, low, close, volume, source, finality
```

Research modules may transform this frame, pivot it, or load a versioned snapshot. They may not embed a separate Yahoo, Stooq, Alpha Vantage, or bespoke MarketData downloader.

## Finality

| State | Storage | Permitted use |
|---|---|---|
| `confirmed` | `lake/confirmed` | formal research, validation, and published metrics |
| `provisional` | `staging/provisional` | same-day monitoring and diagnostics only |

A provisional observation never overwrites confirmed history. Promotion requires a fresh completed-session download and validation. A `404` or `no_data` response is an availability outcome, not evidence that a market session closed at a fabricated value.

The built-in 16:15 ET daily cutoff is an operational safeguard, not exchange-settlement proof. Production promotion should also verify the trading calendar, provider completeness, and any post-close revisions before publishing a run.

## Adjustment policy

Daily-candle requests must state split and dividend adjustment choices in their manifest. Comparing studies requires the same policy. Event studies that need as-traded prices should request and label that basis explicitly; total-return studies should use fully adjusted inputs consistently.

## Alternative and proprietary inputs

Alternative data may enter only through typed, point-in-time interfaces and may not substitute for prices:

- FRED/ALFRED for macro observations and vintages;
- licensed index announcements for rebalance membership;
- official auction/NBBO and borrow/locate data for execution claims;
- USDA/NOAA for agricultural research;
- China-specific providers for A-share and domestic futures projects.

These inputs must carry `event_date`, `known_date` or vintage metadata, source, and license/provenance fields. Public fixtures must be synthetic or redistributable.

## Repository rule

The repository stores schemas, code, tests, manifests, and tiny synthetic examples. Raw and normalized vendor data live under `QUANT_DATA_HOME` and remain outside Git.

## Provider references

- [MarketData historical stock candles](https://www.marketdata.app/docs/api/stocks/candles/)
- [MarketData option-chain endpoint](https://www.marketdata.app/docs/api/options/chain/)
- [MarketData asset-class coverage](https://www.marketdata.app/data/)
