# Data contract

## One logical data plane

Use one external `QUANT_DATA_HOME` for all projects:

```text
quant-data/
  raw/marketdata/
    confirmed/request-cache/   checksum-verified completed-session responses
    provisional/request-cache/ checksum-verified partial-session responses
    options/request-cache/     option-chain reference snapshots
  lake/confirmed/
    bars/resolution=D/year=YYYY/bars.parquet
    quant_system/              optional derived research datasets
  staging/provisional/
    bars/resolution=5/year=YYYY/bars.parquet
  manifests/marketdata/
    confirmed.json
    provisional.json
```

Repository code and sample fixtures live in Git. Vendor data, credentials, raw caches, full runs, and portfolio positions stay outside Git.

## Finality

| State | Permitted use | Storage |
|---|---|---|
| `confirmed` | formal signals, model fitting, backtests | `lake/confirmed` |
| `provisional` | monitoring and explicitly labeled previews | `staging/provisional` |

`unavailable` is an availability outcome, not a persisted finality value. A provider miss or `404` never justifies manufacturing a formal session.

## Canonical keys

- Canonical bars: `(symbol, date)` within a finality-specific snapshot
- Provider request identity: provider, dataset, symbol, date range, resolution,
  adjustment policy, finality, and schema version
- Fundamentals/events: provider observation plus `known_at`, `effective_at`, and revision/vintage identifiers
- Universe membership: `(universe_id, symbol, valid_from, valid_to, known_at)`

## Required snapshot metadata

- provider and endpoint
- normalized request and fetch timestamp
- symbol mapping and asset class
- adjustment policy
- data state/finality
- minimum and maximum observation times
- row count, schema version, and checksum
- code/config version and parent snapshot IDs

## Retention

- Preserve confirmed canonical bars, unique vintages, manifests, and selected showcase evidence.
- Retain provisional intraday snapshots briefly unless attached to an incident or paper-trading decision.
- MarketData runs rebuild research features from the canonical store and never persist a second raw-price bundle.
- Rebuildable signals and offline-demo research bundles should have bounded retention.
- Prefer one partitioned Parquet representation over simultaneous CSV, pickle, and Parquet mirrors.
- Earnings, event, and fundamental response caches live under `QUANT_DATA_HOME` and pair each CSV with an atomic request/checksum manifest.
