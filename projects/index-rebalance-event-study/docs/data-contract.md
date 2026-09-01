# Data contract

## Public event schema

Private event parsers must emit a table with these columns:

| Column | Meaning |
|---|---|
| `announcement_date` | Date the information became public |
| `effective_close_date` | Close at which the index change is implemented |
| `segment` | Provider-neutral segment label such as `primary` |
| `action` | `add` or `delete` |
| `security_name` | Point-in-time security name |
| `symbol` | Point-in-time trading symbol |
| `change_type` | Optional migration or entry/exit classification |
| `source_id` | Private provenance identifier, not a redistributed document |
| `is_synthetic` | `true` only for fabricated fixtures |

Production research should use permanent security identifiers and a
point-in-time symbol map. Current ticker strings are insufficient for renamed,
delisted, merged, or multi-class securities.

## Confirmed daily-bar schema

The shared `quant_marketdata` package owns storage and retrieval. This project
accepts exactly the canonical fields:

```text
date, symbol, open, high, low, close, volume, source, finality
```

Formal event-study bars must have `finality=confirmed`. `(date, symbol)` must be
unique, OHLC values must be positive and coherent, and volume must be
non-negative. `QUANT_DATA_HOME` points to the external shared cache.

## Private intraday inputs

Pre-close execution analysis additionally requires user-supplied intraday data
with at least `timestamp, symbol, open, high, low, close, volume`. Production
claims require the relevant official auction and executable quote fields as
well. Continuous minute candles alone cannot establish an auction fill or an
option execution price.

Licensed index notices, constituent histories, pro-forma weights, auction
records, NBBO, options data, and vendor caches must not be committed.
