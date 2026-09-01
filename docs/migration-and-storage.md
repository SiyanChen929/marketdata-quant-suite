# Migration and storage plan

The suite was assembled as a code-first copy. Existing projects and market-data folders remain untouched. No large cache was copied into this repository.

## Safe migration

1. Choose one external `QUANT_DATA_HOME` with enough free space.
2. Inventory and checksum the newest confirmed MarketData daily store.
3. Import it into `lake/confirmed`; preserve its adjustment policy and source cutoff in a manifest.
4. Import same-day snapshots into `staging/provisional` only.
5. Compare old and new datasets on row count, symbols, date coverage, duplicate `(symbol, date)` keys, nulls, price adjustments, and representative backtest results.
6. Run both stores in parallel until the results match within documented tolerances.
7. Point every module at the shared store, then archive or remove redundant caches in a separate, explicitly approved cleanup.

The importer is dry-run by default:

```bash
python scripts/import-bars.py --input /path/to/confirmed_marketdata.csv \
  --finality confirmed --attest-completed-sessions
python scripts/import-bars.py --input /path/to/confirmed_marketdata.csv \
  --finality confirmed --attest-completed-sessions --write
```

It validates the canonical schema, prints a source checksum and coverage summary, and refuses to label filenames containing provisional markers as confirmed. An unlabeled legacy file also needs the explicit completed-session attestation; the importer never infers finality from a plausible filename.

## Retention policy

Keep:

- the canonical confirmed lake;
- bounded raw responses needed for audit and recovery;
- manifests, checksums, schemas, and security-master mappings;
- selected research runs with metrics, trade ledgers, configuration, and charts;
- unique licensed or point-in-time datasets according to their terms.

Regenerate or retire after parity checks:

- duplicate CSV, pickle, and Parquet mirrors of the same bars;
- copied `latest` run directories;
- virtual environments;
- superseded per-symbol shards;
- bulky intermediate factor, signal, and weight panels that a manifest can reproduce.

## Guardrail

Do not delete the current confirmed-close source merely because another lake exists. The replacement becomes canonical only after its manifest and coverage checks prove that it is at least as complete as the source and representative research outputs agree.
