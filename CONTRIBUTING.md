# Contributing

## Before changing research logic

1. State the hypothesis and its decision rule.
2. Identify the point-in-time unit and data availability timestamp.
3. Add a failing test for timing, schema, cost, or risk behavior.
4. Keep selection and final evaluation chronologically separate.
5. Record adverse outcomes and rejected variants.

## Before adding a data source

- Route supported US market prices through `quant_marketdata`.
- Explain why any specialist source supplies a distinct domain rather than a duplicate price feed.
- Add source, request, adjustment, finality, checksum, and license fields to its manifest.
- Use synthetic fixtures in tests.

## Required checks

```bash
./scripts/test-all.sh
python scripts/repository-audit.py
```

Pull requests must not include credentials, vendor data, licensed index inputs, virtual environments, or generated run directories.
