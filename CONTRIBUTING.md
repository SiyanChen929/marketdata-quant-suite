# Contributing

## Before changing research logic

1. State the hypothesis and its decision rule.
2. Identify the point-in-time unit and data availability timestamp.
3. Add a failing test for timing, schema, cost, or risk behavior.
4. Keep selection and final evaluation chronologically separate.
5. Record adverse outcomes and rejected variants.

## Before running an LLM experiment

1. Freeze the research plan and publish its hash outside the repository before the first model call.
2. Keep server-side model fallback off, and record the requested and served model, stop reason and token usage of every call.
3. Record every response so that the run can be replayed offline, and never delete a recording.
4. Count every proposal, including invalid and failed ones, as a trial. On real data, reveal a sealed test window only against a commitment published beforehand.
5. Keep the model SDK in the optional `llm` extra; tests use injected fake clients and never reach the network.

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

Pull requests must not include credentials, vendor data, licensed index inputs, virtual environments, generated run directories, LLM recordings or audit logs that contain vendor prices, or build artifacts such as wheels.
