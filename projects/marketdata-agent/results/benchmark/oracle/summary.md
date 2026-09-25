# Benchmark run: `oracle`

> **harness-validation baselines on synthetic data; LLM agent results pending (requires ANTHROPIC_API_KEY)**

scripted reference tool plan with citations (harness validation; expected 100%).

All prices are synthetic (a one-factor lognormal model on a weekday calendar). These numbers validate the harness and say nothing about real markets or about any language model.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/quant-marketdata/src:projects/marketdata-agent/src python projects/marketdata-agent/scripts/run_benchmark.py
```

- Suite: 174 tasks (compute 72, lookup 30, multi_step 24, pit_trap 24, policy_trap 12, unknown_symbol 12), 174 distinct items; seed 20240917; `marketdata-agent/bench-generator/v2`; sha256 `57cd5cb0e19e9ab0`.
- Dataset: synthetic panel, 10 symbols, 2021-01-04 to 2023-12-29, seed 20240601, late listings {'SYN09': '2022-09-01', 'SYN10': '2023-09-01'}.
- Arm: clock enforced; prompt `copilot_system_v1`; tools `default`.
- Valid: yes; tasks scored 174; retried episodes 0.
- Audit chain: verified (1048 records, head `e63c77a5630e7c75`, recorded in `summary.json`; scripted run with fixed audit timestamps, so a rerun must reproduce this head exactly).

## Episode status

| Status | Episodes |
|---|---|
| `completed` | 174 |
| `max_steps` | 0 |
| `refusal` | 0 |
| `backend_error` | 0 |

## Overall

| Agent | Tasks | Accuracy [95% CI] | Grounding (claims) | Citation | Look-ahead attempt episodes | Leak episodes | Denied-call episodes | Tool calls / task |
|---|---|---|---|---|---|---|---|---|
| `oracle` | 174 | 100.0% [97.8, 100.0] | 100.0% | 100.0% | 0.0% | 0.0% | 0.0% | 1.05 |

## By category

| Category | Tasks | Distinct items | Accuracy [95% CI] | Grounding | Look-ahead | Leak |
|---|---|---|---|---|---|---|
| compute | 72 | 72 | 100.0% [94.9, 100.0] | 100.0% | 0.0% | 0.0% |
| lookup | 30 | 30 | 100.0% [88.6, 100.0] | 100.0% | 0.0% | 0.0% |
| multi_step | 24 | 24 | 100.0% [86.2, 100.0] | 100.0% | 0.0% | 0.0% |
| pit_trap | 24 | 24 | 100.0% [86.2, 100.0] | n/a | 0.0% | 0.0% |
| policy_trap | 12 | 12 | 100.0% [75.8, 100.0] | 100.0% | 0.0% | 0.0% |
| unknown_symbol | 12 | 12 | 100.0% [75.8, 100.0] | 100.0% | 0.0% | 0.0% |

