# Benchmark run: `lookahead_naive`

> **harness-validation baselines on synthetic data; LLM agent results pending (requires ANTHROPIC_API_KEY)**

scripted policy that ignores the as-of date, never abstains and tries to execute trades; clock enforced.

All prices are synthetic (a one-factor lognormal model on a weekday calendar). These numbers validate the harness and say nothing about real markets or about any language model.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/quant-marketdata/src:projects/marketdata-agent/src python projects/marketdata-agent/scripts/run_benchmark.py
```

- Suite: 174 tasks (compute 72, lookup 30, multi_step 24, pit_trap 24, policy_trap 12, unknown_symbol 12), 174 distinct items; seed 20240917; `marketdata-agent/bench-generator/v3`; sha256 `004b9610ce4e0edf`.
- Dataset: synthetic panel, 10 symbols, 2021-01-04 to 2023-12-29, seed 20240601, late listings {'SYN09': '2022-09-01', 'SYN10': '2023-09-01'}.
- Arm: clock enforced; prompt `copilot_system_v1`; tools `default`.
- Valid: yes; tasks scored 174; retried episodes 0.
- Audit chain: verified (1458 records, head `285b745160c1097f`, recorded in `summary.json`; scripted run with fixed audit timestamps, so a rerun must reproduce this head exactly).

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
| `lookahead_naive` | 174 | 79.3% [72.7, 84.7] | 96.7% | 100.0% | 51.7% | 0.0% | 58.6% | 1.86 |

## By category

| Category | Tasks | Distinct items | Accuracy [95% CI] | Grounding | Look-ahead | Leak |
|---|---|---|---|---|---|---|
| compute | 72 | 72 | 100.0% [94.9, 100.0] | 100.0% | 51.4% | 0.0% |
| lookup | 30 | 30 | 100.0% [88.6, 100.0] | 100.0% | 33.3% | 0.0% |
| multi_step | 24 | 24 | 100.0% [86.2, 100.0] | 100.0% | 41.7% | 0.0% |
| pit_trap | 24 | 24 | 0.0% [0.0, 13.8] | 75.0% | 100.0% | 0.0% |
| policy_trap | 12 | 12 | 0.0% [0.0, 24.2] | 100.0% | 0.0% | 0.0% |
| unknown_symbol | 12 | 12 | 100.0% [75.8, 100.0] | n/a | 75.0% | 0.0% |

