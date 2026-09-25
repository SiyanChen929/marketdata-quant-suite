# Benchmark run: `no_guard`

> **harness-validation baselines on synthetic data; LLM agent results pending (requires ANTHROPIC_API_KEY)**

ablation A1: the lookahead_naive policy with the refusal of later dates lifted.

All prices are synthetic (a one-factor lognormal model on a weekday calendar). These numbers validate the harness and say nothing about real markets or about any language model.

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=packages/quant-marketdata/src:projects/marketdata-agent/src python projects/marketdata-agent/scripts/run_benchmark.py
```

- Suite: 174 tasks (compute 72, lookup 30, multi_step 24, pit_trap 24, policy_trap 12, unknown_symbol 12), 174 distinct items; seed 20240917; `marketdata-agent/bench-generator/v2`; sha256 `57cd5cb0e19e9ab0`.
- Dataset: synthetic panel, 10 symbols, 2021-01-04 to 2023-12-29, seed 20240601, late listings {'SYN09': '2022-09-01', 'SYN10': '2023-09-01'}.
- Arm: clock OFF (ablation A1); prompt `copilot_system_v1`; tools `default`.
- Valid: yes; tasks scored 174; retried episodes 0.
- Audit chain: verified (1156 records, head `0f101751c456ef49`, recorded in `summary.json`; scripted run with fixed audit timestamps, so a rerun must reproduce this head exactly).

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
| `no_guard` | 174 | 47.7% [40.4, 55.1] | 100.0% | 100.0% | 51.1% | 48.3% | 6.9% | 1.25 |

## By category

| Category | Tasks | Distinct items | Accuracy [95% CI] | Grounding | Look-ahead | Leak |
|---|---|---|---|---|---|---|
| compute | 72 | 72 | 51.4% [40.1, 62.6] | 100.0% | 51.4% | 51.4% |
| lookup | 30 | 30 | 66.7% [48.8, 80.8] | 100.0% | 33.3% | 33.3% |
| multi_step | 24 | 24 | 70.8% [50.8, 85.1] | 100.0% | 41.7% | 41.7% |
| pit_trap | 24 | 24 | 0.0% [0.0, 13.8] | 100.0% | 100.0% | 100.0% |
| policy_trap | 12 | 12 | 0.0% [0.0, 24.2] | 100.0% | 0.0% | 0.0% |
| unknown_symbol | 12 | 12 | 75.0% [46.8, 91.1] | 100.0% | 66.7% | 25.0% |

