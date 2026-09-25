# Synthetic planted-alpha benchmark

> **SYNTHETIC DATA - NOT EVIDENCE ABOUT REAL MARKETS.** Prices are simulated with known planted signals to test whether the search protocol can recover them, and how often it selects factors when nothing is planted. These results say nothing about the profitability of any factor in real markets.

## Reproduce

```bash
PYTHONPATH=packages/quant-marketdata/src:projects/llm-factor-mining/src python projects/llm-factor-mining/scripts/run_synthetic_benchmark.py
```

Total runtime: 332.8 s. Software: llm_factor_mining 0.2.0, numpy 2.4.6, pandas 2.3.3, python 3.11.15, scipy 1.17.1.

## Setup

- Markets: 60 symbols x 750 sessions; planted SNR levels 0.05, 0.15 with market seeds 0, 1, 2; null markets (snr = 0) with seeds 0, 1, 2, 3, 4, 5, 6, 7, 8, 9. Proposer seed = 10000 + market seed.
- Splits: formation / validation / test fractions [0.6, 0.2, 0.2], embargo = lag + horizon sessions; execution lag 1, horizon 1.
- Budget: 200 trials per run unless the arm says `name@budget` (invalid and degenerate proposals count; exact repeats do not), batches of 20. A trial is degenerate when fewer than max(100, 0.5 x scorable) formation dates have a defined IC.
- Selection: formation screen BH at 0.1 over behavioural classes (m = budget minus behavioural duplicates); validation confirmation BH at 0.1 on one-sided validation p-values (m = candidates carried forward); rank by validation ICIR; greedy decorrelation at |rho| < 0.7; top 5; one sealed test reveal.
- Recovery: a planted signal is recovered if a selected (oriented) factor's test-window values have mean cross-sectional rank correlation rho >= 0.7 with it. Recall applies |rho| to every evaluated trial on the formation window.
- True FDP: share of selected factors whose oriented rank correlation with the planted composite (the true expected-return signal) on the test window is below 0.1; on null markets every selected factor is false.
- Test non-significance rate: share of selected factors whose one-sided test IC fails BH at 0.05 within the selected set. It measures test power, not falsity.
- Behavioural novelty: 1 - max |mean per-date rank correlation| of a selected factor with every reference-library entry on the formation window (value-based; 0 = a re-spelled library factor). Structural novelty: 1 - max subtree Jaccard similarity with the library (syntax only; secondary).

Planted signals (equal weights, each entered as `cs_zscore(expression)`); behavioural novelty against the reference library on the formation window, range over planted markets:

| name | expression | family | behavioural novelty | nearest library entry |
|---|---|---|---|---|
| reversal_5 | `-ts_mean(returns, 5)` | textbook | 0.00 | reversal_5d |
| abnormal_volume_20 | `volume / ts_mean(volume, 20)` | textbook | 0.00 | abnormal_volume_20d |
| volume_return_corr_10 | `-ts_corr(returns, delta(log(volume), 1), 10)` | library_family | 0.68 to 0.72 | volume_change_vs_intraday_return |
| drawn_40 | `delta(ts_cov(ts_argmin(low,20),cs_demean(open),5),1)` | drawn | 0.94 to 0.95 | intraday_reversal_5d, reversal_5d |

Arms:

- `evolutionary`: run
- `llm`: not run: no --backend given (no credentials or replay cache supplied)
- `random`: run

## Headline: planted markets (mean ± sd across complete runs)

| SNR | arm | complete runs | recovery | recall | selected | true FDP | test non-sig. rate | composite test IC | composite test ICIR | behavioural novelty | structural novelty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | evolutionary | 3/3 | 0.08 ± 0.14 | 0.42 ± 0.14 | 1.3 ± 2.3 | 0.00 | 0.50 | 0.0224 | 0.162 | 0.38 | 0.88 |
| 0.05 | random | 3/3 | 0.00 ± 0.00 | 0.42 ± 0.14 | 0.3 ± 0.6 | 0.00 | 0.00 | 0.0186 | 0.131 | 0.59 | 0.86 |
| 0.15 | evolutionary | 3/3 | 0.33 ± 0.14 | 0.50 ± 0.00 | 5.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0762 ± 0.0126 | 0.581 ± 0.074 | 0.27 ± 0.07 | 0.85 ± 0.02 |
| 0.15 | random | 3/3 | 0.33 ± 0.14 | 0.42 ± 0.14 | 5.0 ± 0.0 | 0.00 ± 0.00 | 0.13 ± 0.12 | 0.0677 ± 0.0040 | 0.540 ± 0.050 | 0.35 ± 0.08 | 0.82 ± 0.05 |

Recovery, recall and selected average all complete runs. True FDP, the test non-significance rate, composite test IC/ICIR and the novelty scores describe selected factors, so they average only the runs that selected at least one factor (see the per-run funnel); a value without ± comes from a single run, and `n/a` means no run selected anything.

## Recovery of each planted signal (complete runs)

| SNR | arm | reversal_5 | abnormal_volume_20 | volume_return_corr_10 | drawn_40 |
|---|---|---|---|---|---|
| 0.05 | evolutionary | 0/3 selected, 3/3 found | 1/3 selected, 2/3 found | 0/3 selected, 0/3 found | 0/3 selected, 0/3 found |
| 0.05 | random | 0/3 selected, 3/3 found | 0/3 selected, 2/3 found | 0/3 selected, 0/3 found | 0/3 selected, 0/3 found |
| 0.15 | evolutionary | 2/3 selected, 3/3 found | 2/3 selected, 3/3 found | 0/3 selected, 0/3 found | 0/3 selected, 0/3 found |
| 0.15 | random | 2/3 selected, 3/3 found | 2/3 selected, 2/3 found | 0/3 selected, 0/3 found | 0/3 selected, 0/3 found |

## Null markets (snr = 0): false selections

| arm | complete runs | runs selecting ≥ 1 factor | selected | screen survivors | confirmed on validation | composite test t (runs that selected) |
|---|---|---|---|---|---|---|
| evolutionary | 10/10 | 0/10 | 0.0 ± 0.0 | 4.5 ± 10.9 | 0.0 ± 0.0 | n/a |
| random | 10/10 | 0/10 | 0.0 ± 0.0 | 0.2 ± 0.6 | 0.0 ± 0.0 | n/a |

## Oracle reference (planted signals scored on the test window)

Power = share of planted signals whose one-sided test IC passes BH at 0.05 within the planted set.

| SNR | seed | reversal_5 | abnormal_volume_20 | volume_return_corr_10 | drawn_40 | planted composite | power |
|---|---|---|---|---|---|---|---|
| 0.05 | 0 | IC 0.0180, t 1.90 | IC 0.0407, t 4.23 | IC 0.0177, t 1.64 | IC 0.0238, t 2.25 | IC 0.0554, t 4.72 | 0.75 |
| 0.05 | 1 | IC 0.0274, t 2.59 | IC 0.0300, t 2.56 | IC 0.0244, t 2.28 | IC 0.0264, t 2.54 | IC 0.0522, t 4.65 | 1.00 |
| 0.05 | 2 | IC 0.0160, t 1.69 | IC 0.0424, t 3.82 | IC 0.0258, t 2.64 | IC 0.0308, t 3.09 | IC 0.0554, t 4.90 | 1.00 |
| 0.15 | 0 | IC 0.0537, t 5.71 | IC 0.0830, t 8.80 | IC 0.0562, t 5.52 | IC 0.0416, t 3.88 | IC 0.1333, t 11.94 | 1.00 |
| 0.15 | 1 | IC 0.0652, t 6.39 | IC 0.0700, t 5.86 | IC 0.0625, t 5.92 | IC 0.0568, t 5.12 | IC 0.1370, t 12.16 | 1.00 |
| 0.15 | 2 | IC 0.0506, t 5.19 | IC 0.0870, t 7.39 | IC 0.0701, t 7.44 | IC 0.0681, t 7.36 | IC 0.1445, t 12.95 | 1.00 |

## Composite long-short on the test window (synthetic)

Annualized net Sharpe ratio of the composite's daily overlapping top-minus-bottom quintile portfolio at 10 bps per unit of one-way turnover, from the test window only. It is not a strategy estimate.

| SNR | arm | runs with a selection | net Sharpe (mean ± sd) | range |
|---|---|---|---|---|
| 0.05 | evolutionary | 1 | -0.33 | -0.33 |
| 0.05 | random | 1 | 0.52 | 0.52 |
| 0.15 | evolutionary | 3 | 5.47 ± 0.91 | 4.57 to 6.38 |
| 0.15 | random | 3 | 4.23 ± 1.16 | 2.94 to 5.21 |
| 0.05 | oracle (planted composite) | 3 | n/a | 2.77 to 4.14 |
| 0.15 | oracle (planted composite) | 3 | n/a | 10.60 to 13.71 |

## Per-run selection funnel

| SNR | seed | arm | trials | evaluated | degenerate | behav. duplicates | screen survivors | confirmed | rejected (corr.) | selected | recovery | true FDP | composite test IC | ledger head |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0 | random | 200 | 188 | 12 | 27 | 19 | 0 | 0 | 0 | 0.00 | n/a | n/a | `2bedbf9185c8` |
| 0.05 | 0 | evolutionary | 200 | 192 | 8 | 23 | 51 | 0 | 0 | 0 | 0.00 | n/a | n/a | `1e85dd76eee5` |
| 0.05 | 1 | random | 200 | 192 | 8 | 27 | 7 | 0 | 0 | 0 | 0.00 | n/a | n/a | `ac5f69cf5221` |
| 0.05 | 1 | evolutionary | 200 | 194 | 6 | 17 | 31 | 0 | 0 | 0 | 0.00 | n/a | n/a | `6784e7f81907` |
| 0.05 | 2 | random | 200 | 182 | 18 | 22 | 9 | 1 | 0 | 1 | 0.00 | 0.00 | 0.0186 | `d27f61e1b67e` |
| 0.05 | 2 | evolutionary | 200 | 193 | 7 | 44 | 58 | 11 | 7 | 4 | 0.25 | 0.00 | 0.0224 | `fc5664ef3d71` |
| 0.15 | 0 | random | 200 | 188 | 12 | 27 | 63 | 55 | 4 | 5 | 0.25 | 0.00 | 0.0689 | `24b6026af74b` |
| 0.15 | 0 | evolutionary | 200 | 195 | 5 | 40 | 92 | 85 | 19 | 5 | 0.50 | 0.00 | 0.0836 | `2940093f9e73` |
| 0.15 | 1 | random | 200 | 192 | 8 | 27 | 57 | 50 | 7 | 5 | 0.50 | 0.00 | 0.0710 | `794e287bcb7f` |
| 0.15 | 1 | evolutionary | 200 | 195 | 5 | 23 | 77 | 70 | 13 | 5 | 0.25 | 0.00 | 0.0616 | `dbccfff37251` |
| 0.15 | 2 | random | 200 | 182 | 18 | 22 | 55 | 39 | 3 | 5 | 0.25 | 0.00 | 0.0633 | `4ee2cf37c3ed` |
| 0.15 | 2 | evolutionary | 200 | 194 | 6 | 61 | 86 | 75 | 24 | 5 | 0.25 | 0.00 | 0.0833 | `ae252b07c4b9` |
| 0.00 | 0 | random | 200 | 188 | 12 | 27 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `c5abc3438825` |
| 0.00 | 0 | evolutionary | 200 | 193 | 7 | 53 | 35 | 0 | 0 | 0 | 0.00 | n/a | n/a | `1d872c289bb8` |
| 0.00 | 1 | random | 200 | 192 | 8 | 27 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `79703e499f0e` |
| 0.00 | 1 | evolutionary | 200 | 193 | 7 | 54 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `1d3a6c463a59` |
| 0.00 | 2 | random | 200 | 182 | 18 | 22 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `bb487fbfd5af` |
| 0.00 | 2 | evolutionary | 200 | 195 | 5 | 28 | 4 | 0 | 0 | 0 | 0.00 | n/a | n/a | `ca4d3094b29b` |
| 0.00 | 3 | random | 200 | 180 | 20 | 24 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `7d2bcfb8298d` |
| 0.00 | 3 | evolutionary | 200 | 185 | 15 | 41 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `0bb6a20588c3` |
| 0.00 | 4 | random | 200 | 180 | 20 | 20 | 2 | 0 | 0 | 0 | 0.00 | n/a | n/a | `caa648618786` |
| 0.00 | 4 | evolutionary | 200 | 190 | 10 | 13 | 6 | 0 | 0 | 0 | 0.00 | n/a | n/a | `ad32395d7f48` |
| 0.00 | 5 | random | 200 | 184 | 16 | 18 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `ca8d318804ab` |
| 0.00 | 5 | evolutionary | 200 | 186 | 14 | 46 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `d5fcd691cd89` |
| 0.00 | 6 | random | 200 | 187 | 13 | 23 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `ad1d09c81762` |
| 0.00 | 6 | evolutionary | 200 | 192 | 8 | 18 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `b29e397e4483` |
| 0.00 | 7 | random | 200 | 189 | 11 | 19 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `6ac61f4ba1db` |
| 0.00 | 7 | evolutionary | 200 | 191 | 9 | 12 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `417fb068c49a` |
| 0.00 | 8 | random | 200 | 185 | 15 | 19 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `3b220a0b2703` |
| 0.00 | 8 | evolutionary | 200 | 188 | 12 | 22 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `80fcea568990` |
| 0.00 | 9 | random | 200 | 182 | 18 | 18 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `db52517d0860` |
| 0.00 | 9 | evolutionary | 200 | 191 | 9 | 20 | 0 | 0 | 0 | 0 | 0.00 | n/a | n/a | `b01dccba6569` |

## Incomplete or aborted runs

None: every run used its full budget.

## Caveats

- Synthetic data: the data-generating process, the planted signals and the SNR levels were chosen by the authors of this benchmark (the `drawn` signal by a pre-registered random draw, see `results/planted_signal_draw.json`). A proposer's ranking here need not transfer to real markets.
- The two `textbook` signals favour any proposer with prior knowledge of classic factors (for example an LLM); `library_family` is a documented idea whose Kakushadze (2016) formula is in the reference library; `drawn` was sampled from the random baseline's own grammar distribution, which if anything favours that baseline.
- 3 seeds per planted cell and 10 null seeds give coarse estimates; standard deviations are across runs.
- Budgets of a few hundred trials are small for genetic programming; results for GP hold at this budget only (arms `name@budget` run larger budgets under the same trial accounting).
- Ledger heads are SHA-256 hashes of each run's trial ledger (timestamps disabled), so a rerun on the same software stack can be checked record by record.
