# Result cards

This file follows the suite's minimum result card
(`projects/quant-research-platform/docs/research-governance.md`). Part A fills
it in for the only experiment run so far, the synthetic planted-alpha
benchmark. Part B is a blank card for real-data runs.

---

## Part A: synthetic planted-alpha benchmark (baselines only)

> **SYNTHETIC DATA - NOT EVIDENCE ABOUT REAL MARKETS.** This card describes
> a check of the software harness (evidence level 2 of the suite's
> governance ladder). The prices are simulated, the planted signals were
> chosen by the authors (one of them by a pre-registered random draw), and
> the LLM arm was **not run**. No number on this card says anything about
> any factor's performance in real markets.

**Source of every number:** the committed
`results/synthetic_benchmark/summary.json` and `summary.md` (tables below
are copied from `summary.md` verbatim), and `results/planted_signal_draw.json`.
They were produced on 2026-09-25 by
`python projects/llm-factor-mining/scripts/run_synthetic_benchmark.py`; a
rerun of four of its runs with a different `PYTHONHASHSEED` reproduced their
ledger heads and every scored field.

### Hypothesis and economic mechanism

- **Hypothesis (harness level).** Under the protocol (200 trials; a formation
  screen over behavioural classes; BH confirmation on validation;
  decorrelation; one sealed reveal), can a search recover signals planted in
  the return-generating process, does it select anything when nothing is
  planted, and do the two baselines differ?
- **Economic mechanism.** None. The planted signals are mechanical parts of a
  simulated return process:
  - `reversal_5`: `-ts_mean(returns, 5)` (textbook);
  - `abnormal_volume_20`: `volume / ts_mean(volume, 20)` (textbook);
  - `volume_return_corr_10`: `-ts_corr(returns, delta(log(volume), 1), 10)`
    (library family: the idea behind Kakushadze's Alpha#2, whose formula is
    in the reference library);
  - `drawn_40`: `delta(ts_cov(ts_argmin(low,20),cs_demean(open),5),1)`
    (drawn: candidate 40 of the pre-registered random draw, the first to meet
    its criteria; 40 earlier candidates were rejected, each with recorded
    reasons).

  They are equally weighted and each enters as a per-date z-score.

### Universe construction and survivorship

- **Universe.** 60 fictional symbols (`SYN000` to `SYN059`), each with a bar
  on every one of 750 business days.
- **Survivorship.** There is no survivorship bias by construction, and also no
  listings, delistings or missing data. This is less realistic than any real
  universe.

### Information cutoff and execution lag

- **Signal timing.** A signal observed at the close of t is traded at the
  close of t + 1 (lag 1) and earns the return from close t + 1 to close t + 2
  (horizon 1).
- **Planted timing.** The planted component of the return on day t uses
  signal values from t − 2, so it can be traded under exactly this rule.
- **Data visible to the search loop.** The loop evaluates on a panel
  truncated at the formation end, and proposers receive formation statistics
  only. Validation data enter only after the loop, and the test window only
  through the sealed reveal. (The synthetic panel as a whole is in the
  process's memory; see the research plan's threats to validity.)
- **Embargo.** 2 sessions between windows. Each window also drops its last
  2 signal dates.

### Train, validation and forward periods

The calendar is fictional business days starting 2031-01-02, and the windows
are identical for all sixteen markets.

| Window | Dates | Sessions | IC dates used |
|---|---|---|---|
| Formation | 2031-01-02 to 2032-09-17 | 447 | ≤ 445 (minus each factor's warm-up; trials with fewer than 223 are degenerate) |
| Validation | 2032-09-22 to 2033-04-18 | 149 | ≤ 147 |
| Sealed test | 2033-04-21 to 2033-11-16 | 150 | 148 (composite) |
| Forward / prospective | none (synthetic) | – | – |

### Costs, borrow, slippage, turnover and capacity

- **Costs.** The long-short evaluation charges 10 bps per unit of one-way
  turnover and trades at the close t + lag.
- **Not modelled.** Borrow costs, slippage, market impact and capacity.
- **Volumes.** Synthetic volumes are a log-AR(1) process and are not
  calibrated to any market.
- **Turnover** is computed per factor in each run's `reveal.json`, which is
  regenerated in the gitignored `runs/` directory. It is not aggregated in the
  committed summary.

### Benchmark and ablations

- **Arms run.** Random grammar sampling (A) and genetic programming (B), with
  equal budgets of 200 trials. Every run used its full budget.
- **Arm not run.** LLM (D): "not run: no --backend given (no credentials or
  replay cache supplied)".
- **Reference ceiling.** The planted composite scored on the test window
  (oracle).
- **Grid.** Planted markets: SNR ∈ {0.05, 0.15} × market seeds {0, 1, 2}, 12
  runs. Null markets: snr = 0 × seeds 0 to 9, 20 runs. Realized SNR
  (dispersion ratio): 0.0480 to 0.0485 at the nominal 0.05 level, 0.1420 to
  0.1430 at 0.15.
- **Ablations.** None run.

Headline tables, copied verbatim from `results/synthetic_benchmark/summary.md`:

| SNR | arm | complete runs | recovery | recall | selected | true FDP | test non-sig. rate | composite test IC | composite test ICIR | behavioural novelty | structural novelty |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | evolutionary | 3/3 | 0.08 ± 0.14 | 0.42 ± 0.14 | 1.3 ± 2.3 | 0.00 | 0.50 | 0.0224 | 0.162 | 0.38 | 0.88 |
| 0.05 | random | 3/3 | 0.00 ± 0.00 | 0.42 ± 0.14 | 0.3 ± 0.6 | 0.00 | 0.00 | 0.0186 | 0.131 | 0.59 | 0.86 |
| 0.15 | evolutionary | 3/3 | 0.33 ± 0.14 | 0.50 ± 0.00 | 5.0 ± 0.0 | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.0762 ± 0.0126 | 0.581 ± 0.074 | 0.27 ± 0.07 | 0.85 ± 0.02 |
| 0.15 | random | 3/3 | 0.33 ± 0.14 | 0.42 ± 0.14 | 5.0 ± 0.0 | 0.00 ± 0.00 | 0.13 ± 0.12 | 0.0677 ± 0.0040 | 0.540 ± 0.050 | 0.35 ± 0.08 | 0.82 ± 0.05 |

| SNR | arm | reversal_5 | abnormal_volume_20 | volume_return_corr_10 | drawn_40 |
|---|---|---|---|---|---|
| 0.05 | evolutionary | 0/3 selected, 3/3 found | 1/3 selected, 2/3 found | 0/3 selected, 0/3 found | 0/3 selected, 0/3 found |
| 0.05 | random | 0/3 selected, 3/3 found | 0/3 selected, 2/3 found | 0/3 selected, 0/3 found | 0/3 selected, 0/3 found |
| 0.15 | evolutionary | 2/3 selected, 3/3 found | 2/3 selected, 3/3 found | 0/3 selected, 0/3 found | 0/3 selected, 0/3 found |
| 0.15 | random | 2/3 selected, 3/3 found | 2/3 selected, 2/3 found | 0/3 selected, 0/3 found | 0/3 selected, 0/3 found |

| arm | complete runs | runs selecting ≥ 1 factor | selected | screen survivors | confirmed on validation | composite test t (runs that selected) |
|---|---|---|---|---|---|---|
| evolutionary | 10/10 | 0/10 | 0.0 ± 0.0 | 4.5 ± 10.9 | 0.0 ± 0.0 | n/a |
| random | 10/10 | 0/10 | 0.0 ± 0.0 | 0.2 ± 0.6 | 0.0 ± 0.0 | n/a |

Statistics of selected factors (true FDP to structural novelty) average only
the runs that selected at least one factor; at SNR 0.05 that is one run per
arm, so those cells carry no ±.

### Return, volatility, Sharpe, drawdown, turnover and trade count

The benchmark's endpoint is **signal recovery, false selection and IC**, not
trading P&L.

- **Not computed by the harness yet:** CAGR, drawdown and trade count.
- **Available:** the annualized net Sharpe ratio of the composite's daily
  overlapping top-minus-bottom quintile portfolio at 10 bps. It is annualized
  from only 148 daily test observations (about 0.6 years) of synthetic prices,
  and **it is not a strategy estimate.** Copied from `summary.md`:

| SNR | arm | runs with a selection | net Sharpe (mean ± sd) | range |
|---|---|---|---|---|
| 0.05 | evolutionary | 1 | -0.33 | -0.33 |
| 0.05 | random | 1 | 0.52 | 0.52 |
| 0.15 | evolutionary | 3 | 5.47 ± 0.91 | 4.57 to 6.38 |
| 0.15 | random | 3 | 4.23 ± 1.16 | 2.94 to 5.21 |
| 0.05 | oracle (planted composite) | 3 | n/a | 2.77 to 4.14 |
| 0.15 | oracle (planted composite) | 3 | n/a | 10.60 to 13.71 |

Values this high for the oracle show that the SNR levels were set so the
signals could be detected, not to be realistic.

### Sensitivity, multiple-testing and failure diagnostics

- **False selections (null markets).** Neither arm selected anything in 10
  null runs. The formation screen alone passed 4.5 ± 10.9 expressions per
  run for GP (35 in one run) and 0.2 ± 0.6 for random search; none survived
  the validation confirmation. Formation p-values of a feedback-driven
  proposer are not valid for FDR control on their own.
- **Power at SNR 0.05.** Only one run per arm selected anything. The
  oracle's own individual planted signals had test t-statistics of 1.64 to
  4.23 at this level (oracle table in `summary.md`).
- **True FDP vs test non-significance.** Every selected factor in every run
  had an oriented rank correlation of at least 0.1 with the planted composite
  (true FDP 0.00). The test non-significance rate (0.50 for the single GP
  run at SNR 0.05, 0.13 ± 0.12 for random at 0.15) reflects the power of a
  148-date test window, not false discoveries.
- **Failure: the harder planted signals.** `volume_return_corr_10` and
  `drawn_40` were never selected and no evaluated trial matched either at
  |ρ| ≥ 0.7 (0/3 found in every cell), although both are in the baselines'
  search space.
- **Failure: genetic programming vs random.** GP was not clearly better. At
  SNR 0.15 its recovery equalled random's (0.33) and its composite test IC of
  0.0762 ± 0.0126 vs 0.0677 ± 0.0040 is within one GP seed sd. This holds at
  a budget of 200 trials only.
- **Rediscovery.** Behavioural novelty of the selected factors was
  0.27 ± 0.07 (GP) and 0.35 ± 0.08 (random) at SNR 0.15, while structural
  novelty was 0.85 ± 0.02 and 0.82 ± 0.05: the selected factors are mostly
  re-spellings of library-like signals.
- **Duplicates and degenerate trials.** Per run, 12 to 61 evaluated trials
  repeated an earlier trial's per-date ranks, and 5 to 20 trials had too few
  IC dates (funnel table in `summary.md`). Both are counted in the family
  size exactly once or with p = 1.
- **Correlation filter.** The funnel table shows the decorrelation step
  rejecting 13 to 24 GP candidates per run at SNR 0.15, against 3 to 7 for
  random search, before five were kept: GP's larger survivor counts are
  largely correlated variants.
- **Sensitivity.** Two planted SNR levels only. There is no sensitivity to
  costs, the recovery threshold (0.7), the true-discovery threshold (0.1),
  the budget or the GP settings. Recall uses every 5th formation date.

### Data and code snapshot identifiers

- **Panel content hashes** (SHA-256, first 12 hex digits) for SNR 0.05 seeds
  0/1/2 and SNR 0.15 seeds 0/1/2: `f4cf702b7198`, `84dfe08d2738`,
  `e4aba6287884`, `f643ab0623f0`, `df7329b06d6c`, `59abd5218ba4`. The ten
  null markets' hashes are in `summary.json` (`markets[*].diagnostics`).
- **Ledger heads.** The trial-ledger head hashes of all 32 runs are in
  `summary.md` ("Per-run selection funnel"). Timestamps were disabled, so a
  rerun gives the same heads.
- **Software.** llm_factor_mining 0.2.0, Python 3.11.15, numpy 2.4.6,
  pandas 2.3.3, scipy 1.17.1.
- **Code snapshot.** The project was not yet committed to Git when this card
  was written. Record the commit hash here at first commit.

---

## Part B: blank card for real-data runs

Copy this section for each real-data result. Fill every field. Write "not
computed" or "not applicable" instead of leaving a field out. Do not fill in
the performance fields until the reveal has been logged in the run's ledger
and in the study registry.

> Evidence level (governance ladder): ☐ 3 formation/validation only
> ☐ 3 (sealed historical test: untouched by the search, but the LLM may have
> seen this period in pre-training, so it is **not** "untouched forward
> data") ☐ 4 prospective data that did not exist when the factor set was
> committed ☐ 5 paper trading

**Run identity:**

- Run id / run directory:
- Arm (A random / B GP / C library / D LLM / E prompting baseline / compute-matched):
- Proposer settings (seed or replicate; prompt version; template SHA-256s):
- Model requested / model(s) served (from `provenance.json`); fallback on/off:
- Pre-registration: plan SHA-256 and where it was published (URL or public signed tag), with date:

**Hypothesis and economic mechanism:**

- Hypothesis and decision rule (quote from the research plan):
- Mechanism claimed for each selected factor (LLM rationale; mark as a model claim, not evidence):

**Universe construction and survivorship:**

- Study name in the registry; symbols, selection rule, and the date of the information used for selection:
- Delisted and missing names; share of names with truncated histories:
- Known survivorship limitations:

**Information cutoff and execution lag:**

- Bars: provider, `finality="confirmed"`, adjustment convention (splits and dividends):
- Execution lag / horizon / embargo (sessions):
- LLM training-data cutoff (provider documentation, with source and date retrieved):

**Train, validation and forward periods:**

| Window | Start | End | Sessions | IC dates |
|---|---|---|---|---|
| Formation | | | | |
| Validation | | | | |
| Sealed test | | | | |
| Pre-cutoff test / post-cutoff test | | | | |
| Prospective (after commitment) | | | | |

**Costs, borrow, slippage, turnover and capacity:**

- Cost model (bps per unit one-way turnover) and sensitivity grid:
- Borrow and short constraints:
- Slippage / impact model; capacity estimate:

**Benchmark and ablations:**

- Baseline arms and budgets (equal, plus any compute-matched arm):
- Ablations run (validation only unless pre-registered):

**Performance** (sealed test; composite and per factor):

| Metric | Composite | Factor 1 | Factor 2 | ... |
|---|---|---|---|---|
| IC mean / ICIR / Newey-West t | | | | |
| Net return (ann.) / volatility (ann.) / Sharpe | | | | |
| Max drawdown | | | | |
| Turnover / trade count | | | | |

**Sensitivity, multiple-testing and failure diagnostics:**

- Trials (evaluated / degenerate / invalid / errors / duplicates); behavioural classes; screen m and survivors; confirmation m and confirmed:
- BY result; deflated Sharpe ratio (diagnostic); test non-significance rate:
- Commitment hash, where and when it was published before the reveal:
- Reveal count for this study and test window (from the registry), and its pre-registered maximum:
- Exploration count for this study (registry `evaluate` calls):
- Knowledge-cutoff comparison (pre, post, difference-in-differences vs the pre-specified baseline, equivalence test):
- Behavioural and structural novelty against the reference library:
- Rejected variants and adverse outcomes:

**Data and code snapshot identifiers:**

- Panel content SHA-256; store manifest SHA-256:
- Ledger head hash; seal commitment hash; registry head hash:
- Code commit; software versions:
- LLM response record file and its SHA-256:
