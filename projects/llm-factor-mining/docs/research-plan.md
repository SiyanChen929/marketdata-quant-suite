# Research plan: LLM-guided formulaic factor mining under a sealed-holdout protocol

Status as of 2026-09-26: this is a **pre-registration draft**.

- **Done.** The harness and two baselines are implemented, and a synthetic
  benchmark validates them ([result card](result-card.md)).
- **Not yet run.** No LLM experiment and no real-market experiment has been
  run. All hypotheses below are open.
- **Freezing the plan.** The plan is frozen **before the first LLM call**
  (not only before the first real-data reveal): its SHA-256 is published
  outside this repository, on a public registry such as OSF or as a signed
  tag pushed to a public remote, and quoted on every result card. A commit
  in one's own repository alone is not a pre-registration, because history
  can be rewritten. Any later change is logged in a "deviations" section
  rather than edited in place.
- **Disclosure.** H2 and the benchmark's planted signals were revised on
  2026-09-25 *after* the baseline harness runs had been seen (a review found
  that the earlier "non-textbook" signal was outside the baselines' search
  space and belonged to a documented factor family). In the same round of
  revisions:
  - the window menu that both baselines share (`GrammarConfig.windows`) gained
    the value 1 (it was 2, 3, 5, 10, 20, 40, 60), so that the library-family
    signal's one-session change `delta(log(volume), 1)` became reachable; GP's
    operator settings were not changed;
  - the selection procedure took its present two-stage form (formation screen
    plus validation confirmation).

  The revised benchmark was re-run with the baselines before any LLM
  experiment; it is harness validation, not a confirmatory test. H2's
  confirmatory seeds (100 to 119) exclude the harness seeds 0 to 2 already
  seen. On 2026-09-26, after a review of those harness results, H4's
  synthetic endpoint was changed from raw recall to recovery and
  null-contrasted recall, because raw recall turned out to be about as high
  on null markets as on planted ones.

## 1. Motivation

Formulaic alphas are short programs over price and volume data. Kakushadze
(2016) catalogues 101 of them. Searching over such programs is an old idea;
genetic programming (Koza, 1992) is the classic approach. More recent work
uses reinforcement learning (Yu et al., 2023) and interactive LLM prompting
(Wang et al., 2023). LLM-guided program search has also produced verifiable
discoveries in mathematics (Romera-Paredes et al., 2024).

Finance adds two problems that mathematics does not have:

1. **Multiple testing.** Every candidate is a hypothesis test on one noisy,
   non-repeatable history, and the factor literature already suffers from
   data snooping (Harvey, Liu & Zhu, 2016; White, 2000; Bailey et al., 2017).
   A search method that tries more candidates finds more false positives
   unless every trial is counted.
2. **Look-ahead through pre-training.** An LLM may have read about which
   signals worked in which period. LLMs have been used to predict returns
   from text (Lopez-Lira & Tang, 2023), and such predictions can suffer from
   look-ahead bias (Glasserman & Lin, 2023). An LLM's apparent edge can
   therefore be memory rather than search.

This project measures LLM-guided search under a protocol built to remove both
problems before any claim is made.

## 2. Research questions

- **RQ1 (discovery at equal budget).** At an equal trial budget, does
  LLM-guided proposal yield frozen factor sets with higher sealed-test
  performance than random grammar sampling and genetic programming? It must
  do so under the two-stage multiplicity procedure: a formation screen whose
  family counts every budgeted trial, and a confirmatory test on the
  validation window, which no proposer sees.
- **RQ2 (search efficiency).** How do the arms differ in validity rate,
  duplicate rate, trials-to-first-recovery (synthetic) and cost per
  test-surviving factor?
- **RQ3 (memorization and look-ahead).** How much of any LLM advantage is
  explained by two things?
  - (a) Rediscovery of known factors, measured primarily by *behavioural*
    (value-based) novelty against a reference library, with structural
    novelty as a secondary, syntactic description, and by recovery of the
    benchmark's drawn planted signal.
  - (b) Period-specific knowledge from pre-training, measured with a
    knowledge-cutoff difference-in-differences and anonymization probes.
- **RQ4 (mechanism).** Which components of the LLM arm matter? The
  candidates are formation feedback, the rationale requirement, prompt
  anonymization and reasoning effort.

## 3. Hypotheses and decision rules

All tests are one-sided in the stated direction at α = 0.05. Holm's
step-down correction is applied within each hypothesis family. Each
hypothesis is supported only if every condition listed for it holds;
otherwise it is reported as not supported. Both outcomes are publishable.

### H1: LLM beats both baselines on the sealed test (RQ1, real data)

- **Primary endpoint.** Composite test IC of each arm's frozen sets: the
  equal-weight composite of per-date z-scored selected factors, scored on the
  embargoed sealed test window. An arm's R replicates are pooled into one
  daily IC series by averaging their composites' daily IC.
- **Two estimands, stated separately.** Replicates of one arm and universe
  share a single test window, so they measure the variability of the
  *search procedure* given one market history, not generalization to other
  histories.
  - (E1, market sampling) the mean over test dates of the daily difference
    in pooled composite IC, LLM minus baseline. Test: one-sided stationary
    block bootstrap over test dates (mean block length 10 sessions, 10,000
    resamples, bootstrap seed 20260925), in the spirit of Hansen's (2005) SPA
    test.
  - (E2, procedure variability) the difference in replicate-level composite
    test ICIR. Test: one-sided exact Mann–Whitney U over the R LLM and R
    baseline replicates (unpaired: LLM replicate r and baseline seed r share
    nothing, so any pairing would be arbitrary).
- **Comparisons.** Baseline ∈ {random, GP}; Holm correction over the two
  comparisons, separately for E1 and E2.
- **Supported iff**, for both baselines and in every universe, the E1 test
  rejects after Holm (p < 0.05), the E2 test rejects after Holm, and the
  median replicate ICIR difference is at least δ_min. δ_min is fixed before
  the reveal; the default is 0.02 per period.
- **Secondary endpoints**, reported with the same test but without a support
  claim:
  - the number of selected factors whose test IC is BH-significant at 0.05;
  - the test non-significance rate (on real data the true FDP is unknown);
  - composite long-short net Sharpe at 10 bps;
  - sensitivity to 0, 5 and 20 bps costs.

### H2: LLM recovers the drawn planted signal more often (RQ1/RQ3a, synthetic)

- **Endpoint.** Whether `drawn_40` is recovered: a selected, oriented factor
  has ρ ≥ 0.7 with it on the test window. `drawn_40` was drawn at random from
  the typed grammar by a procedure fixed before the draw
  (`benchmark/draw.py`; every candidate and rejection is in
  `results/planted_signal_draw.json`) and is far, in values, from every
  library entry. No proposer can know it in advance; it comes from the
  random baseline's own sampling distribution, which if anything favours that
  baseline.
- **Design.** Paired by market seed: seeds 100 to 119 (20 seeds) at each of
  SNR 0.05 and 0.15, all arms on the same markets, plus the same null-market
  protocol as the harness.
- **Test.** Exact McNemar (binomial) test on discordant seeds for LLM vs each
  baseline at each SNR level, with Holm correction over 4 comparisons.
- **Supported iff** at least one Holm-adjusted p < 0.05 and, in every
  comparison, the LLM has at least as many LLM-only recoveries as the baseline
  has baseline-only recoveries.
- **Reported descriptively.** Recovery of the two textbook signals and of the
  `library_family` signal (the idea behind Kakushadze's Alpha#2) carries no
  support claim: an LLM advantage there is consistent with prior knowledge,
  not with search ability.
- **Harness status.** In the revised harness run (3 seeds per SNR), neither
  baseline selected or even found `drawn_40` or the library-family signal,
  although both are in their search space.

### H3: The LLM edge is not explained by period memorization (RQ3b, real data)

- **Estimand.** The difference-in-differences D, where IC is test-window
  rank IC of the pooled composite and pre/post split the test dates at the
  model's provider-documented training-data cutoff. Dates whose forward
  return straddles the cutoff are dropped (see `protocol.contamination`).

  ```
  D = (IC_post − IC_pre)[LLM] − (IC_post − IC_pre)[GP]
  ```

  The comparison baseline is pre-specified as genetic programming (arm B):
  like the LLM it uses formation feedback, and it has no pre-training.
- **Why a DiD.** Differencing against a baseline removes decay that is common
  to all factors, for example crowding or regime change. What remains is
  decay specific to what the LLM could have memorized.
- **Inference.** The decision uses D alone, with a 90% confidence interval
  from the stationary block bootstrap of H1 (resampling dates within the pre
  and post windows separately). Comparing "significant before" with "not
  significant after" is not a test and is not used.
- **Memorization is flagged iff** the interval lies entirely below 0 (the
  LLM's IC drops more after the cutoff than GP's).
- **No memorization is declared only by equivalence:** a TOST at α = 0.05
  with the margin |D| < 0.01 (daily IC), fixed now, i.e. the 90% interval
  lies inside (−0.01, 0.01).
- **Supported iff** equivalence is established and the LLM's post-cutoff
  composite IC is significantly positive (one-sided Newey–West t, α = 0.05).
  Otherwise H3 is reported as memorization flagged or as inconclusive.
- **Power guard.** If fewer than 250 post-cutoff test sessions exist, H3 is
  declared inconclusive (underpowered). The plan then extends to the
  prospective window (§5.3).

### H4: Formation feedback improves search (RQ4)

- **Comparison.** LLM with feedback vs LLM with feedback removed, at equal
  budget.
- **Endpoints.**
  - synthetic: recovery and trials-to-first-recovery over at least 20 seeds,
    plus recall only as a contrast with null markets (planted minus null
    recall per arm). Raw recall is not an endpoint: in the harness it was
    about as high on null markets, where nothing is planted, as on planted
    ones, so it measures whether a proposer writes such expressions, not
    detection;
  - real data: validation-window composite ICIR only, so there is no extra
    test reveal.
- **Test.** Wilcoxon signed-rank test, paired by seed or replicate.
- **Supported iff** p < 0.05 on the synthetic recovery endpoint and the
  real-data validation difference has the same sign.

## 4. Planned contributions

1. **Protocol.** An auditable protocol for LLM-driven factor search, with:
   - a typed factor language that cannot look ahead;
   - a formation-only information barrier enforced by closed types;
   - a hash-chained ledger that counts invalid, failed and degenerate
     proposals as trials;
   - a formation screen over that full family, with behavioural duplicates
     merged, followed by a confirmatory test on unseen validation data;
   - a commit-then-reveal test hold-out with a project-wide reveal register.
2. **Benchmark.** A planted-alpha synthetic benchmark with null markets and
   a ground truth that mixes textbook signals, a documented factor family and
   a signal drawn at random by a procedure fixed in code before the draw was
   run (self-attested, not externally registered). Only recovery of
   the drawn signal speaks to search beyond prior knowledge.
3. **Comparison.** An equal-budget comparison of LLM, random-grammar and
   genetic-programming search on US equities (pending).
4. **Contamination diagnostics.** A knowledge-cutoff difference-in-differences,
   library-novelty stratification and anonymization probes (pending runs).
5. **Artifacts.** Open, replayable artifacts: recorded LLM responses, prompt
   template hashes, ledger heads and data hashes for every run.

Contribution 1 and the harness part of 2 exist today. Contributions 3 to 5
depend on experiments that have not been run.

## 5. Experimental protocol

### 5.1 Data

- **Source.** US equity daily bars from MarketData, read only through the
  suite's `quant_marketdata` gateway (`MarketDataStore.read_bars` /
  `MarketDataClient.get_bulk_daily_bars`) with `finality="confirmed"`.
  Provisional sessions never enter an evaluation.
- **Fields.** Open, high, low, close and volume. The derived terminals are
  close-to-close `returns`, `vwap_proxy` (the typical price; daily bars carry
  no true VWAP) and `dollar_volume`.
- **Adjustments.** How the provider adjusts for splits and dividends must be
  checked and documented before any claim, because close-to-close returns
  include or omit dividends accordingly.
- **Data-quality report, required before the first formation run:**
  - coverage by date and symbol;
  - zero or negative prices;
  - extreme returns;
  - stale prices;
  - volume outliers.
- **Provenance.** Every run records the panel content hash and the store's
  manifest hash.

### 5.2 Universes

- **U1: liquid large caps.** About 200 to 500 names selected by median dollar
  volume, computed on formation data only.
- **U2: a broader liquid universe.** About 1,000 names, with a lower
  dollar-volume floor.
- **Membership is frozen at the formation end.** Using later information to
  choose names would be look-ahead.
- **Survivorship limitation.** The suite has no point-in-time index
  membership (see the root README's "Current boundary"). If the provider does
  not return delisted names, both universes suffer from survivorship bias.
  This is reported as a threat and quantified where possible, for example by
  the share of names with truncated histories.

### 5.3 Windows

- **Chronological split.** Formation, validation and sealed test, split by
  session counts (default fractions 0.6 / 0.2 / 0.2). The embargo between
  windows is at least `lag + max_horizon` sessions, and each window drops its
  last `lag + h` signal dates.
- **Timing.** Execution lag is 1 session and the primary horizon is 1
  session. Horizons of 5 and 21 sessions are secondary analyses, and the
  embargo grows with them.
- **Knowledge cutoff.** The sealed test window must extend past the
  provider-documented training-data cutoff of every model evaluated, so that
  the H3 split exists. The cutoff date is taken from provider documentation
  at run time; this plan does not assume one.
- **Prospective window (evidence level 4 of the suite's governance ladder).**
  After the main reveal, frozen factor sets are committed to the ledger, which
  is timestamped. They are scored on sessions that did not exist at
  commitment time. This is the only window that is out of sample for the LLM
  by construction.

### 5.4 Budgets, replicates and arms

- **Budget.** The primary budget is B = 200 trials per run. The scaling study
  uses B ∈ {200, 500, 1000} on synthetic data and validation only. Equal
  *trial* budgets are not equal *compute*: a GP trial costs well under a
  second, an LLM batch costs thinking tokens. Arm F therefore gives the
  baselines the LLM arm's compute, and recovery (and null-contrasted recall)
  is reported against both trials and cost. The harness finding that GP did not beat random search
  holds at B = 200 only.
- **Batches.** Batch size is 20.
- **What counts as a trial.** Every non-duplicate processed proposal: valid
  and evaluated, degenerate (fewer than max(100, ½ × scorable) formation
  dates with a defined IC), invalid, or failed during evaluation. Exact
  canonical repeats and repeated invalid text are logged but do not consume
  budget. A run that stops before its budget (stalled, or aborted after
  repeated proposer failures) is reported as incomplete and never averaged
  with complete runs; an aborted run is neither committed nor revealed.
- **Replicates.**
  - Baselines vary the proposer seed.
  - The LLM arm uses a replicate id in the request tag. Current models accept
    no sampling parameters, so replicate variation comes from the model
    itself and is recorded, not controlled.
  - Real data uses R = 5 replicates per arm and universe.
  - Synthetic data uses at least 20 market seeds per SNR level.

| Arm | Description | Status |
|---|---|---|
| A: random grammar | i.i.d. typed trees; no feedback | implemented |
| B: genetic programming | tournament selection, subtree crossover, subtree/point mutation, 10% immigrants; fitness \|formation ICIR\| − 0.002·complexity; settings fixed before experiments | implemented |
| C: classic library | the 18-entry reference catalog passed through the same selection and seal | planned (needs a library proposer) |
| D: LLM (protocol) | `LLMProposer`, prompt v2, formation-only feedback, JSON-schema output, adaptive thinking, effort `high`, fallback off | implemented, **not yet run** |
| E: Alpha-GPT-style prompting (optional) | an interactive-alpha-mining prompting style (Wang et al., 2023) re-implemented inside the same DSL, budget and protocol | planned |
| F: compute-matched baselines | GP and random search with a budget matched to the LLM arm's measured cost (wall-clock or dollars, fixed from the pilot; at least 10^4 trials), under the same trial accounting (m = its budget) | infrastructure implemented (`name@budget` arms; the round cap `max(100, 10⌈B/b⌉)` scales with the budget, and a test checks that a 10^4-trial arm's cap leaves room for its whole budget), **not yet run** |

### 5.5 Metrics

- **Per run.**
  - Trial counts: trials, evaluated, invalid, errors, duplicates, validity
    rate.
  - Selection funnel: behavioural classes and duplicates, degenerate trials,
    formation-screen survivors, validation-confirmed candidates, and the
    number of factors selected (at most top-k = 5).
  - Sealed test: IC, ICIR and Newey-West t of each selected factor and of the
    composite.
  - Test non-significance rate (selected factors not BH-significant on the
    test: a power measure). On synthetic data also the true FDP (selected
    factors whose oriented rank correlation with the planted composite is
    below 0.1) and the null-market false-selection rate.
  - Portfolio: composite long-short gross and net return, volatility and
    Sharpe, with 10 bps per unit of one-way turnover, and turnover itself.
  - Novelty: behavioural novelty against the library (primary) and structural
    novelty (secondary).
  - Provenance: served model ids and token usage.
- **Synthetic runs also report** recovery, recall and per-signal
  recovered/found counts, with recall on the null markets next to recall on
  the planted ones ("found" means a close copy was written, not that a
  signal was detected).
- **The result card** reports the governance ladder's metrics: return,
  volatility, Sharpe, drawdown, turnover and trade count. The harness does not
  yet compute drawdown or trade count; adding them is part of real-data
  preparation.

### 5.6 Statistical analysis

- **Within a run** (implemented):
  - formation screen: BH at q = 0.10 on two-sided formation p-values of
    behavioural-class representatives, with m = budget minus behavioural
    duplicates (degenerate, invalid and failed trials count with p = 1).
    Because GP and the LLM choose later expressions after seeing formation
    results, these p-values are *not* valid for FDR control; the screen is a
    filter. The harness null markets show why: GP passed up to 35
    expressions through it on data with nothing planted;
  - confirmation (the claim-carrying step): BH at q = 0.10 on one-sided
    validation p-values of the screened, oriented candidates, m = number of
    candidates. No proposer ever sees validation data, so these p-values are
    valid given the candidate set;
  - BY at the same q, for robustness under arbitrary dependence
    (`fdr_method="by"`, `confirm_method="by"`; implemented, not yet reported:
    the harness summary uses BH only);
  - the deflated Sharpe ratio of formation |ICIR| (Bailey & López de Prado,
    2014) as a diagnostic, with 2N trials because selecting on |ICIR| is
    two-sided;
  - Newey-West (1987) HAC t-statistics with bandwidth
    `max(h − 1, ⌊4 (T/100)^{2/9}⌋)`, referred to t(n − 1).
- **Across runs** (planned): paired non-parametric tests as specified in §3.
  Seed-level results are always shown, not only means.
- **Data-snooping controls** (future work, not yet implemented): White's
  (2000) Reality Check, Hansen's (2005) SPA test and the Romano-Wolf (2005)
  step-down procedure. Each would run over the full ledger of each arm, with a
  stationary bootstrap over dates.
- **Known simplifications**, to be revisited before submission:
  - the t(n − 1) reference for the Newey-West t is a convention, not an
    exact small-sample result;
  - the deflated Sharpe assumes independent trials;
  - behavioural deduplication merges only trials with identical per-date
    ranks. Highly correlated but distinct variants remain separate
    hypotheses. For BH (a step-up rule) such clusters make the screen *less*
    strict, not more; its validity then rests on positive dependence, and BY
    is to be reported alongside.

### 5.7 Reveal accounting

- **Study registry (implemented).** A project-wide, hash-chained register
  (`protocol/registry.py`, CLI `register-study`) is required for every
  real-data command:
  - a *study* fixes the data (symbols, dates), the split and the
    pre-registered maximum number of test reveals before anything is
    evaluated; it cannot be re-registered with other terms;
  - every `evaluate` call on the study is logged as exploration and counted;
  - `search` on store data only commits; the commitment is recorded in the
    registry and must be published outside the repository before `reveal`;
  - `reveal` checks the published commitment, the run's ledger and data
    hash, and refuses once the study's reveals are used up. Refusals by the
    seal and by the registry are logged; the command's own pre-checks (a
    commitment hash that does not match, reloaded data that do not match the
    committed data hash, a ledger without the commitment) stop before any
    test metric is computed and are not logged;
  - within one ledger, the seal allows one commitment and one reveal per
    (data hash, test window), whatever the metric settings.
- **What it does not do.** The registry, like the ledger, is self-attested:
  it makes repeated reveals visible and countable, not impossible. Changing
  the universe requires a new study, which is visible in the registry.
- **Pre-registered reveals.** The number of test reveals is fixed in advance:
  arms A, B, C and D (plus E if run) × 2 universes × R replicates at B = 200.
- **Reporting.** Every reveal is reported, including failed or aborted ones.
- **Ablations** use validation data only (§6). They reveal the test window
  only if pre-registered as separate arms.

## 6. Ablations (RQ4)

Ablations run on synthetic data (at least 20 seeds) and on the real-data
validation window only.

| Ablation | Levels | Implementation status |
|---|---|---|
| Formation feedback | on (default) / off (context without the top, bottom, previous-round and rejected blocks) | needs a feedback-stripping wrapper |
| Rationale requirement | on (the current `PROPOSAL_SCHEMA`, whose SHA-256 is recorded) / off (expression-only schema) | needs an expression-only schema version |
| Anonymization | on (default) / off, i.e. true dates and tickers in the prompt, **as a contamination probe only**, in a pre-cutoff window | needs an explicit probe mode; the default context design forbids it |
| Reasoning effort | low / medium / high (default) / max | available through `--effort` |
| Model | the default model plus optional others | available through `--model`; served ids are logged |
| Budget | 200 / 500 / 1000 | available |

For the anonymization probe, the question is whether revealing the calendar
and the tickers raises validation IC before the cutoff but not after it. If
it does, that is direct evidence of recall.

## 7. Threats to validity

- **Survivorship and universe construction.** Without point-in-time
  membership or delisted histories, universes favour survivors. The design
  freezes membership from formation-period data, reports how many names have
  truncated histories, and treats cross-universe consistency as a robustness
  check.
- **Look-ahead through pre-training.** Anonymization does not stop a model
  from knowing that, say, short-term reversal is a documented anomaly. It
  removes only the cues that link statistics to a specific period.
  - H3's DiD and the prospective window address period-specific recall.
  - Novelty stratification and H2 address generic prior knowledge.
- **Prompt leakage.** Harness-written prompt text is checked for ISO dates,
  month names (full names; "March", "May" and abbreviations such as "Sept."
  when capitalized), four-digit years (also sentence-final) and panel
  symbols; a hit aborts the run. Rejected-proposal text (the model's own
  expression plus a validator message, or a fixed, count-free text for
  degenerate and failed trials) that trips the check is redacted, not
  echoed. The check is heuristic:
  - numbers inside feedback statistics and the integer counters are not
    checked for year-like values;
  - the ratio of a trial's t-statistic to its ICIR still implies the
    approximate formation sample size (it grows like the square root of the
    number of IC dates);
  - it cannot detect subtle distributional fingerprints of a period.
  The rendered prompt hashes of every call are logged for audit.
- **Prompt steering.** The v1 template gave example mechanisms ("liquidity
  provision", "investor attention", "overreaction") that match the stories of
  the textbook planted signals. v2, the default, gives none; v1 is kept for
  the record and has never been used in an experiment.
- **Unequal search spaces (RQ1 comparability).** All arms share the
  language, the validator's limits, the budget and the evaluation, but not the
  reachable space: the LLM may propose anything the validator accepts
  (windows up to 252 sessions, depth up to 10, any literal within the limits),
  while both baselines draw windows from (1, 2, 3, 5, 10, 20, 40, 60),
  literals from {0.5, 1, 2} and exponents from {0.5, 2}, and random trees
  have depth at most 4. Library factors such as `momentum_12_1` are outside
  the baselines' support. An LLM advantage could partly reflect this larger
  space; results will be reported with this caveat, and a baseline with a
  wider menu is a candidate robustness arm.
- **Information barrier.** The harness gives proposers formation statistics
  only, and for store data the loader does not read later windows until
  their phase. It is an interface, not a sandbox: in-process code could read
  anything in memory, and for synthetic data the full panel is in memory. An
  LLM receives nothing but the rendered prompt.
- **Multiple testing and researcher degrees of freedom.** Within a run, every
  budgeted trial enters the formation screen's family, and the claim rests on
  the validation confirmation. Across runs, degrees of freedom come from
  prompt versions, budgets, arms, universes and exploratory `evaluate` calls.
  This is controlled by:
  - pre-registration, published externally before the first LLM call;
  - the study registry (fixed windows, counted exploration, capped reveals);
  - Holm correction across arm comparisons;
  - reporting all runs, including incomplete and aborted ones.
  Prompt templates are versioned and hashed, so any change to a template
  shows up as a new SHA-256 in every later run record. The plan requires such
  a change to be made as a new prompt version and disclosed.
- **Costs, capacity and execution realism.** The long-short evaluation:
  - trades at the close t + lag at a flat 10 bps per unit of one-way
    turnover;
  - ignores borrow costs, market impact and weight drift;
  - uses vendor daily closes, not official auction prints;
  - uses `vwap_proxy`, which is not a true VWAP.
  Economic significance claims need cost sensitivity and capacity analysis;
  IC-based claims are less sensitive to these choices.
- **LLM nondeterminism and model drift.** There are no sampling controls, so
  replicates vary. Served model ids are logged and server-side fallback is off
  by default. Every outcome of every call, failures included, is recorded in
  order, so every run replays exactly; record files are write-only, so a new
  live run is always a new sample. Each benchmark cell has its own request
  tag (SNR and market seed), so cells never share cached responses.
- **Synthetic benchmark design.** The planted signals, SNR levels and the
  data-generating process were chosen by the authors (the drawn signal by the
  procedure of `benchmark/draw.py`, fixed in code before the draw was run).
  Recall uses every 5th formation date. Recovery depends on a correlation
  threshold (0.7) and the true FDP on another (0.1). Synthetic rankings need
  not transfer to real markets.
- **Shared proposal streams.** In the harness grid the proposer seed is
  10000 + market seed at every SNR level, so the random arm evaluates the
  same expressions on the SNR 0.05, SNR 0.15 and null market of each seed,
  and GP starts from the same first batch (common random numbers). Those
  cells are not independent samples of the proposer. Whether H2's
  confirmatory runs use cell-specific proposer seeds is to be fixed when the
  plan is frozen.
- **Short windows.** A 147-date synthetic validation window and a 148-date
  test window limit power: at SNR 0.05 the confirmation step selected factors
  in only one run per arm. Real-data windows should span several years, and
  power is reported alongside the test non-significance rate.

## 8. Compute and cost budget

LLM cost is estimated from measured usage, not assumed:

```
cost = Σ_runs Σ_calls ( T_in(call) · p_in + T_out(call) · p_out )
n_calls(run) ≥ ⌈B / b⌉ + (pause_turn continuations)
```

- **Terms.**
  - B is the trial budget and b the batch size.
  - T_in and T_out are input and output tokens. Thinking tokens are billed as
    output.
  - More calls are needed when the model returns duplicates, bounded by
    `max_rounds = max(100, 10⌈B/b⌉)` (100 at B = 200, b = 20).
- **Measuring tokens.** Take T_in and T_out from a pilot run's
  `provenance.json` (`proposer_provenance.usage_totals`) or from the API's
  token counting.
- **Prices.** p_in and p_out **must be read from the provider's current
  pricing page at run time.** No price is assumed in this repository.
- **Replays are free.** Replays of recorded responses make no API calls.
- **Measured prompt size** (2026-09-25, v2 templates):
  - the rendered system prompt is 908 characters;
  - the user prompt is 3,012 characters in round 0 and about 6,000
    characters with full feedback blocks (8 best, 4 worst, 10 from the
    previous round, 8 rejected; measured with 45-character expressions).
  Output size depends on thinking length and must be measured.
- **Planned call volume** (to be confirmed by the pilot). With B = 200 and
  b = 20, each run needs at least 10 calls:
  - synthetic: (20 seeds × 2 SNR levels + 10 null markets) × 10 ≈ 500 calls;
  - real data: 2 universes × 5 replicates ≈ 100 calls;
  - ablations: a few hundred more.
- **Evaluation compute.** 200-trial runs on the 60 × 750 synthetic panel took
  about 8 to 14 s each, scoring included (`runtime_seconds` in
  `results/synthetic_benchmark/summary.json`). Real panels are larger, and
  their timings will be measured and reported.

## 9. Timeline (targets, not commitments)

| Phase | Target window | Deliverable |
|---|---|---|
| 0: harness | done (2026-09) | DSL, evaluator, protocol, baselines A/B, synthetic benchmark, tests |
| 1: protocol hardening | 2026-10 | library baseline C, ablation wrappers, block-bootstrap tests for H1/H3, drawdown/trade-count metrics; plan frozen and its hash published externally **before the first LLM call** (the reveal register and the `reveal` CLI exist) |
| 2: synthetic LLM study | 2026-11 | LLM arm on seeds 100–119 × 2 SNR levels plus null markets, with recorded responses; compute-matched arm F; H2 and synthetic parts of H4 |
| 3: real-data preparation | 2026-12 to 2027-01 | universes, confirmed-bar download through `quant_marketdata`, data-quality report, formation/validation-only runs |
| 4: sealed reveal | 2027-02 | pre-registered test reveals; H1 and H3; result cards |
| 5: writing | 2027-03 to 2027-04 | manuscript; prospective window accrues from the commitment date for a later version |

## 10. Candidate venues (targets only)

No submission has been made. Fit and deadlines will be checked against each
call for papers.

- **Conferences:** ACM International Conference on AI in Finance (ICAIF);
  the KDD Applied Data Science track; NeurIPS or ICML workshops on machine
  learning in finance or on LLM agents.
- **Journals:** the Journal of Financial Data Science; Quantitative Finance.

## References

- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2017). The probability of backtest overfitting. *Journal of Computational Finance*, 20(4). *(metadata to be verified before submission)*
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Benjamini, Y., & Yekutieli, D. (2001). The control of the false discovery rate in multiple testing under dependency. *Annals of Statistics*, 29(4), 1165–1188.
- Glasserman, P., & Lin, C. (2023). Assessing look-ahead bias in stock return predictions generated by GPT sentiment analysis. arXiv:2309.17322. *(metadata to be verified before submission)*
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of Business & Economic Statistics*, 23(4), 365–380.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). ... and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Ji, Z., et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys*, 55(12).
- Kakushadze, Z. (2016). 101 formulaic alphas. *Wilmott*, 2016(84), 72–81.
- Koza, J. R. (1992). *Genetic Programming*. MIT Press.
- Lopez-Lira, A., & Tang, Y. (2023). Can ChatGPT forecast stock price movements? arXiv:2304.07619.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Romano, J. P., & Wolf, M. (2005). Stepwise multiple testing as formalized data snooping. *Econometrica*, 73(4), 1237–1282.
- Romera-Paredes, B., et al. (2024). Mathematical discoveries from program search with large language models. *Nature*, 625, 468–475.
- Wang, S., et al. (2023). Alpha-GPT: Human-AI interactive alpha mining for quantitative investment. arXiv:2308.00016. *(metadata to be verified before submission)*
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5), 1097–1126.
- Wu, S., et al. (2023). BloombergGPT: A large language model for finance. arXiv:2303.17564.
- Yu, S., et al. (2023). Generating synergistic formulaic alpha collections via reinforcement learning. *KDD 2023*. *(metadata to be verified before submission)*
