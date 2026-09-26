# Project selection

## Included

### Research line 1: LLM-guided factor mining

`llm-factor-mining` asks whether LLM-guided program search finds equity factors that survive multiple-testing control over every trial and a single sealed test more often than random grammar search and genetic programming at the same budget, and how much of any edge is memorization. It contributes a typed, look-ahead-free factor language, a hash-chained trial ledger, two-stage selection, a commit-then-reveal hold-out and a planted-alpha benchmark with null markets.

### Research line 2: MarketData Agent

`marketdata-agent` asks whether a tool-using LLM copilot can answer quantitative market questions with verifiable numeric grounding while respecting point-in-time and execution constraints. It contributes a governed runtime (an as-of clock that refuses later dates, a policy gate that cannot enable execution, hash-chained audit), a grounding verifier and a benchmark with cutoff and trade-request traps.

Both are described together in the [research agenda](research-agenda.md).

### Quant research platform

The flagship shows end-to-end engineering: data quality, point-in-time features, strategy plug-ins, portfolio construction, risk, next-session execution, transaction costs, walk-forward validation, and research evidence.

### Equity pairs research

This study contributes statistical depth: sector-aware screening, cointegration, false-discovery control, causal lags, borrow and turnover costs, nested validation, capacity diagnostics, and transparent negative out-of-sample results.

### Index rebalance event study

The public project is a compact event-study template: a provider-neutral event schema, confirmed-bar retrieval, date-balanced portfolios, explicit costs, volatility diagnostics, and execution limitations. It preserves the design lessons from a larger private study, where licensed inputs supported matched controls, placebo tests, and rejection of a directional thesis after costs; those private results are not reproduced by the synthetic public fixture.

## Deferred extensions

- Agricultural macro regimes have strong revision-aware research design, though futures prices and macro vintages require specialist sources.
- China commodity pairs require domestic futures contract, roll, and session semantics.
- CSI 300 factor evolution requires point-in-time A-share membership, adjustment factors, trading-status fields, and China-specific execution metadata.
- The volatility-forecasting course project can become a compact risk-model module after its data interface and tests are modernized.
- Other agentic prototypes stay deferred until they carry their own research question, ground truth and tests.

## Why the LLM work is now standalone

An earlier version of this page held that trading-copilot and agentic-evolver prototypes were better represented as interfaces and governance features within the flagship than as separate projects. That is superseded for the two research lines above, for four reasons:

1. **Each has its own research question and evidence.** Both carry written hypotheses with decision rules, in research plans that are still drafts: neither plan has been frozen or registered outside this repository yet (the factor-mining plan is to be frozen, and its hash published externally, before the first model call). Each also has a benchmark with independent ground truth, committed synthetic results and a paper draft. A feature inside the flagship would have none of these.
2. **Their evaluation needs isolation.** Sealed test windows, a reveal register, trial ledgers and recorded model calls are experimental controls. Mixed into the platform's strategy workflow, they would be easy to bypass and hard to audit.
3. **The LLM dependency must stay optional.** Both projects keep the model SDK in an optional extra and pass their tests offline with fake clients. The flagship and the case studies stay free of it.
4. **Part of the governance rule became code.** The platform's rule says that agents may not inspect sealed results, approve their own proposals, bypass risk gates or place live orders. The parts against inspecting sealed results, self-approval and placing orders are enforced and tested there: proposers see formation data only and the test window stays sealed until a committed reveal in `llm-factor-mining`; in `marketdata-agent` the policy gate refuses every execution tool, order proposals stay inert and pending human approval, and the audit log records every attempt. Neither project has a risk gate, so that part of the rule stays with the platform.

Both still consume the shared contract: confirmed bars through `quant_marketdata` only, and no second price source.

The public story stays focused: one data plane, one reusable engine, two case studies that supply the empirical discipline, and two research lines on trustworthy AI for quantitative research.
