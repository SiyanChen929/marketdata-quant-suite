# Unified system blueprint

## Storyline

The portfolio follows a research idea from raw data to a decision record:

```text
MarketData request
  -> canonical, checksummed snapshot
  -> point-in-time feature or event panel
  -> strategy hypothesis
  -> cost-aware portfolio simulation
  -> chronological validation
  -> risk and execution diagnostics
  -> immutable evidence for human review
```

The five projects occupy different layers of that story:

- `quant-research-platform` supplies the reusable engine: ingestion interfaces, data-quality gates, signals, construction, risk, costs, validation, and reporting.
- `equity-pairs-research` is a statistical-arbitrage study. It shows cointegration testing, multiple-testing controls, nested chronological validation, portfolio allocation, and why an attractive ex-post subset is not evidence.
- `index-rebalance-event-study` is a compact public event-study template. It shows point-in-time event construction, date-balanced aggregation, cost inputs, volatility diagnostics, and market-microstructure limitations. Matched-control, placebo, and directional-rejection findings belong to the private source study and are not claimed as reproduced public results.
- `llm-factor-mining` puts a language model at the "strategy hypothesis" step. It counts every proposed factor as a trial in a hash-chained ledger, selects in two stages on formation and validation data, and reveals a committed test window once.
- `marketdata-agent` puts a language model at the "human review" step, as a question-answering copilot over the confirmed lake. Its reads are bounded by an as-of date, its numbers are checked against the tool outputs they cite, and it has no execution path.

The [research agenda](research-agenda.md) explains how the two LLM research lines relate to the three case-study projects.

## Boundaries

### Data plane

`quant_marketdata` owns provider authentication, HTTP requests, response normalization, checksums, finality, and physical storage. Strategy modules receive validated frames and never call a second market-price client.

The external data root is organized by meaning:

```text
QUANT_DATA_HOME/
  raw/marketdata/          immutable request-hash cache
  lake/confirmed/          completed-session research inputs
  staging/provisional/     uncertain or still-forming observations
  manifests/               request, schema, checksum, and snapshot metadata
```

### Research plane

Each run records its data interval, input snapshot, configuration, cost model, and validation split. Selection occurs before evaluation. The review date is the independent unit for rebalance research; chronological folds are the independent units for general strategy research.

### Model plane

Language models are optional, external components. They receive only what the harness renders: formation-window statistics for factor proposers, and point-in-time tool results for the copilot. Every call records the requested and served model, the stop reason and token usage, and runs can be replayed offline from their recordings. Experiment configurations keep server-side model fallback off, because it would change the model under test. Tests use injected fake clients and never reach the network.

### Execution plane

The current suite stops at research and paper-trading controls. The copilot can only record inert order proposals that wait for human approval outside the agent. Daily or intraday bars do not establish executable fills. A live boundary would additionally require broker state, order lifecycle management, idempotency, kill switches, borrow/locate checks, official auction data where relevant, and independent operational approval.

## Extension rule

Add a project only when it contributes a distinct research capability and can consume the shared contract. If its primary market is unsupported, implement a clearly named specialist adapter with its own lineage; do not disguise a proxy as equivalent data.
