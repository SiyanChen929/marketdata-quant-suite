# Architecture

The platform is the shared operating layer for a portfolio of focused research projects. Strategy repositories own hypotheses and domain logic; the platform owns common data contracts, validation, portfolio/risk controls, execution assumptions, and evidence production.

## Boundaries

| Layer | Owns | Does not own |
|---|---|---|
| Data plane | canonical schemas, provider adapters, snapshots, finality, checksums | strategy-specific features |
| Strategy plug-ins | factors, signals, invalidation rules | duplicated backtest engines |
| Portfolio/risk | sizing, exposures, drawdown and quality gates | data-vendor credentials |
| Backtest/execution | lagged fills, costs, ledger, attribution | claims of live executability |
| Governance | run IDs, manifests, validation gates, human approval | autonomous promotion |
| Interface | dashboards and explanations | changing deterministic calculations |

## Research plug-ins

- Statistical arbitrage: equity and commodity pairs.
- Event-driven: index-rebalance microstructure.
- Macro regimes: agriculture and alternative data.
- Factor discovery: typed factor definitions with sealed evaluation.
- Volatility: ARIMA/GARCH calibration and stress coverage.

Each plug-in consumes an immutable snapshot ID and emits a standard signal frame. It should not download another copy of the same US price history.

## Standard signal frame

Minimum fields:

```text
decision_time, effective_time, symbol, strategy_id, signal, score,
target_horizon, invalidation, data_snapshot_id, feature_version
```

The portfolio engine may reject or scale a signal. The final order decision therefore remains distinct from the raw research signal.
