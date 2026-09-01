# Research governance

## Evidence ladder

1. Unit and integration tests establish software behavior.
2. Synthetic data establishes end-to-end plumbing only.
3. Formation/validation splits support model selection.
4. Untouched forward data evaluates the frozen design.
5. Paper trading tests data arrival, operational controls, and executable costs.
6. Capital deployment requires independent risk approval and broker reconciliation.

Do not promote a result by skipping a level.

## Minimum result card

Every public result should disclose:

- hypothesis and economic mechanism;
- universe construction and survivorship limitations;
- information cutoff and execution lag;
- train, validation, and forward periods;
- costs, borrow, slippage, turnover, and capacity assumptions;
- benchmark and ablations;
- CAGR/return, volatility, Sharpe, drawdown, turnover, and trade count;
- sensitivity, multiple-testing, and failure diagnostics;
- data and code snapshot identifiers.

## Agentic research

Agents may propose bounded changes and explain deterministic outputs. They may not inspect sealed results before reveal, approve their own proposals, bypass risk gates, or place live orders without explicit human confirmation.
