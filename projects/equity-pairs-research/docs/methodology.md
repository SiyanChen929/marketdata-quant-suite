# Equity pairs trading: research and implementation methodology

## Executive decision

This project implements a transparent cointegration strategy rather than an
opaque optimizer.  Within each current GICS sector, it exhaustively tests every
different-issuer pair, estimates a log-price equilibrium, controls the
sector-level false discovery rate, and trades held-out deviations from that
equilibrium.  The construction is intentionally auditable: a researcher can
trace every selected pair from raw prices, through the tests and hedge ratio, to
every next-session position, cost, and return.

The literal top ten pairs in each sector over all eight years are a descriptive
answer only.  They are not backtested on the same eight years and presented as
out-of-sample evidence.  Tradable candidates are selected using the first five
years, then frozen and tested over the final three years.  The final "best ten"
is the user's requested ex-post CAGR ranking of those held-out pair tests; it is
clearly labeled as a research ranking rather than a deployable selection rule.

## Widely used strategy families

| Family | Pair selection / model | Typical signal | Strengths | Main failure modes |
|---|---|---|---|---|
| Distance | Minimize squared distance between normalized price paths | Trade a two-standard-deviation divergence toward convergence | Simple, scalable, historically canonical | No formal equilibrium relation; scale and formation-window sensitivity |
| Cointegration / error correction | Engle-Granger or Johansen stationary linear combination of non-stationary prices | Z-score of the estimated equilibrium error | Hedge ratio is explicit; equilibrium is statistically testable | Test asymmetry, structural breaks, multiple testing, unstable hedge ratios |
| Time-series / OU | Fit an AR(1), Ornstein-Uhlenbeck, or state-space model to a spread | Entry/exit from estimated mean, volatility, and half-life | Direct link from dynamics to thresholds and holding period | Gaussian/stationary assumptions; parameter error |
| Kalman / dynamic hedge | State-space model with time-varying intercept and beta | Standardized innovation or dynamic residual | Adapts to slow relationship drift | Easy to overfit; filter-noise choices materially change turnover |
| Factor/PCA residual stat-arb | Remove market/sector/common-component returns | Trade idiosyncratic residual mean reversion across a basket | Diversified and scalable beyond pairs | Crowding, factor instability, portfolio/constraint complexity |
| Copula | Model marginal distributions and nonlinear dependence | Conditional-tail mispricing indices | Can represent asymmetric and tail dependence | Model selection risk and parameter instability; often high complexity for limited incremental edge |
| ML/clustering/graph | Use features, embeddings, clustering, or network matching to preselect economically similar names | Usually combined with a residual or spread model | Reduces search space; can incorporate fundamentals | Leakage, unstable clusters, explainability, large validation burden |
| Stochastic control | Optimize continuous holdings under a spread process and costs | State-dependent optimal trading bands | Directly accounts for risk and costs | Strong model assumptions; estimation and implementation complexity |

Gatev, Goetzmann, and Rouwenhorst's canonical distance study used a formation
period followed by a distinct trading period and found that market
microstructure and costs matter.  Engle and Granger provide the equilibrium and
error-correction foundation for the implemented approach.  Krauss's survey
organizes the literature into distance, cointegration, time-series, stochastic
control, and other approaches.  Avellaneda and Lee show the related,
portfolio-level PCA residual approach and document meaningful decay across
subperiods—an important warning against treating a historical edge as static.

## Implemented statistical screen

For confirmed daily close observations \(P^y_t\) and \(P^x_t\), the formation relation is

\[
\log(P^y_t) = \alpha + \beta \log(P^x_t) + \epsilon_t.
\]

The pipeline applies Engle-Granger in both orientations.  It chooses the
orientation with the lower residual-unit-root p-value for hedge construction,
but Bonferroni-adjusts that two-way choice:

\[
p_{pair} = \min(1, 2\min(p_{y|x},p_{x|y})).
\]

Within each sector, Benjamini-Hochberg adjusted q-values control the false
discovery rate across the many tested pairs.  Strict selections additionally
require:

- sufficient overlapping price history;
- both log-price series to look integrated of order one (ADF does not reject a
  level unit root, but rejects a return unit root);
- a positive hedge ratio;
- estimated mean-reversion half-life between 2 and 252 trading days; and
- a residual Hurst estimate below 0.5.

If a sector has fewer than ten strict pairs, the output is deterministically
filled with the next-lowest pair p-values and marks those rows `fallback`.  This
meets the requested ten-per-sector coverage without pretending weaker evidence
passed the strict gate.

## Signal, execution, and costs

The held-out spread uses the frozen formation-window \(\alpha\) and \(\beta\).
Its z-score uses a trailing 60-session mean and sample standard deviation, with
40 observations required:

\[
z_t = \frac{\epsilon_t-\bar{\epsilon}_{t,60}}
{s(\epsilon)_{t,60}}.
\]

Default rules are:

- enter long spread when \(z_t \le -2\);
- enter short spread when \(z_t \ge +2\);
- exit when \(|z_t| \le 0.5\);
- stop when \(|z_t| \ge 4\); and
- force exit after 60 trading days.

A close observed on date \(t\) changes holdings only for the \(t+1\)
close-to-close return.  Leg weights are gross-normalized:

\[
w_y=\frac{1}{1+|\beta|}, \qquad
w_x=\frac{-\beta}{1+|\beta|}.
\]

spread.  The backtest rebalances to those gross-normalized targets each close
while a trade is active, counts the drift correction as turnover, and charges
it.  Daily net return deducts configurable one-way transaction cost from
absolute leg turnover and a configurable annual borrow charge from short
notional.  Defaults are 5 basis points of each dollar traded and 30 basis points
annualized on the short leg.  These assumptions are research proxies, not a
substitute for symbol-specific bid/ask, market impact, locate availability, and
borrow fees.

## Portfolios and interpretation

Three curves are produced:

1. an equal-capital portfolio of every formation-selected pair;
2. a sector-balanced portfolio that weights each sector equally, then its pairs
   equally; and
3. an exploratory equal-capital portfolio of the ten highest held-out net-CAGR
   pairs.

The third portfolio is selected with knowledge of test results.  It visualizes
the requested winners but is not an unbiased estimate of a rule that could have
been known at the start of the test.  The first two are the defensible aggregate
held-out curves under the fixed formation selection.

## FRED macro treatment

FRED series describe performance regimes (rates, 2s10s slope, volatility,
inflation, unemployment, and recessions); they do not gate trades in the default
strategy.  Standard FRED observations are today's revised vintage.  In
particular, monthly macro series and the NBER recession indicator cannot be used
as contemporaneously known signals without ALFRED vintages and release-time
alignment.  The report therefore labels these joins as ex-post attribution.
The FRED key is read only from the `FRED_API_KEY` environment variable; keys are
never accepted from repository files or written to snapshot metadata.

## What productization still requires

- Point-in-time index constituents and historical sector classifications.
  Screening today's S&P 500 backward introduces survivorship and
  membership-selection bias.  Same-CIK share classes are excluded, but that does
  not solve historical membership bias.
- Delisted-security total returns, audited corporate actions, exchange calendars,
  and bad-tick controls appropriate to an execution book of record. The shared
  `quant_marketdata` package supplies confirmed MarketData daily bars and keeps
  their cache outside this project repository.
- Point-in-time borrow availability/fees, bid/ask and impact models tied to
  intended order size, and realistic execution timing.
- Rolling re-formation evaluated with purged/nested walk-forward validation,
  parameter stability and structural-break monitoring, factor/beta neutrality,
  overlapping-name and sector exposure constraints, liquidity capacity, and
  kill switches.
- Paper trading and reconciliation against an independent data/execution source
  before any capital is put at risk.

No backtest can guarantee or "maximize" future profit.  The useful output is a
falsifiable, cost-aware research result with a clear path from prototype to a
controlled live experiment.

## Primary references and documentation

- Engle, R. F., and Granger, C. W. J. (1987), *Co-integration and Error
  Correction: Representation, Estimation, and Testing*:
  https://ideas.repec.org/a/ecm/emetrp/v55y1987i2p251-76.html
- Gatev, E., Goetzmann, W. N., and Rouwenhorst, K. G., *Pairs Trading:
  Performance of a Relative Value Arbitrage Rule*:
  https://www.nber.org/papers/w7032
- Avellaneda, M., and Lee, J.-H. (2010), *Statistical Arbitrage in the U.S.
  Equities Market*: https://doi.org/10.1080/14697680903124632
- Elliott, R. J., van der Hoek, J., and Malcolm, W. P. (2005), *Pairs Trading*:
  https://doi.org/10.1080/14697680500149370
- Krauss, C. (2017), *Statistical Arbitrage Pairs Trading Strategies: Review and
  Outlook*: https://doi.org/10.1111/joes.12153
- statsmodels cointegration and unit-root API:
  https://www.statsmodels.org/stable/api.html
- MarketData API documentation:
  https://www.marketdata.app/docs/api/
- FRED observations and vintage documentation:
  https://fred.stlouisfed.org/docs/api/fred/series_observations.html and
  https://fred.stlouisfed.org/docs/api/fred/series_vintagedates.html
- S&P U.S. Indices methodology:
  https://www.spglobal.com/spdji/en/methodology/article/sp-us-indices-methodology/
