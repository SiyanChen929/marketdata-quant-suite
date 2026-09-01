# Specialist extensions

The following local research programs are meaningful but are excluded from the one-MarketData suite until their primary price and reference-data domains have explicit adapters:

| Extension | Missing shared-domain requirement | Safe integration path |
|---|---|---|
| China commodity pairs | Domestic futures contracts, rolls, sessions, limits, and fees | Add a China-futures adapter and map it to the canonical bar schema with contract lineage |
| CSI 300 factor evolution | Point-in-time membership, A-share adjustment factors, trading status, industry, and corporate actions | Retain a China-specific provider and register immutable universe snapshots |
| Agricultural macro regimes | Commodity futures plus ALFRED/FRED, USDA, NOAA, and licensed series | Use specialist adapters; share only the portfolio/risk/evidence layers |
| Volatility forecasting | Modern package, provider adapter, tests, and data manifest | Refit against shared MarketData SPY bars as a risk-model plug-in |

The boundary is a data-quality decision. A single provider is valuable only within the asset classes and fields it actually supports.
