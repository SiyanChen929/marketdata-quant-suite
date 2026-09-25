"""Reference catalog of well-known formulaic signal *families* written in the DSL.

Most entries are this project's own renderings of textbook ideas (momentum,
reversal, low volatility, liquidity, volume-price interaction).  Three are
translations of published formulas from Kakushadze (2016), "101 Formulaic
Alphas" - Alpha#2, Alpha#6 and Alpha#13 - into the DSL (``rank`` ->
``cs_rank``, ``correlation`` -> ``ts_corr``, ``covariance`` -> ``ts_cov``);
their ``source`` field names the alpha.  Other entries tagged
``kakushadze_style`` are the project's own variants in the style of that
catalog, not reproductions.  (Attributions were made from the published
formulas and should be re-checked against the paper before submission.)
The catalog serves two roles: seed baselines for evaluation and a reference
set for novelty scoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from .canonical import structural_hash
from .nodes import Node
from .parser import parse


@dataclass(frozen=True)
class LibraryFactor:
    """One named reference expression."""

    name: str
    expression: str
    family: str
    rationale: str
    kakushadze_style: bool = False
    source: str = "project rendering of a textbook idea"

    @property
    def node(self) -> Node:
        return parse(self.expression)

    @property
    def hash(self) -> str:
        return structural_hash(self.node)


LIBRARY: tuple[LibraryFactor, ...] = (
    LibraryFactor(
        "momentum_12_1",
        "delay(close, 21) / delay(close, 252) - 1",
        "momentum",
        "Twelve-month price momentum skipping the most recent month (approx. 21 sessions).",
    ),
    LibraryFactor(
        "momentum_6_1",
        "delay(close, 21) / delay(close, 126) - 1",
        "momentum",
        "Six-month price momentum skipping the most recent month.",
    ),
    LibraryFactor(
        "reversal_5d",
        "-(close / delay(close, 5) - 1)",
        "reversal",
        "Short-term (one-week) reversal: recent losers are expected to rebound.",
    ),
    LibraryFactor(
        "reversal_21d",
        "-(close / delay(close, 21) - 1)",
        "reversal",
        "One-month reversal.",
    ),
    LibraryFactor(
        "low_volatility_20d",
        "-ts_std(returns, 20)",
        "volatility",
        "Low realized-volatility anomaly over one month of daily returns.",
    ),
    LibraryFactor(
        "low_volatility_60d",
        "-ts_std(returns, 60)",
        "volatility",
        "Low realized-volatility anomaly over roughly one quarter.",
    ),
    LibraryFactor(
        "range_volatility_20d",
        "-ts_mean((high - low) / close, 20)",
        "volatility",
        "Range-based (high-low) volatility proxy; lower range ranks higher.",
    ),
    LibraryFactor(
        "max_return_21d",
        "-ts_max(returns, 21)",
        "volatility",
        "Lottery-like extreme daily return over the last month; high maximum ranks lower.",
    ),
    LibraryFactor(
        "illiquidity_20d",
        "-log(ts_mean(dollar_volume, 20))",
        "liquidity",
        "Low average dollar volume (a size/liquidity proxy) ranks higher.",
    ),
    LibraryFactor(
        "abnormal_volume_20d",
        "volume / ts_mean(volume, 20) - 1",
        "liquidity",
        "Volume relative to its one-month average (attention proxy).",
    ),
    LibraryFactor(
        "open_volume_corr_10d",
        "-ts_corr(open, volume, 10)",
        "volume_price",
        "Negative rolling correlation between open price and volume.",
        kakushadze_style=True,
        source="Kakushadze (2016), Alpha#6",
    ),
    LibraryFactor(
        "volume_change_vs_intraday_return",
        "-ts_corr(cs_rank(delta(log(volume), 2)), cs_rank((close - open) / open), 6)",
        "volume_price",
        "Negative correlation of ranked volume changes with ranked intraday returns.",
        kakushadze_style=True,
        source="Kakushadze (2016), Alpha#2",
    ),
    LibraryFactor(
        "rank_volume_close_cov_5d",
        "-cs_rank(ts_cov(cs_rank(close), cs_rank(volume), 5))",
        "volume_price",
        "Negative ranked covariance of cross-sectional price and volume ranks.",
        kakushadze_style=True,
        source="Kakushadze (2016), Alpha#13",
    ),
    LibraryFactor(
        "typical_price_gap",
        "cs_rank(vwap_proxy - close) * cs_rank(ts_rank(volume, 10))",
        "volume_price",
        "Close below the typical price on relatively heavy volume.",
        kakushadze_style=True,
        source="project variant in the style of Kakushadze (2016)",
    ),
    LibraryFactor(
        "intraday_reversal_5d",
        "-cs_rank(ts_mean((close - open) / open, 5))",
        "reversal",
        "Reversal of the average open-to-close move over one week.",
        kakushadze_style=True,
        source="project variant in the style of Kakushadze (2016)",
    ),
    LibraryFactor(
        "decayed_reversal_10d",
        "-decay_linear(returns, 10)",
        "reversal",
        "Linearly decayed average of recent returns, sign-flipped.",
        kakushadze_style=True,
        source="project variant in the style of Kakushadze (2016)",
    ),
    LibraryFactor(
        "price_zscore_20d",
        "-ts_zscore(close, 20)",
        "reversal",
        "Mean reversion toward the one-month average price.",
    ),
    LibraryFactor(
        "high_recency_20d",
        "-ts_argmax(close, 20) / 19",
        "momentum",
        "A recent one-month high (small arg-max lag) ranks higher.",
        kakushadze_style=True,
        source="project variant in the style of Kakushadze (2016)",
    ),
)


def get(name: str) -> LibraryFactor:
    """Look up a library entry by name."""

    for factor in LIBRARY:
        if factor.name == name:
            return factor
    raise KeyError(f"unknown library factor {name!r}")


@lru_cache(maxsize=1)
def library_nodes() -> tuple[Node, ...]:
    """Parsed ASTs of the catalog, in catalog order."""

    return tuple(factor.node for factor in LIBRARY)
