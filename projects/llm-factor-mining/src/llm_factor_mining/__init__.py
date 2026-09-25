"""LLM-guided formulaic factor mining on the MarketData Quant Suite.

Layers: the typed factor language (:mod:`.dsl`), confirmed daily panels
(:mod:`.data`), causal leakage-aware evaluation (:mod:`.evaluate`), proposers
(:mod:`.proposers`), the research protocol - splits, trial ledger, sealed
hold-out, contamination checks - (:mod:`.protocol`), multiple-testing
inference (:mod:`.inference`), the search loop (:mod:`.search`) and the
synthetic planted-alpha benchmark (:mod:`.benchmark`).
"""

from ._version import __version__
from .data import Panel, PanelError, load_confirmed_panel, panel_from_bars
from .dsl import DSLLimits, ParseError, parse, validate
from .evaluate import EvaluationWindow, Evaluator, MetricConfig, evaluate_factor
from .protocol import FrozenFactorSet, SealedHoldout, Splits, TrialLedger, chronological_splits
from .search import SearchConfig, SearchResult, run_search

__all__ = [
    "DSLLimits",
    "EvaluationWindow",
    "Evaluator",
    "FrozenFactorSet",
    "MetricConfig",
    "Panel",
    "PanelError",
    "ParseError",
    "SealedHoldout",
    "SearchConfig",
    "SearchResult",
    "Splits",
    "TrialLedger",
    "__version__",
    "chronological_splits",
    "evaluate_factor",
    "load_confirmed_panel",
    "panel_from_bars",
    "parse",
    "run_search",
    "validate",
]
