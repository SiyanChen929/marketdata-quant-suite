"""Typed configuration for the research pipeline."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ResearchConfig:
    years: int = 8
    formation_years: int = 5
    top_pairs_per_sector: int = 10
    minimum_history_ratio: float = 0.90
    minimum_formation_observations: int = 756
    cointegration_alpha: float = 0.05
    maximum_half_life_days: int = 252
    adf_maxlag: int = 5
    exclude_same_issuer: bool = True
    workers: int = 4


@dataclass(frozen=True)
class StrategyConfig:
    zscore_method: str = "sma"
    zscore_lookback: int = 60
    zscore_min_periods: int = 40
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0
    maximum_holding_days: int = 60
    transaction_cost_bps: float = 5.0
    transaction_cost_scenarios_bps: tuple[float, ...] = (2.5, 5.0, 10.0, 20.0)
    annual_short_borrow_bps: float = 30.0
    minimum_trades_for_ranking: int = 3


@dataclass(frozen=True)
class MacroConfig:
    series: dict[str, str] = field(
        default_factory=lambda: {
            "DFF": "Effective Federal Funds Rate",
            "DGS2": "2-Year Treasury Yield",
            "DGS10": "10-Year Treasury Yield",
            "CPIAUCSL": "Consumer Price Index",
            "UNRATE": "Unemployment Rate",
            "USREC": "NBER Recession Indicator",
            "VIXCLS": "CBOE Volatility Index",
        }
    )


@dataclass(frozen=True)
class PipelineConfig:
    research: ResearchConfig = field(default_factory=ResearchConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    macro: MacroConfig = field(default_factory=MacroConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reject_unknown(section: str, values: dict[str, Any], cls: type) -> None:
    valid = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - valid)
    if unknown:
        raise ValueError(f"Unknown {section} configuration keys: {', '.join(unknown)}")


def validate_config(config: PipelineConfig) -> PipelineConfig:
    r = config.research
    s = config.strategy
    if not isinstance(r.years, int) or isinstance(r.years, bool) or r.years < 3:
        raise ValueError("research.years must be at least 3")
    if not isinstance(r.formation_years, int) or isinstance(r.formation_years, bool) or not 1 <= r.formation_years < r.years:
        raise ValueError("research.formation_years must be between 1 and years - 1")
    if not isinstance(r.top_pairs_per_sector, int) or isinstance(r.top_pairs_per_sector, bool) or r.top_pairs_per_sector < 1:
        raise ValueError("research.top_pairs_per_sector must be positive")
    if not math.isfinite(r.minimum_history_ratio) or not 0 < r.minimum_history_ratio <= 1:
        raise ValueError("research.minimum_history_ratio must be in (0, 1]")
    if not isinstance(r.minimum_formation_observations, int) or isinstance(r.minimum_formation_observations, bool) or r.minimum_formation_observations < 100:
        raise ValueError("research.minimum_formation_observations must be at least 100")
    if not math.isfinite(r.cointegration_alpha) or not 0 < r.cointegration_alpha < 1:
        raise ValueError("research.cointegration_alpha must be in (0, 1)")
    if not isinstance(r.maximum_half_life_days, int) or isinstance(r.maximum_half_life_days, bool) or r.maximum_half_life_days < 2:
        raise ValueError("research.maximum_half_life_days must be at least 2")
    if not isinstance(r.workers, int) or isinstance(r.workers, bool) or r.workers < 1:
        raise ValueError("research.workers must be positive")
    if not isinstance(r.adf_maxlag, int) or isinstance(r.adf_maxlag, bool) or r.adf_maxlag < 0:
        raise ValueError("research.adf_maxlag cannot be negative")
    if not isinstance(s.zscore_lookback, int) or isinstance(s.zscore_lookback, bool) or s.zscore_lookback < 20:
        raise ValueError("strategy.zscore_lookback must be at least 20")
    if s.zscore_method not in {"sma", "ewma"}:
        raise ValueError("strategy.zscore_method must be 'sma' or 'ewma'")
    if not isinstance(s.zscore_min_periods, int) or isinstance(s.zscore_min_periods, bool) or not 2 <= s.zscore_min_periods <= s.zscore_lookback:
        raise ValueError("strategy.zscore_min_periods must be between 2 and zscore_lookback")
    if not all(math.isfinite(value) for value in (s.exit_z, s.entry_z, s.stop_z)) or not 0 <= s.exit_z < s.entry_z < s.stop_z:
        raise ValueError("require 0 <= exit_z < entry_z < stop_z")
    if not isinstance(s.maximum_holding_days, int) or isinstance(s.maximum_holding_days, bool) or s.maximum_holding_days < 1:
        raise ValueError("strategy.maximum_holding_days must be positive")
    if not math.isfinite(s.transaction_cost_bps) or not math.isfinite(s.annual_short_borrow_bps) or s.transaction_cost_bps < 0 or s.annual_short_borrow_bps < 0:
        raise ValueError("cost assumptions cannot be negative")
    if not s.transaction_cost_scenarios_bps or any(
        not math.isfinite(value) or value < 0 for value in s.transaction_cost_scenarios_bps
    ):
        raise ValueError("strategy.transaction_cost_scenarios_bps must contain finite non-negative values")
    if not isinstance(s.minimum_trades_for_ranking, int) or isinstance(s.minimum_trades_for_ranking, bool) or s.minimum_trades_for_ranking < 1:
        raise ValueError("strategy.minimum_trades_for_ranking must be positive")
    return config


def load_config(path: str | Path | None = None) -> PipelineConfig:
    """Load a JSON config, applying dataclass defaults for omitted fields."""
    if path is None:
        return validate_config(PipelineConfig())
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed_sections = {"research", "strategy", "macro"}
    unknown_sections = sorted(set(raw) - allowed_sections)
    if unknown_sections:
        raise ValueError(f"Unknown configuration sections: {', '.join(unknown_sections)}")
    research = raw.get("research", {})
    strategy = raw.get("strategy", {})
    macro = raw.get("macro", {})
    _reject_unknown("research", research, ResearchConfig)
    _reject_unknown("strategy", strategy, StrategyConfig)
    _reject_unknown("macro", macro, MacroConfig)
    strategy = dict(strategy)
    if "transaction_cost_scenarios_bps" in strategy:
        strategy["transaction_cost_scenarios_bps"] = tuple(
            strategy["transaction_cost_scenarios_bps"]
        )
    return validate_config(
        PipelineConfig(
            research=ResearchConfig(**research),
            strategy=StrategyConfig(**strategy),
            macro=MacroConfig(**macro),
        )
    )
