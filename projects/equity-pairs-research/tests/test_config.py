from __future__ import annotations

from dataclasses import replace
import json

import pytest

from equity_pairs.config import (
    PipelineConfig,
    ResearchConfig,
    StrategyConfig,
    load_config,
    validate_config,
)


def test_load_config_applies_nested_defaults_and_serializes(tmp_path):
    path = tmp_path / "research.json"
    path.write_text(
        json.dumps(
            {
                "research": {"years": 10, "workers": 1},
                "strategy": {"entry_z": 2.5},
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.research.years == 10
    assert config.research.formation_years == ResearchConfig().formation_years
    assert config.research.workers == 1
    assert config.strategy.entry_z == 2.5
    assert config.strategy.exit_z == StrategyConfig().exit_z
    assert config.macro.series["DFF"] == "Effective Federal Funds Rate"
    assert config.to_dict()["strategy"]["entry_z"] == 2.5


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"unexpected": {}}, "Unknown configuration sections: unexpected"),
        ({"research": {"typo": 1}}, "Unknown research configuration keys: typo"),
        ({"strategy": {"typo": 1}}, "Unknown strategy configuration keys: typo"),
        ({"macro": {"typo": 1}}, "Unknown macro configuration keys: typo"),
    ],
)
def test_load_config_rejects_unknown_sections_and_keys(tmp_path, payload, message):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "research",
    [
        replace(ResearchConfig(), years=2),
        replace(ResearchConfig(), formation_years=0),
        replace(ResearchConfig(), formation_years=8),
        replace(ResearchConfig(), top_pairs_per_sector=0),
        replace(ResearchConfig(), minimum_history_ratio=0.0),
        replace(ResearchConfig(), minimum_history_ratio=1.01),
        replace(ResearchConfig(), minimum_formation_observations=99),
        replace(ResearchConfig(), cointegration_alpha=0.0),
        replace(ResearchConfig(), maximum_half_life_days=1),
        replace(ResearchConfig(), adf_maxlag=-1),
        replace(ResearchConfig(), workers=0),
    ],
)
def test_validate_config_rejects_invalid_research_settings(research):
    with pytest.raises(ValueError):
        validate_config(PipelineConfig(research=research))


@pytest.mark.parametrize(
    "strategy",
    [
        replace(StrategyConfig(), zscore_lookback=19),
        replace(StrategyConfig(), zscore_method="median"),
        replace(StrategyConfig(), zscore_min_periods=1),
        replace(StrategyConfig(), zscore_min_periods=61),
        replace(StrategyConfig(), exit_z=-0.1),
        replace(StrategyConfig(), exit_z=2.0),
        replace(StrategyConfig(), stop_z=2.0),
        replace(StrategyConfig(), maximum_holding_days=0),
        replace(StrategyConfig(), transaction_cost_bps=-0.1),
        replace(StrategyConfig(), annual_short_borrow_bps=-0.1),
        replace(StrategyConfig(), minimum_trades_for_ranking=0),
        replace(StrategyConfig(), transaction_cost_scenarios_bps=(-1.0,)),
    ],
)
def test_validate_config_rejects_invalid_strategy_settings(strategy):
    with pytest.raises(ValueError):
        validate_config(PipelineConfig(strategy=strategy))


def test_default_config_is_valid_and_boundary_values_are_accepted():
    assert load_config() == PipelineConfig()

    config = PipelineConfig(
        research=replace(
            ResearchConfig(),
            years=3,
            formation_years=2,
            minimum_history_ratio=1.0,
            cointegration_alpha=0.999,
            adf_maxlag=0,
            workers=1,
        ),
        strategy=replace(
            StrategyConfig(),
            zscore_lookback=20,
            zscore_min_periods=20,
            exit_z=0.0,
            transaction_cost_bps=0.0,
            annual_short_borrow_bps=0.0,
        ),
    )
    assert validate_config(config) is config
