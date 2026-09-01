"""YAML configuration loading for quant_system."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from quant_system.data.paths import expand_configured_path


@dataclass(frozen=True)
class ExecutionConfig:
    """Execution assumptions for daily backtests."""

    signal_timing: str = "close_to_next_open"
    fill_price: str = "open"
    slippage_bps: float = 5.0
    commission_per_share: float = 0.005
    short_borrow_fee_annual: float = 0.03
    cash_rate_annual: float = 0.0
    min_trade_notional: float = 1.0
    max_adv_participation: float = 0.0
    market_impact_bps_per_1pct_adv: float = 0.0
    market_impact_exponent: float = 1.25
    spread_bps: float = 0.0
    min_liquidity_cost_bps: float = 0.0
    daily_turnover_cap: float = 0.0


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio construction and exposure controls."""

    initial_capital: float = 100_000.0
    construction: str = "score_weighted"
    max_position_weight: float = 0.10
    target_gross_exposure: float = 1.0
    target_net_exposure: float = 0.5
    volatility_target: float | None = 0.25
    rebalance: str = "weekly"
    throttle_min_weight_change: float = 0.02
    throttle_min_score_change: float = 7.0
    throttle_min_new_weight: float = 0.005
    max_total_positions: int = 0
    min_target_weight: float = 0.0
    trade_location_overlay: bool = False
    trade_location_min_entry_quality: float = 55.0
    trade_location_min_reward_risk: float = 1.50
    trade_location_ideal_above_ma20_pct: float = 0.03
    trade_location_chase_max_above_ma20_pct: float = 0.12
    trade_location_pullback_min_drawdown: float = 0.02
    trade_location_pullback_max_drawdown: float = 0.16
    trade_location_gap_risk_reduce: float = 65.0
    trade_location_gap_risk_block: float = 90.0
    trade_location_event_risk_block: float = 70.0
    trade_location_block_new_extended_entries: bool = True
    trade_location_hold_extended_multiplier: float = 0.35
    trade_location_continuation_multiplier: float = 0.80
    trade_location_pullback_multiplier: float = 1.15
    leader_addon_overlay: bool = False
    leader_addon_multiplier: float = 1.20
    leader_addon_min_final_score: float = 70.0
    leader_addon_min_relative_strength_score: float = 80.0
    leader_addon_min_theme_score: float = 65.0
    leader_addon_min_mom_return: float = 0.10
    leader_addon_max_names_per_date: int = 5
    leader_addon_require_benchmark_risk_on: bool = True
    leader_addon_require_controlled_pullback: bool = False
    leader_addon_min_drawdown_from_high: float = 0.02
    leader_addon_max_drawdown_from_high: float = 0.15
    leader_addon_max_above_ma20_pct_circuit_breaker: float = 0.08
    leader_addon_max_event_risk_score_circuit_breaker: float = 20.0
    leader_addon_max_overnight_gap_risk_score_circuit_breaker: float = 30.0
    leader_addon_require_same_theme_peer_support: bool = False
    leader_addon_same_theme_peer_min_count: int = 2
    leader_addon_same_theme_peer_min_share: float = 0.20
    leader_addon_gap_quality_support_overlay: bool = False
    leader_addon_gap_quality_max_symbol_gap_risk_score: float = 20.0
    leader_addon_gap_quality_same_theme_max_gap_risk_score: float = 18.0
    leader_addon_gap_quality_same_theme_min_count: int = 2
    leader_addon_gap_quality_same_theme_min_share: float = 0.25
    leader_hold_buffer_overlay: bool = False
    leader_hold_buffer_min_final_score: float = 66.0
    leader_hold_buffer_min_relative_strength_score: float = 82.0
    leader_hold_buffer_min_theme_score: float = 60.0
    leader_hold_buffer_min_mom_return: float = 0.0
    leader_hold_buffer_require_theme_active: bool = False
    leader_hold_buffer_max_drawdown_from_high: float = 1.0
    leader_hold_buffer_weight_fraction: float = 0.5
    leader_hold_buffer_max_weight: float = 0.02
    leader_hold_buffer_max_days: int = 3
    leader_hold_buffer_require_benchmark_risk_on: bool = True
    leader_hold_buffer_max_event_risk_score_circuit_breaker: float = 25.0
    leader_hold_buffer_max_overnight_gap_risk_score_circuit_breaker: float = 35.0
    leader_delayed_exit_overlay: bool = False
    leader_delayed_exit_min_final_score: float = 70.0
    leader_delayed_exit_min_relative_strength_score: float = 90.0
    leader_delayed_exit_min_theme_score: float = 65.0
    leader_delayed_exit_min_mom_return: float = 0.15
    leader_delayed_exit_max_drawdown_from_high: float = 0.18
    leader_delayed_exit_retain_fraction: float = 0.85
    leader_delayed_exit_max_weight: float = 0.08
    leader_delayed_exit_max_days: int = 3
    leader_delayed_exit_require_benchmark_risk_on: bool = True
    leader_delayed_exit_require_theme_active: bool = False
    leader_delayed_exit_max_event_risk_score_circuit_breaker: float = 20.0
    leader_delayed_exit_max_overnight_gap_risk_score_circuit_breaker: float = 30.0
    leader_delayed_exit_stability_overlay: bool = False
    leader_delayed_exit_max_final_score_drop: float = 6.0
    leader_delayed_exit_max_relative_strength_drop: float = 5.0
    leader_persistence_overlay: bool = False
    leader_persistence_min_final_score: float = 68.0
    leader_persistence_min_relative_strength_score: float = 88.0
    leader_persistence_min_theme_score: float = 62.0
    leader_persistence_min_mom_return: float = 0.15
    leader_persistence_min_theme_peer_count: int = 2
    leader_persistence_min_theme_peer_share: float = 0.40
    leader_persistence_retain_fraction_of_cut: float = 0.35
    leader_persistence_max_weight_bonus: float = 0.015
    leader_persistence_require_theme_active: bool = True
    leader_persistence_require_benchmark_risk_on: bool = True
    leader_persistence_max_event_risk_score_circuit_breaker: float = 20.0
    leader_persistence_max_overnight_gap_risk_score_circuit_breaker: float = 30.0
    leader_persistence_stability_overlay: bool = False
    leader_persistence_min_desired_weight_fraction: float = 0.40
    leader_persistence_max_final_score_drop: float = 6.0
    leader_persistence_max_relative_strength_drop: float = 5.0
    leader_persistence_peer_strength_reward_overlay: bool = False
    leader_persistence_peer_strength_reward_min_theme_peer_count: int = 3
    leader_persistence_peer_strength_reward_min_theme_peer_share: float = 0.55
    leader_persistence_peer_strength_reward_min_relative_strength_score: float = 92.0
    leader_persistence_peer_strength_reward_retain_fraction_boost: float = 0.20
    leader_persistence_peer_strength_reward_max_weight_bonus: float = 0.0075
    leader_persistence_peer_strength_reward_max_drawdown_from_high_circuit_breaker: float = 0.14
    conviction_sizing_overlay: bool = False
    conviction_sizing_power: float = 1.25
    conviction_sizing_max_multiplier: float = 1.35
    conviction_sizing_min_names: int = 8
    conviction_sizing_min_final_score: float = 60.0
    conviction_sizing_min_relative_strength_score: float = 55.0
    conviction_sizing_min_theme_score: float = 50.0
    conviction_sizing_max_event_risk_score_circuit_breaker: float = 25.0
    conviction_sizing_max_overnight_gap_risk_score_circuit_breaker: float = 35.0
    conviction_sizing_require_benchmark_risk_on: bool = True
    conviction_sizing_require_same_theme_peer_support: bool = False
    conviction_sizing_allowed_themes: tuple[str, ...] = ()
    conviction_sizing_same_theme_peer_min_count: int = 3
    conviction_sizing_same_theme_peer_min_share: float = 0.45
    conviction_sizing_same_theme_peer_min_avg_relative_strength_score: float = 80.0
    conviction_sizing_same_theme_peer_min_avg_mom_return: float = 0.10


@dataclass(frozen=True)
class RiskConfig:
    """Risk controls applied by the portfolio and backtest engine."""

    max_drawdown_reduce_exposure: float = 0.12
    max_drawdown_cash_mode: float = 0.20
    drawdown_reduction_multiplier: float = 0.5
    atr_stop_multiple: float = 2.5
    trailing_stop_atr_multiple: float = 3.0
    drawdown_reset: str = "yearly"
    min_holding_days: int = 0
    intraday_enabled: bool = False
    intraday_fetch_from_provider: bool = False
    intraday_timeframe: str = "5min"
    intraday_fill_policy: str = "same_bar_close"
    intraday_start_date: str | None = None
    intraday_symbol_stop_loss_pct: float = 0.08
    intraday_portfolio_drawdown_limit: float = 0.06
    intraday_portfolio_exit_fraction: float = 0.50
    intraday_breadth_down_pct: float = 0.75
    intraday_breadth_exit_fraction: float = 0.35


@dataclass(frozen=True)
class UniverseConfig:
    """Universe settings separated into base, tradable, and candidate layers."""

    mode: str = "dynamic"
    base: str = "us_equities"
    include_etfs: bool = True
    symbols: tuple[str, ...] = ()
    custom_symbols: tuple[str, ...] = ()
    custom_only: bool = False
    base_symbols_path: str | None = None
    strict_backtest: bool = False
    benchmark: str = "SPY"
    min_price: float = 5.0
    aggressive_min_price: float = 2.0
    min_market_cap: float = 500_000_000.0
    min_avg_dollar_volume_20d: float = 20_000_000.0
    min_avg_volume_20d: float = 500_000.0
    min_history_days: int = 252
    exclude_otc: bool = True
    exclude_pink_sheets: bool = True
    exclude_leveraged_etfs: bool = True
    exclude_inverse_etfs: bool = True
    require_short_borrow_for_shorts: bool = True
    max_short_borrow_fee_annual: float = 0.10
    allow_speculative_bucket: bool = True
    speculative_bucket_max_gross: float = 0.10
    max_stale_price_days: int = 30
    max_unhandled_corporate_action_return: float = 0.80
    max_unhandled_corporate_action_count: int = 0
    min_quality_adjusted_history_days: int = 60
    quality_adjusted_position_multiplier: float = 0.35
    theme_baskets_path: str = "configs/themes.yml"


@dataclass(frozen=True)
class FundamentalsConfig:
    """Fundamental scoring settings."""

    enabled: bool = True
    provider: str = "csv"
    sector_neutral: bool = True
    winsorize_pct: float = 0.01
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "growth": 0.25,
            "quality": 0.25,
            "balance_sheet": 0.20,
            "valuation": 0.15,
            "revision": 0.15,
        }
    )
    long_min_score: float = 40.0
    short_max_score: float = 40.0


@dataclass(frozen=True)
class EventsConfig:
    """Event risk settings."""

    enabled: bool = True
    provider: str = "csv"
    earnings_mode: str = "risk_reduce"
    no_new_positions_before_earnings_days: int = 7
    reduce_position_before_earnings_days: int = 2
    exit_position_before_earnings_days: int = 1
    resume_after_earnings_days: int = 2
    earnings_risk_multiplier: float = 0.5
    block_high_event_risk: bool = True
    max_event_risk_score_for_normal_strategies: float = 70.0


@dataclass(frozen=True)
class OptionsOverlayConfig:
    """Translate equity signals into optional options exposure guidance."""

    enabled: bool = False
    chain_provider: str = "marketdata_or_csv"
    option_chain_path: str = "data/cache/option_chains.csv"
    option_chain_cache_dir: str = "data/cache"
    option_chain_dte: int | None = 45
    option_chain_side: str = "call"
    option_chain_strike_limit: int = 80
    min_option_open_interest: int = 100
    min_option_volume: int = 10
    max_option_spread_pct_mid: float = 0.20
    max_option_premium_pct_equity: float = 0.12
    max_single_option_premium_pct_equity: float = 0.025
    min_underlying_score_for_calls: float = 68.0
    min_relative_strength_for_calls: float = 75.0
    prefer_spreads_when_iv_risk_high: bool = True
    target_delta: float = 0.55
    target_dte_min: int = 30
    target_dte_max: int = 90
    high_risk_delta: float = 0.35
    high_risk_max_premium_pct_equity: float = 0.01
    theta_warning_dte: int = 21


@dataclass(frozen=True)
class DataQualityConfig:
    """Data quality and broker reconciliation settings."""

    broker_prices_path: str | None = None
    broker_close_diff_threshold: float = 0.01
    block_on_broker_price_diff: bool = False
    block_on_stale_price: bool = True
    block_on_abnormal_jump: bool = False
    missing_bar_warning_threshold: int = 0


@dataclass(frozen=True)
class LiveTradingConfig:
    """Live and paper-trading execution guardrails."""

    enabled: bool = False
    broker: str = "paper"
    account_id_env: str = "BROKER_ACCOUNT_ID"
    api_key_env: str = "BROKER_API_KEY"
    api_secret_env: str = "BROKER_API_SECRET"
    mode: str = "paper"
    require_manual_approval: bool = True
    allow_market_orders: bool = False
    default_order_type: str = "limit"
    tif: str = "day"
    limit_price_buffer_bps: float = 20.0
    min_order_notional: float = 250.0
    max_order_notional: float = 25_000.0
    max_single_symbol_weight: float = 0.10
    max_total_order_notional: float = 150_000.0
    max_daily_turnover_pct_equity: float = 0.35
    block_if_data_quality_flagged: bool = True
    block_if_event_risk_above: float = 70.0
    block_if_minute_pretrade_risk_above: float = 90.0
    block_if_overnight_gap_risk_above: float = 90.0
    reduce_only_when_risk_flagged: bool = True
    paper_cash: float = 100_000.0
    order_ticket_dir: str = "runs/live"


@dataclass(frozen=True)
class RegimeConfig:
    """Regime-aware validation settings."""

    enabled: bool = True
    exposure_overlay: bool = False
    risk_off_exposure_multiplier: float = 1.0
    partial_risk_on_exposure_multiplier: float = 0.75
    full_exposure_regime_score: float = 60.0
    partial_exposure_regime_score: float = 45.0
    analog_momentum_overlay: bool = False
    analog_inactive_exposure_multiplier: float = 1.0
    analog_active_exposure_multiplier: float = 1.0
    analog_lookback_days: int = 504
    analog_quantile: float = 0.80
    overbought_overlay: bool = False
    overbought_warning_score: float = 60.0
    overbought_reduce_score: float = 75.0
    overbought_cash_score: float = 90.0
    overbought_reduce_multiplier: float = 0.50
    overbought_cash_multiplier: float = 0.0
    intraday_risk_overlay: bool = False
    intraday_warning_score: float = 45.0
    intraday_reduce_score: float = 65.0
    intraday_cash_score: float = 90.0
    intraday_reduce_multiplier: float = 0.70
    intraday_cash_multiplier: float = 0.35
    intraday_benchmark_drop_warn: float = -0.025
    intraday_benchmark_drop_reduce: float = -0.040
    intraday_benchmark_gap_reduce: float = -0.030
    intraday_breadth_down_threshold: float = 0.60
    intraday_large_loss_threshold: float = -0.050
    intraday_volume_expansion_threshold: float = 1.50
    intraday_leader_exception_overlay: bool = False
    intraday_leader_exception_restore_multiplier: float = 0.85
    intraday_leader_exception_min_final_score: float = 70.0
    intraday_leader_exception_min_relative_strength_score: float = 88.0
    intraday_leader_exception_min_theme_score: float = 65.0
    intraday_leader_exception_min_mom_return: float = 0.10
    intraday_leader_exception_min_drawdown_from_high: float = 0.03
    intraday_leader_exception_max_drawdown_from_high: float = 0.14
    intraday_leader_exception_max_above_ma20_pct: float = 0.05
    intraday_leader_exception_min_adx_circuit_breaker: float = 18.0
    intraday_leader_exception_max_event_risk_score_circuit_breaker: float = 18.0
    intraday_leader_exception_max_overnight_gap_risk_score_circuit_breaker: float = 25.0
    intraday_leader_exception_require_theme_active: bool = False
    intraday_leader_exception_require_price_above_ma50: bool = True
    regime_leader_exception_overlay: bool = False
    regime_leader_exception_restore_multiplier: float = 0.75
    regime_leader_exception_min_regime_score: float = 50.0
    regime_leader_exception_min_final_score: float = 72.0
    regime_leader_exception_min_relative_strength_score: float = 90.0
    regime_leader_exception_min_theme_score: float = 62.0
    regime_leader_exception_min_mom_return: float = 0.12
    regime_leader_exception_min_drawdown_from_high: float = 0.03
    regime_leader_exception_max_drawdown_from_high: float = 0.14
    regime_leader_exception_max_above_ma20_pct: float = 0.05
    regime_leader_exception_min_adx_circuit_breaker: float = 18.0
    regime_leader_exception_max_event_risk_score_circuit_breaker: float = 18.0
    regime_leader_exception_max_overnight_gap_risk_score_circuit_breaker: float = 25.0
    regime_leader_exception_min_benchmark_ret_20d_circuit_breaker: float = -0.04
    regime_leader_exception_max_realized_vol_20d_circuit_breaker: float = 0.35
    regime_leader_exception_min_benchmark_drawdown_52w_circuit_breaker: float = -0.18
    regime_leader_exception_require_theme_active: bool = False
    regime_leader_exception_require_price_above_ma50: bool = True
    crowding_overlay: bool = False
    max_theme_gross_exposure: float = 0.35
    theme_gross_exposure_caps: dict[str, float] = field(default_factory=dict)
    max_sector_gross_exposure: float = 0.45
    sector_gross_exposure_caps: dict[str, float] = field(default_factory=dict)
    max_industry_gross_exposure: float = 0.30
    industry_gross_exposure_caps: dict[str, float] = field(default_factory=dict)
    crowding_reduce_multiplier_floor: float = 0.35
    leader_crowding_relief_overlay: bool = False
    leader_crowding_relief_min_relative_strength_score: float = 85.0
    leader_crowding_relief_min_theme_score: float = 60.0
    leader_crowding_relief_weight: float = 0.35
    leader_crowding_relief_max_names_per_bucket: int = 2
    leader_crowding_relief_max_weight_per_symbol: float = 0.015
    leader_crowding_relief_max_bucket_weight_restore: float = 0.03
    leader_crowding_relief_require_benchmark_risk_on: bool = True
    leader_crowding_relief_allow_non_risk_on_exception: bool = False
    leader_crowding_relief_min_final_score_exception: float = 70.0
    leader_crowding_relief_max_event_risk_score_circuit_breaker: float = 20.0
    leader_crowding_relief_max_overnight_gap_risk_score_circuit_breaker: float = 35.0
    leader_crowding_relief_min_base_multiplier_circuit_breaker: float = 0.75
    leader_crowding_relief_same_theme_peer_exception_overlay: bool = False
    leader_crowding_relief_same_theme_peer_min_count: int = 3
    leader_crowding_relief_same_theme_peer_min_active_share: float = 0.45
    leader_crowding_relief_same_theme_peer_min_avg_mom_return: float = 0.15
    leader_crowding_relief_same_theme_peer_restore_overlay: bool = False
    leader_crowding_relief_same_theme_peer_restore_min_count: int = 3
    leader_crowding_relief_same_theme_peer_restore_min_active_share: float = 0.50
    leader_crowding_relief_same_theme_peer_restore_min_avg_mom_return: float = 0.12
    leader_crowding_relief_controlled_pullback_overlay: bool = False
    leader_crowding_relief_controlled_pullback_min_final_score: float = 64.0
    leader_crowding_relief_controlled_pullback_min_distance_from_high: float = 0.04
    leader_crowding_relief_controlled_pullback_max_distance_from_high: float = 0.18
    leader_crowding_relief_controlled_pullback_max_above_ma20_pct: float = 0.06
    leader_crowding_relief_controlled_pullback_max_mom_return: float = 0.12
    overnight_gap_risk_overlay: bool = False
    overnight_gap_reduce_score: float = 65.0
    overnight_gap_cash_score: float = 90.0
    overnight_gap_reduce_multiplier: float = 0.65
    overnight_gap_cash_multiplier: float = 0.25
    overnight_gap_vol_warn: float = 0.035
    overnight_gap_vol_reduce: float = 0.055
    overnight_gap_tail_reduce: float = 0.10
    gap_beta_guard_overlay: bool = False
    gap_beta_benchmark_symbol: str = "SPY"
    gap_beta_min_gap_score: float = 80.0
    gap_beta_min_gap_vol_20d: float = 0.045
    gap_beta_min_realized_vol_63d: float = 0.50
    gap_beta_min_beta_63d: float = 1.75
    gap_beta_reduce_multiplier: float = 0.75
    gap_beta_min_multiplier: float = 0.50
    gap_beta_max_position_cut: float = 0.50
    gap_beta_guard_require_benchmark_risk_off: bool = False
    minute_pretrade_budget_overlay: bool = False
    minute_pretrade_reduce_score: float = 65.0
    minute_pretrade_cash_score: float = 90.0
    minute_pretrade_reduce_multiplier: float = 0.60
    minute_pretrade_cash_multiplier: float = 0.25
    allow_current_year_optimization: bool = False
    train_start: str = "2015-01-01"
    train_end: str = "2023-12-31"
    validation_start: str = "2024-01-01"
    validation_end: str = "2025-12-31"
    forward_start: str = "2026-01-01"
    risk_on_vol_threshold: float = 0.30
    theme_activation_threshold: float = 60.0


@dataclass(frozen=True)
class AppConfig:
    """Top-level application config."""

    start_date: str = "2018-01-01"
    end_date: str | None = None
    timeframe: str = "1d"
    data_provider: str = "csv"
    data_path: str = "data/sample/ohlcv.csv"
    intraday_data_path: str = "data/cache/marketdata_intraday.csv"
    fundamentals_path: str = "data/sample/fundamentals.csv"
    events_path: str = "data/sample/events.csv"
    output_dir: str = "runs"
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    fundamentals: FundamentalsConfig = field(default_factory=FundamentalsConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    options_overlay: OptionsOverlayConfig = field(default_factory=OptionsOverlayConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    live: LiveTradingConfig = field(default_factory=LiveTradingConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    strategies: dict[str, dict[str, Any]] = field(default_factory=dict)
    ensemble: dict[str, float] = field(default_factory=dict)


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping."""

    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return data


def _tuple_symbols(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value.upper(),)
    return tuple(str(item).upper() for item in value)


def _dataclass_from_dict(cls: type, data: dict[str, Any]):
    fields = getattr(cls, "__dataclass_fields__")
    filtered = {key: value for key, value in data.items() if key in fields}
    if cls is UniverseConfig:
        for key in ("symbols", "custom_symbols"):
            if key in filtered:
                filtered[key] = _tuple_symbols(filtered[key])
    return cls(**filtered)


def load_config(path: str | Path) -> AppConfig:
    """Load an AppConfig from YAML."""

    raw = load_yaml(path)
    universe_raw = dict(raw.get("universe", {}))
    if "symbols" in universe_raw and "custom_symbols" not in universe_raw:
        universe_raw["custom_symbols"] = universe_raw.get("symbols", [])
    return AppConfig(
        start_date=str(raw.get("start_date", "2018-01-01")),
        end_date=raw.get("end_date"),
        timeframe=str(raw.get("timeframe", "1d")),
        data_provider=str(raw.get("data_provider", raw.get("provider", "csv"))),
        data_path=expand_configured_path(raw.get("data_path", "data/sample/ohlcv.csv")),
        intraday_data_path=expand_configured_path(raw.get("intraday_data_path", "data/cache/marketdata_intraday.csv")),
        fundamentals_path=expand_configured_path(raw.get("fundamentals_path", "data/sample/fundamentals.csv")),
        events_path=expand_configured_path(raw.get("events_path", "data/sample/events.csv")),
        output_dir=expand_configured_path(raw.get("output_dir", "runs")),
        universe=_dataclass_from_dict(UniverseConfig, universe_raw),
        execution=_dataclass_from_dict(ExecutionConfig, dict(raw.get("execution", {}))),
        portfolio=_dataclass_from_dict(PortfolioConfig, dict(raw.get("portfolio", {}))),
        risk=_dataclass_from_dict(RiskConfig, dict(raw.get("risk", {}))),
        fundamentals=_dataclass_from_dict(FundamentalsConfig, dict(raw.get("fundamentals", {}))),
        events=_dataclass_from_dict(EventsConfig, dict(raw.get("events", {}))),
        options_overlay=_dataclass_from_dict(OptionsOverlayConfig, dict(raw.get("options_overlay", {}))),
        data_quality=_dataclass_from_dict(DataQualityConfig, dict(raw.get("data_quality", {}))),
        live=_dataclass_from_dict(LiveTradingConfig, dict(raw.get("live", {}))),
        regime=_dataclass_from_dict(RegimeConfig, dict(raw.get("regime", {}))),
        strategies=dict(raw.get("strategies", {})),
        ensemble=dict(raw.get("ensemble", {})),
    )


def dump_config(config: AppConfig, path: str | Path) -> None:
    """Persist the original dataclass config as a readable YAML-ish repr."""

    from dataclasses import asdict

    Path(path).write_text(yaml.safe_dump(asdict(config), sort_keys=False), encoding="utf-8")
