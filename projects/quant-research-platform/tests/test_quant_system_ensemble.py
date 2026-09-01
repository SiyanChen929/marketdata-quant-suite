import warnings

import pandas as pd
from pandas.errors import PerformanceWarning

from quant_system.strategies.ensemble import combine_strategy_signals


def test_ensemble_preserves_theme_overlay_diagnostics() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-15"]),
            "symbol": ["AAA"],
            "signal": [1.0],
            "final_score": [75.0],
            "technical_score": [80.0],
            "relative_strength_score": [85.0],
            "theme_score": [70.0],
            "fundamental_score": [55.0],
            "event_risk_score": [5.0],
            "reason_for_entry": ["test"],
            "score_decomposition": ["theme_breadth_accel=1.25"],
            "theme_strength_delta_score": [1.75],
            "theme_strength_delta_eligible": [True],
            "theme_breadth_acceleration_score": [1.25],
            "theme_breadth_acceleration_eligible": [True],
            "theme_leader_tilt_score": [0.5],
            "theme_leader_tilt_eligible": [False],
            "pullback_short_term_reset_score": [42.0],
            "pullback_short_term_reset_ret_5d": [-0.03],
            "pullback_short_term_reset_blocked": [False],
            "short_term_volume_tilt_score": [33.0],
            "short_term_volume_tilt_ret_5d": [-0.04],
            "short_term_volume_tilt_volume_expansion": [1.25],
            "short_term_volume_tilt_blocked": [False],
        }
    )

    out = combine_strategy_signals({"momentum": frame}, {"momentum": 1.0})

    assert out.loc[0, "theme_strength_delta_score"] == 1.75
    assert bool(out.loc[0, "theme_strength_delta_eligible"])
    assert out.loc[0, "theme_breadth_acceleration_score"] == 1.25
    assert bool(out.loc[0, "theme_breadth_acceleration_eligible"])
    assert out.loc[0, "theme_leader_tilt_score"] == 0.5
    assert out.loc[0, "pullback_short_term_reset_score"] == 42.0
    assert out.loc[0, "pullback_short_term_reset_ret_5d"] == -0.03
    assert not bool(out.loc[0, "pullback_short_term_reset_blocked"])
    assert out.loc[0, "short_term_volume_tilt_score"] == 33.0
    assert out.loc[0, "short_term_volume_tilt_ret_5d"] == -0.04
    assert out.loc[0, "short_term_volume_tilt_volume_expansion"] == 1.25
    assert not bool(out.loc[0, "short_term_volume_tilt_blocked"])


def test_ensemble_combines_optional_overlay_columns_without_fragmentation_warning() -> None:
    rows = []
    for idx in range(10):
        rows.append(
            {
                "date": pd.Timestamp("2026-05-15"),
                "symbol": f"AAA{idx}",
                "signal": 1.0,
                "final_score": 60.0 + idx,
                "gap_adjusted_continuation_score": 2.0,
                "gap_adjusted_continuation_boost": 1.0,
                "gap_adjusted_continuation_eligible": True,
                "gap_adjusted_continuation_blocked": False,
            }
        )
    frame = pd.DataFrame(rows)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = combine_strategy_signals({"momentum": frame, "trend": frame.copy()}, {"momentum": 0.7, "trend": 0.3})

    assert not [warning for warning in caught if issubclass(warning.category, PerformanceWarning)]
    assert out["gap_adjusted_continuation_score"].eq(2.0).all()
    assert out["gap_adjusted_continuation_eligible"].all()
