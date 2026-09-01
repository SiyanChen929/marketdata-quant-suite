import pandas as pd

from quant_system.backtest.reports import compute_overlay_activation_summary


def test_overlay_activation_summary_counts_active_and_blocked_rows() -> None:
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-05-14", "2026-05-14", "2026-05-15"]),
            "symbol": ["AAA", "BBB", "AAA"],
            "signal": [1.0, 0.0, 1.0],
            "final_score": [70.0, 55.0, 72.0],
            "gap_adjusted_continuation_score": [2.0, 0.0, 0.0],
            "gap_adjusted_continuation_boost": [1.0, 0.0, 0.0],
            "gap_adjusted_continuation_eligible": [True, False, False],
            "gap_adjusted_continuation_blocked": [False, True, False],
            "theme_breadth_acceleration_score": [0.0, 1.5, 0.0],
            "theme_breadth_acceleration_boost": [0.0, 0.5, 0.0],
            "theme_breadth_acceleration_eligible": [False, True, False],
            "boundary_rank_promotion_score": [0.0, 3.0, 0.0],
            "boundary_rank_promotion_promoted": [False, True, False],
            "boundary_rank_promotion_blocked": [False, False, False],
        }
    )

    out = compute_overlay_activation_summary(signals)
    gap = out[out["overlay"].eq("gap_adjusted_continuation")].iloc[0]
    breadth = out[out["overlay"].eq("theme_breadth_acceleration")].iloc[0]
    boundary = out[out["overlay"].eq("boundary_rank_promotion")].iloc[0]

    assert gap["active_rows"] == 1
    assert gap["blocked_rows"] == 1
    assert gap["active_signal_rows"] == 1
    assert gap["unique_active_symbols"] == 1
    assert gap["active_dates"] == 1
    assert gap["avg_boost_active"] == 1.0
    assert breadth["active_rows"] == 1
    assert breadth["active_signal_rows"] == 0
    assert boundary["active_rows"] == 1
    assert boundary["active_signal_rows"] == 0


def test_overlay_activation_summary_empty_schema() -> None:
    out = compute_overlay_activation_summary(pd.DataFrame())

    assert "overlay" in out.columns
    assert out.empty
