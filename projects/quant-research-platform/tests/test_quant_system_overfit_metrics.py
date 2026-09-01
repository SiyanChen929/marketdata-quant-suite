from __future__ import annotations

import numpy as np
import pandas as pd

from quant_system.optimization.robustness import deflated_sharpe_ratio, estimate_pbo


def test_estimate_pbo_flags_unstable_parameter_selection() -> None:
    details = pd.DataFrame(
        [
            {"scenario": "lucky", "split_id": 1, "validation_objective": 3.0},
            {"scenario": "lucky", "split_id": 2, "validation_objective": -1.0},
            {"scenario": "stable", "split_id": 1, "validation_objective": 1.0},
            {"scenario": "stable", "split_id": 2, "validation_objective": 1.0},
        ]
    )

    result = estimate_pbo(details)

    assert result["status"] == "ok"
    assert result["combinations"] == 2
    assert result["pbo"] > 0.0


def test_deflated_sharpe_ratio_penalizes_more_trials() -> None:
    rng = np.random.default_rng(17)
    returns = pd.Series(rng.normal(0.0005, 0.01, 252))

    one_trial = deflated_sharpe_ratio(returns, trials=1)
    many_trials = deflated_sharpe_ratio(returns, trials=200)

    assert one_trial["status"] == "ok"
    assert many_trials["status"] == "ok"
    assert many_trials["benchmark_sharpe"] > one_trial["benchmark_sharpe"]
    assert many_trials["z_score"] < one_trial["z_score"]
