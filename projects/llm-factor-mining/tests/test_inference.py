from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pytest

from llm_factor_mining.inference import (
    EULER_MASCHERONI,
    benjamini_hochberg,
    benjamini_yekutieli,
    deflated_sharpe_ratio,
    expected_maximum_sharpe,
    holm,
    multiple_test,
    probabilistic_sharpe_ratio,
    sample_moments,
    t_to_p,
)


P4 = [0.01, 0.04, 0.03, 0.005]


def test_bh_hand_example_all_rejected() -> None:
    # sorted: .005 .01 .03 .04 vs thresholds .0125 .025 .0375 .05 -> all pass
    result = benjamini_hochberg(P4, 0.05)
    assert result.rejected.tolist() == [True, True, True, True]
    # adjusted p_(i) = min_{j>=i} m p_(j) / j  ->  .02 .02 .04 .04 (sorted order)
    np.testing.assert_allclose(result.adjusted, [0.02, 0.04, 0.04, 0.02])
    assert result.n_rejected == 4 and result.m == 4


def test_bh_step_up_and_family_padding() -> None:
    p = [0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216]
    # m = 10: thresholds 0.005 * i; only the first two pass
    ten = benjamini_hochberg(p, 0.05)
    assert ten.rejected.tolist() == [True, True] + [False] * 8
    # m = 20 (ten more untested hypotheses with p = 1): thresholds 0.0025 * i
    twenty = benjamini_hochberg(p, 0.05, m=20)
    assert twenty.rejected.tolist() == [True] + [False] * 9
    assert twenty.adjusted[0] == pytest.approx(20 * 0.001 / 1)
    assert twenty.adjusted[1] == pytest.approx(min(20 * 0.008 / 2, 20 * 0.039 / 3, 20 * 0.041 / 4, 20 * 0.042 / 5))
    with pytest.raises(ValueError):
        benjamini_hochberg(p, 0.05, m=5)


def test_by_hand_example() -> None:
    # c(4) = 1 + 1/2 + 1/3 + 1/4; thresholds alpha i / (m c) = 0.006 i
    c4 = 1 + 1 / 2 + 1 / 3 + 1 / 4
    result = benjamini_yekutieli(P4, 0.05)
    assert result.rejected.tolist() == [True, False, False, True]
    expected_sorted = [4 * c4 * 0.005, 4 * c4 * 0.01 / 2, 4 * c4 * 0.03 / 3, 4 * c4 * 0.04 / 4]
    expected_sorted = np.minimum.accumulate(np.asarray(expected_sorted)[::-1])[::-1]
    # original order: 0.01 -> rank 2, 0.04 -> 4, 0.03 -> 3, 0.005 -> 1
    np.testing.assert_allclose(
        result.adjusted, [expected_sorted[1], expected_sorted[3], expected_sorted[2], expected_sorted[0]]
    )
    assert result.adjusted[0] == pytest.approx(0.0416666667, rel=1e-9)


def test_holm_hand_example() -> None:
    # .005 <= .05/4, .01 <= .05/3, .03 > .05/2 -> stop
    result = holm(P4, 0.05)
    assert result.rejected.tolist() == [True, False, False, True]
    # adjusted: 4*.005=.02, max(.02, 3*.01)=.03, max(.03, 2*.03)=.06, max(.06, .04)=.06
    np.testing.assert_allclose(result.adjusted, [0.03, 0.06, 0.06, 0.02])


def test_nan_pvalues_count_as_one_and_dispatch() -> None:
    result = multiple_test([0.001, float("nan")], 0.05, method="bh")
    assert result.rejected.tolist() == [True, False]
    assert result.adjusted[1] == 1.0
    assert multiple_test(P4, 0.05, method="by").n_rejected == 2
    assert multiple_test(P4, 0.05, method="holm").n_rejected == 2
    with pytest.raises(ValueError):
        multiple_test(P4, 0.05, method="bonferroni")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        benjamini_hochberg([1.5], 0.05)
    assert benjamini_hochberg([], 0.05, m=3).n_rejected == 0


def test_t_to_p() -> None:
    z = NormalDist().inv_cdf(0.975)
    assert t_to_p(z) == pytest.approx(0.05, abs=1e-12)
    assert t_to_p(z, alternative="greater") == pytest.approx(0.025, abs=1e-12)
    assert t_to_p(-z, alternative="less") == pytest.approx(0.025, abs=1e-12)
    assert t_to_p(2.228138851986273, df=10) == pytest.approx(0.05, abs=1e-9)
    assert math.isnan(t_to_p(float("nan")))


def test_probabilistic_sharpe_hand_computed() -> None:
    # SR = 0.1, T = 101, normal returns: z = 0.1 * 10 / sqrt(1 + (3 - 1)/4 * 0.01)
    z = 0.1 * 10 / math.sqrt(1 + 0.5 * 0.01)
    expected = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    assert probabilistic_sharpe_ratio(0.1, 101) == pytest.approx(expected, abs=1e-12)
    # negative skew and fat tails widen the denominator and lower the PSR
    z2 = 0.1 * 10 / math.sqrt(1 - (-1.0) * 0.1 + (6 - 1) / 4 * 0.01)
    expected2 = 0.5 * (1 + math.erf(z2 / math.sqrt(2)))
    assert probabilistic_sharpe_ratio(0.1, 101, skew=-1.0, kurtosis=6.0) == pytest.approx(expected2, abs=1e-12)
    assert expected2 < expected
    assert math.isnan(probabilistic_sharpe_ratio(0.1, 1))


def test_expected_maximum_and_deflated_sharpe_hand_computed() -> None:
    normal = NormalDist()
    n, variance = 10, 0.25
    expected_max = math.sqrt(variance) * (
        (1 - EULER_MASCHERONI) * normal.inv_cdf(1 - 1 / n)
        + EULER_MASCHERONI * normal.inv_cdf(1 - 1 / (n * math.e))
    )
    assert expected_maximum_sharpe(n, variance) == pytest.approx(expected_max, rel=1e-10)
    assert expected_maximum_sharpe(1, variance) == 0.0

    sharpe, t_obs = 0.9, 250
    z = (sharpe - expected_max) * math.sqrt(t_obs - 1) / math.sqrt(1 + 0.5 * sharpe**2)
    dsr = deflated_sharpe_ratio(sharpe, t_obs, n_trials=n, sharpe_variance=variance)
    assert dsr == pytest.approx(normal.cdf(z), abs=1e-10)
    # more trials -> higher bar -> lower DSR; one trial -> PSR against zero
    assert deflated_sharpe_ratio(sharpe, t_obs, n_trials=100, sharpe_variance=variance) < dsr
    assert deflated_sharpe_ratio(sharpe, t_obs, n_trials=1, sharpe_variance=variance) == pytest.approx(
        probabilistic_sharpe_ratio(sharpe, t_obs)
    )
    with pytest.raises(ValueError):
        expected_maximum_sharpe(0, variance)


def test_sample_moments() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(0.1, 1.0, 200_000)
    sharpe, skew, kurt, n = sample_moments(x)
    assert n == 200_000
    assert sharpe == pytest.approx(0.1, abs=0.01)
    assert skew == pytest.approx(0.0, abs=0.03)
    assert kurt == pytest.approx(3.0, abs=0.05)
    assert math.isnan(sample_moments([1.0, 2.0])[0])


def test_expected_maximum_of_absolute_sharpes_needs_twice_the_trials() -> None:
    # |ICIR| selection is two-sided: E[max_i |Z_i|] over N trials ~ E[max of 2N signed Z]
    rng = np.random.default_rng(4)
    n_trials, reps = 200, 4000
    draws = np.abs(rng.standard_normal((reps, n_trials))).max(axis=1).mean()
    assert expected_maximum_sharpe(2 * n_trials, 1.0) == pytest.approx(draws, rel=0.02)
    assert expected_maximum_sharpe(n_trials, 1.0) < draws * 0.95
