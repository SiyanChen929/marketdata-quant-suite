"""Constrained, covariance-aware allocation across pair-trading sleeves.

The objects being weighted here are complete pair strategies, not their individual
stock legs.  A weight of 10% therefore assigns 10% of portfolio capital to that
pair sleeve; the pair backtester remains responsible for the hedge ratio and the
long/short leg weights inside the sleeve.

All optimizers are deterministic and long-only.  Covariance is estimated with
Ledoit-Wolf shrinkage by default and repaired to be positive definite before use.
Sector and underlying-name constraints are linear incidence caps.  In particular,
a 20% name-incidence cap means that no more than 20% of sleeve capital may be
assigned to pairs containing that ticker; it is deliberately more conservative
than a cap based on the instantaneous, signed stock-leg exposure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, minimize
from sklearn.covariance import LedoitWolf


SUPPORTED_METHODS = (
    "inverse_volatility",
    "equal_risk_contribution",
    "minimum_variance",
    "maximum_diversification",
    "shrunk_tangency",
)

_METHOD_ALIASES = {
    "inverse_vol": "inverse_volatility",
    "risk_parity": "equal_risk_contribution",
    "erc": "equal_risk_contribution",
    "min_variance": "minimum_variance",
    "min_var": "minimum_variance",
    "minimum_covariance": "minimum_variance",
    "min_covariance": "minimum_variance",
    "max_diversification": "maximum_diversification",
    "max_div": "maximum_diversification",
    "tangency": "shrunk_tangency",
    "max_sharpe": "shrunk_tangency",
}


class AllocationError(ValueError):
    """Raised when allocation inputs or constraints cannot produce a portfolio."""


@dataclass(frozen=True)
class AllocationResult:
    """An optimized allocation and its complete audit information."""

    requested_method: str
    effective_method: str
    weights: pd.Series
    covariance: pd.DataFrame
    expected_returns: pd.Series | None
    weight_diagnostics: pd.DataFrame
    constraint_diagnostics: pd.DataFrame
    diagnostics: dict[str, Any]
    optimization_success: bool
    optimization_message: str


@dataclass(frozen=True)
class _LinearCap:
    kind: str
    label: str
    incidence: np.ndarray
    cap: float | None
    lower: float | None = None


def _canonical_method(method: str) -> str:
    value = str(method).strip().lower()
    value = _METHOD_ALIASES.get(value, value)
    if value not in SUPPORTED_METHODS:
        raise AllocationError(
            f"Unsupported allocation method {method!r}; choose one of "
            f"{', '.join(SUPPORTED_METHODS)}"
        )
    return value


def _clean_returns(
    returns: pd.DataFrame,
    *,
    minimum_observations: int,
) -> pd.DataFrame:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.empty or returns.shape[1] == 0:
        raise AllocationError("returns must contain at least one pair sleeve")
    if not returns.columns.is_unique:
        raise AllocationError("returns columns must be unique pair identifiers")
    if isinstance(minimum_observations, bool) or minimum_observations < 2:
        raise AllocationError("minimum_observations must be an integer of at least 2")
    numeric = returns.apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    all_missing = numeric.columns[numeric.notna().sum().eq(0)].tolist()
    if all_missing:
        raise AllocationError(
            "return histories are entirely missing for: " + ", ".join(map(str, all_missing))
        )
    sparse = numeric.columns[numeric.notna().sum().lt(minimum_observations)].tolist()
    if sparse:
        raise AllocationError(
            f"return histories need at least {minimum_observations} observations for: "
            + ", ".join(map(str, sparse))
        )
    # Missing values are filled with each sleeve's in-window mean.  This retains
    # dates without silently treating missing observations as zero P&L.
    complete = numeric.fillna(numeric.mean(axis=0))
    if len(complete) < minimum_observations:
        raise AllocationError(
            f"returns provide {len(complete)} observations; at least "
            f"{minimum_observations} are required"
        )
    return complete.astype(float)


def estimate_covariance(
    returns: pd.DataFrame,
    *,
    annualization: int = 252,
    estimator: str = "ledoit_wolf",
    minimum_observations: int = 3,
    eigenvalue_floor: float = 1e-10,
) -> pd.DataFrame:
    """Estimate an annualized, positive-definite sleeve covariance matrix.

    ``ledoit_wolf`` is the production default. ``sample`` is exposed for
    controlled comparisons and still receives a small eigenvalue floor.
    Estimator metadata is stored in ``DataFrame.attrs``.
    """
    clean = _clean_returns(returns, minimum_observations=minimum_observations)
    if isinstance(annualization, bool) or not isinstance(annualization, int) or annualization < 1:
        raise AllocationError("annualization must be a positive integer")
    if not math.isfinite(eigenvalue_floor) or eigenvalue_floor <= 0:
        raise AllocationError("eigenvalue_floor must be finite and positive")
    estimator_name = str(estimator).strip().lower()
    if estimator_name == "ledoit_wolf":
        fitted = LedoitWolf(assume_centered=False).fit(clean.to_numpy(dtype=float))
        daily = np.asarray(fitted.covariance_, dtype=float)
        shrinkage = float(fitted.shrinkage_)
    elif estimator_name == "sample":
        daily = clean.cov(ddof=1).to_numpy(dtype=float)
        shrinkage = 0.0
    else:
        raise AllocationError("covariance estimator must be 'ledoit_wolf' or 'sample'")
    daily = np.atleast_2d(daily)
    if daily.shape != (clean.shape[1], clean.shape[1]) or not np.isfinite(daily).all():
        raise AllocationError("covariance estimation produced an invalid matrix")
    daily = (daily + daily.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(daily)
    scale = max(float(np.trace(daily)) / len(eigenvalues), eigenvalue_floor)
    floor = max(eigenvalue_floor, scale * 1e-8)
    repaired = (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T
    repaired = (repaired + repaired.T) / 2.0 * annualization
    result = pd.DataFrame(repaired, index=clean.columns, columns=clean.columns)
    result.attrs.update(
        {
            "estimator": estimator_name,
            "shrinkage": shrinkage,
            "annualization": annualization,
            "observations": len(clean),
            "eigenvalue_floor_daily": floor,
        }
    )
    return result


def _aligned_vector(
    values: float | Mapping[str, float] | pd.Series,
    assets: pd.Index,
    *,
    label: str,
) -> pd.Series:
    if np.isscalar(values):
        try:
            scalar = float(values)
        except (TypeError, ValueError) as exc:
            raise AllocationError(f"{label} must be numeric") from exc
        result = pd.Series(scalar, index=assets, dtype=float, name=label)
    else:
        try:
            supplied = pd.Series(values, dtype=float)
        except (TypeError, ValueError) as exc:
            raise AllocationError(f"{label} must contain numeric values") from exc
        missing = assets.difference(supplied.index)
        extra = supplied.index.difference(assets)
        if len(missing) or len(extra):
            fragments = []
            if len(missing):
                fragments.append("missing " + ", ".join(map(str, missing)))
            if len(extra):
                fragments.append("unknown " + ", ".join(map(str, extra)))
            raise AllocationError(f"{label} labels do not match returns: {'; '.join(fragments)}")
        result = supplied.reindex(assets).rename(label)
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise AllocationError(f"{label} must be finite")
    return result


def _metadata_for_assets(
    pair_metadata: pd.DataFrame | None,
    assets: pd.Index,
    *,
    required_columns: set[str],
) -> pd.DataFrame:
    if pair_metadata is None:
        raise AllocationError(
            "pair_metadata is required for " + ", ".join(sorted(required_columns))
        )
    if not isinstance(pair_metadata, pd.DataFrame):
        raise TypeError("pair_metadata must be a pandas DataFrame")
    metadata = pair_metadata.copy()
    if "pair" in metadata.columns:
        if metadata["pair"].duplicated().any():
            raise AllocationError("pair_metadata contains duplicate pair identifiers")
        metadata = metadata.set_index("pair", drop=False)
    elif metadata.index.has_duplicates:
        raise AllocationError("pair_metadata index contains duplicate pair identifiers")
    missing_columns = required_columns - set(metadata.columns)
    if missing_columns:
        raise AllocationError(
            "pair_metadata missing columns: " + ", ".join(sorted(missing_columns))
        )
    missing_assets = assets.difference(metadata.index)
    if len(missing_assets):
        raise AllocationError(
            "pair_metadata missing pairs: " + ", ".join(map(str, missing_assets))
        )
    return metadata.reindex(assets)


def _validate_caps(caps: Mapping[str, float], *, label: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, raw_cap in caps.items():
        try:
            cap = float(raw_cap)
        except (TypeError, ValueError) as exc:
            raise AllocationError(f"{label} cap for {key!r} must be numeric") from exc
        if not math.isfinite(cap) or not 0 <= cap <= 1:
            raise AllocationError(f"{label} cap for {key!r} must be in [0, 1]")
        result[str(key)] = cap
    return result


def _coerce_exposure_matrix(
    exposures: pd.DataFrame | Mapping[str, Mapping[str, float]],
    assets: pd.Index,
    *,
    label: str,
) -> pd.DataFrame:
    """Return a pair-by-exposure matrix with exact pair label alignment."""
    if isinstance(exposures, pd.DataFrame):
        matrix = exposures.copy()
    else:
        try:
            matrix = pd.DataFrame(exposures)
        except (TypeError, ValueError) as exc:
            raise AllocationError(f"{label} must be convertible to a DataFrame") from exc
    if matrix.index.has_duplicates or matrix.columns.has_duplicates:
        raise AllocationError(f"{label} labels must be unique")
    # Accept either pair-by-exposure or exposure-by-pair orientation.  Pair by
    # exposure is preferred; factor models often arrive in the other orientation.
    if set(matrix.index) == set(assets) and len(matrix.index) == len(assets):
        matrix = matrix.reindex(assets)
    elif set(matrix.columns) == set(assets) and len(matrix.columns) == len(assets):
        matrix = matrix.reindex(columns=assets).T
    else:
        raise AllocationError(f"one axis of {label} must exactly match returns columns")
    matrix = matrix.apply(pd.to_numeric, errors="coerce")
    if matrix.isna().any().any() or not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise AllocationError(f"{label} must contain finite numeric values")
    matrix.columns = matrix.columns.astype(str)
    return matrix


def _parse_two_sided_bound(
    raw_bound: float | tuple[float, float],
    *,
    label: str,
    scalar_is_upper: bool = False,
) -> tuple[float, float]:
    """Validate a scalar band or explicit lower/upper exposure bound."""
    if np.isscalar(raw_bound):
        try:
            magnitude = float(raw_bound)
        except (TypeError, ValueError) as exc:
            raise AllocationError(f"{label} must be numeric") from exc
        if not math.isfinite(magnitude) or magnitude < 0:
            raise AllocationError(f"{label} must be finite and non-negative")
        return (0.0, magnitude) if scalar_is_upper else (-magnitude, magnitude)
    try:
        lower, upper = map(float, raw_bound)
    except (TypeError, ValueError) as exc:
        raise AllocationError(f"{label} must be a (lower, upper) pair") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise AllocationError(f"{label} requires finite lower <= upper")
    return lower, upper


def _append_vector_bound(
    caps: list[_LinearCap],
    assets: pd.Index,
    exposures: Mapping[str, float] | pd.Series | None,
    raw_bound: float | tuple[float, float] | None,
    *,
    kind: str,
    label: str,
    scalar_is_upper: bool = False,
    require_nonnegative_exposures: bool = False,
) -> None:
    if exposures is None and raw_bound is None:
        return
    if exposures is None:
        raise AllocationError(f"{label} exposures are required when its bounds are supplied")
    if raw_bound is None:
        raise AllocationError(f"{label} bounds are required when its exposures are supplied")
    vector = _aligned_vector(exposures, assets, label=f"{label}_exposures")
    if require_nonnegative_exposures and (vector < 0).any():
        raise AllocationError(f"{label} exposures must be non-negative")
    lower, upper = _parse_two_sided_bound(
        raw_bound,
        label=f"{label} bounds",
        scalar_is_upper=scalar_is_upper,
    )
    caps.append(
        _LinearCap(
            kind,
            label,
            vector.to_numpy(dtype=float),
            upper,
            lower,
        )
    )


def _append_matrix_bounds(
    caps: list[_LinearCap],
    assets: pd.Index,
    exposures: pd.DataFrame | Mapping[str, Mapping[str, float]] | None,
    bounds: Mapping[str, float | tuple[float, float]] | None,
    *,
    kind: str,
    label: str,
) -> None:
    if exposures is None and bounds is None:
        return
    if exposures is None:
        raise AllocationError(f"{label} are required when {label} bounds are supplied")
    if bounds is None:
        raise AllocationError(f"{label} bounds are required when {label} are supplied")
    matrix = _coerce_exposure_matrix(exposures, assets, label=label)
    supplied = {str(key): value for key, value in bounds.items()}
    unknown = sorted(set(supplied) - set(matrix.columns))
    if unknown:
        raise AllocationError(f"{label} bounds reference unknown labels: " + ", ".join(unknown))
    for exposure_label, raw_bound in sorted(supplied.items()):
        lower, upper = _parse_two_sided_bound(
            raw_bound,
            label=f"{label} bounds for {exposure_label!r}",
        )
        caps.append(
            _LinearCap(
                kind,
                exposure_label,
                matrix[exposure_label].to_numpy(dtype=float),
                upper,
                lower,
            )
        )


def _build_linear_caps(
    assets: pd.Index,
    pair_metadata: pd.DataFrame | None,
    *,
    sector_caps: Mapping[str, float] | None,
    name_incidence_caps: float | Mapping[str, float] | None,
    name_incidence: pd.DataFrame | None,
    group_caps: float | Mapping[str, float] | None,
    group_incidence: pd.DataFrame | None,
    linear_exposures: pd.DataFrame | None,
    linear_exposure_bounds: Mapping[str, float | tuple[float, float]] | None,
) -> list[_LinearCap]:
    caps: list[_LinearCap] = []
    if sector_caps:
        metadata = _metadata_for_assets(pair_metadata, assets, required_columns={"sector"})
        validated = _validate_caps(sector_caps, label="sector")
        available = set(metadata["sector"].astype(str))
        unknown = sorted(set(validated) - available)
        if unknown:
            raise AllocationError("sector caps reference unknown sectors: " + ", ".join(unknown))
        sectors = metadata["sector"].astype(str).to_numpy()
        for sector, cap in sorted(validated.items()):
            caps.append(
                _LinearCap("sector", sector, (sectors == sector).astype(float), cap)
            )

    if name_incidence_caps is not None and name_incidence is not None:
        if not isinstance(name_incidence, pd.DataFrame):
            raise TypeError("name_incidence must be a pandas DataFrame")
        if name_incidence.index.has_duplicates or name_incidence.columns.has_duplicates:
            raise AllocationError("name_incidence labels must be unique")
        missing = assets.difference(name_incidence.index)
        extra = name_incidence.index.difference(assets)
        if len(missing) or len(extra):
            raise AllocationError("name_incidence index must exactly match returns columns")
        incidence = name_incidence.reindex(assets).apply(pd.to_numeric, errors="coerce")
        if (
            incidence.isna().any().any()
            or (incidence < 0).any().any()
            or (incidence > 1).any().any()
        ):
            raise AllocationError("name_incidence values must be finite and in [0, 1]")
    elif name_incidence_caps is not None:
        metadata = _metadata_for_assets(
            pair_metadata, assets, required_columns={"dependent", "independent"}
        )
        names = sorted(
            set(metadata["dependent"].astype(str))
            | set(metadata["independent"].astype(str))
        )
        incidence = pd.DataFrame(0.0, index=assets, columns=names)
        for pair, row in metadata.iterrows():
            incidence.loc[pair, str(row["dependent"])] = 1.0
            incidence.loc[pair, str(row["independent"])] = 1.0

    if name_incidence_caps is not None:
        if np.isscalar(name_incidence_caps):
            raw_name_caps = {str(name): float(name_incidence_caps) for name in incidence.columns}
        else:
            raw_name_caps = dict(name_incidence_caps)
        validated_names = _validate_caps(raw_name_caps, label="underlying-name incidence")
        unknown_names = sorted(set(validated_names) - set(map(str, incidence.columns)))
        if unknown_names:
            raise AllocationError(
                "name caps reference unknown underlying names: " + ", ".join(unknown_names)
            )
        incidence.columns = incidence.columns.astype(str)
        for name, cap in sorted(validated_names.items()):
            caps.append(
                _LinearCap(
                    "underlying_name",
                    name,
                    incidence[name].to_numpy(dtype=float),
                    cap,
                )
            )

    if group_caps is not None:
        if group_incidence is None:
            raise AllocationError("group_incidence is required when group_caps are supplied")
        if not isinstance(group_incidence, pd.DataFrame):
            raise TypeError("group_incidence must be a pandas DataFrame")
        groups = _coerce_exposure_matrix(
            group_incidence,
            assets,
            label="group_incidence",
        )
        if (groups < 0).any().any() or (groups > 1).any().any():
            raise AllocationError("group_incidence values must be in [0, 1]")
        if np.isscalar(group_caps):
            raw_group_caps = {str(group): float(group_caps) for group in groups.columns}
        else:
            raw_group_caps = dict(group_caps)
        validated_groups = _validate_caps(raw_group_caps, label="group")
        unknown_groups = sorted(set(validated_groups) - set(groups.columns))
        if unknown_groups:
            raise AllocationError("group caps reference unknown groups: " + ", ".join(unknown_groups))
        for group, cap in sorted(validated_groups.items()):
            caps.append(
                _LinearCap(
                    "group",
                    group,
                    groups[group].to_numpy(dtype=float),
                    cap,
                )
            )

    if linear_exposure_bounds is not None:
        if linear_exposures is None:
            raise AllocationError(
                "linear_exposures is required when linear_exposure_bounds are supplied"
            )
        matrix = _coerce_exposure_matrix(
            linear_exposures,
            assets,
            label="linear_exposures",
        )
        supplied_bounds = {str(key): value for key, value in linear_exposure_bounds.items()}
        unknown = sorted(set(supplied_bounds) - set(matrix.columns))
        if unknown:
            raise AllocationError(
                "linear exposure bounds reference unknown exposures: " + ", ".join(unknown)
            )
        for label, raw_bounds in sorted(supplied_bounds.items()):
            lower_bound, upper_bound = _parse_two_sided_bound(
                raw_bounds,
                label=f"linear exposure bounds for {label!r}",
            )
            caps.append(
                _LinearCap(
                    "linear_exposure",
                    label,
                    matrix[label].to_numpy(dtype=float),
                    upper_bound,
                    lower_bound,
                )
            )
    return caps


def _constraints_for_slsqp(caps: Sequence[_LinearCap]) -> list[dict[str, Any]]:
    constraints: list[dict[str, Any]] = [
        {
            "type": "eq",
            "fun": lambda weights: float(np.sum(weights) - 1.0),
            "jac": lambda weights: np.ones_like(weights),
        }
    ]
    for item in caps:
        incidence = item.incidence.copy()
        if item.cap is not None:
            cap = item.cap
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, row=incidence, limit=cap: float(
                        limit - np.dot(row, weights)
                    ),
                    "jac": lambda weights, row=incidence: -row,
                }
            )
        if item.lower is not None:
            lower = item.lower
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda weights, row=incidence, limit=lower: float(
                        np.dot(row, weights) - limit
                    ),
                    "jac": lambda weights, row=incidence: row,
                }
            )
    return constraints


def _is_feasible(
    weights: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    caps: Sequence[_LinearCap],
    *,
    tolerance: float = 1e-7,
) -> bool:
    return bool(
        np.isfinite(weights).all()
        and abs(float(weights.sum()) - 1.0) <= tolerance
        and np.all(weights >= lower - tolerance)
        and np.all(weights <= upper + tolerance)
        and all(
            item.cap is None
            or float(np.dot(item.incidence, weights)) <= item.cap + tolerance
            for item in caps
        )
        and all(
            item.lower is None
            or float(np.dot(item.incidence, weights)) >= item.lower - tolerance
            for item in caps
        )
    )


def _feasible_start(
    target: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    caps: Sequence[_LinearCap],
) -> tuple[np.ndarray, OptimizeResult]:
    clipped = np.clip(np.asarray(target, dtype=float), lower, upper)
    if clipped.sum() <= 0:
        clipped = (lower + upper) / 2.0
    initial = clipped / clipped.sum()
    initial = np.clip(initial, lower, upper)
    result = minimize(
        lambda weights: float(np.sum(np.square(weights - target))),
        initial,
        jac=lambda weights: 2.0 * (weights - target),
        method="SLSQP",
        bounds=list(zip(lower, upper, strict=True)),
        constraints=_constraints_for_slsqp(caps),
        options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
    )
    candidate = np.asarray(result.x, dtype=float)
    if not result.success or not _is_feasible(candidate, lower, upper, caps):
        raise AllocationError(
            "allocation constraints are infeasible or could not be solved: "
            + str(result.message)
        )
    candidate[np.abs(candidate) < 1e-14] = 0.0
    candidate /= candidate.sum()
    return candidate, result


def _prepare_forecasts(
    returns: pd.DataFrame,
    expected_returns: Mapping[str, float] | pd.Series | None,
    assets: pd.Index,
    *,
    annualization: int,
    mean_shrinkage: float,
) -> tuple[pd.Series, pd.Series]:
    if not math.isfinite(mean_shrinkage) or not 0 <= mean_shrinkage <= 1:
        raise AllocationError("mean_shrinkage must be in [0, 1]")
    raw = (
        pd.Series(returns.mean(axis=0) * annualization, index=assets, dtype=float)
        if expected_returns is None
        else _aligned_vector(expected_returns, assets, label="expected_returns")
    )
    target = float(raw.mean())
    shrunk = ((1.0 - mean_shrinkage) * raw + mean_shrinkage * target).rename(
        "shrunk_expected_return"
    )
    return raw.rename("raw_expected_return"), shrunk


def _inverse_volatility_target(covariance: np.ndarray) -> np.ndarray:
    volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-18))
    inverse = 1.0 / volatility
    return inverse / inverse.sum()


def _objective(
    method: str,
    covariance: np.ndarray,
    expected_returns: np.ndarray,
    *,
    risk_free_rate: float,
    l2_penalty: float,
    risk_budgets: np.ndarray,
):
    asset_volatility = np.sqrt(np.maximum(np.diag(covariance), 1e-18))

    if method == "minimum_variance":
        return lambda weights: float(weights @ covariance @ weights)

    if method == "equal_risk_contribution":
        def erc(weights: np.ndarray) -> float:
            marginal = covariance @ weights
            variance = max(float(weights @ marginal), 1e-18)
            contribution = weights * marginal / variance
            return float(np.sum(np.square(contribution - risk_budgets)))

        return erc

    if method == "maximum_diversification":
        def negative_diversification(weights: np.ndarray) -> float:
            variance = max(float(weights @ covariance @ weights), 1e-18)
            weighted_volatility = float(weights @ asset_volatility)
            return -weighted_volatility / math.sqrt(variance)

        return negative_diversification

    if method == "shrunk_tangency":
        def negative_regularized_sharpe(weights: np.ndarray) -> float:
            variance = max(float(weights @ covariance @ weights), 1e-18)
            excess_return = float(weights @ expected_returns - risk_free_rate)
            return -excess_return / math.sqrt(variance) + l2_penalty * float(weights @ weights)

        return negative_regularized_sharpe

    raise AllocationError(f"No optimizer objective is defined for {method}")


def allocation_diagnostics(
    weights: pd.Series,
    covariance: pd.DataFrame,
    *,
    expected_returns: Mapping[str, float] | pd.Series | None = None,
    pair_betas: Mapping[str, float] | pd.Series | None = None,
    target_risk_budgets: Mapping[str, float] | pd.Series | None = None,
    tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Calculate concentration, risk contribution, return, and beta diagnostics."""
    if not isinstance(weights, pd.Series) or weights.empty:
        raise AllocationError("weights must be a non-empty pandas Series")
    assets = weights.index
    if covariance.index.has_duplicates or covariance.columns.has_duplicates:
        raise AllocationError("covariance labels must be unique")
    if set(covariance.index) != set(assets) or set(covariance.columns) != set(assets):
        raise AllocationError("covariance labels must match weights")
    cov = covariance.reindex(index=assets, columns=assets).to_numpy(dtype=float)
    values = pd.to_numeric(weights, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < -tolerance).any():
        raise AllocationError("weights must be finite and non-negative")
    if abs(float(values.sum()) - 1.0) > 1e-7:
        raise AllocationError("weights must sum to one")
    marginal = cov @ values
    variance = max(float(values @ marginal), 0.0)
    volatility = math.sqrt(variance)
    component_variance = values * marginal
    risk_fraction = component_variance / variance if variance > 0 else np.full(len(values), math.nan)
    detail = pd.DataFrame(
        {
            "pair": assets.astype(str),
            "weight": values,
            "marginal_variance": marginal,
            "component_variance": component_variance,
            "risk_contribution_fraction": risk_fraction,
        }
    ).set_index("pair", drop=False)
    hhi = float(values @ values)
    summary: dict[str, float | int] = {
        "pair_count": int(len(values)),
        "active_pair_count": int((values > tolerance).sum()),
        "minimum_weight": float(values.min()),
        "maximum_weight": float(values.max()),
        "top_three_weight": float(np.sort(values)[-min(3, len(values)):].sum()),
        "weight_hhi": hhi,
        "effective_number_of_pairs": float(1.0 / hhi),
        "expected_annual_volatility": volatility,
        "maximum_risk_contribution": float(np.nanmax(risk_fraction)),
        "minimum_risk_contribution": float(np.nanmin(risk_fraction)),
    }
    if expected_returns is not None:
        forecasts = _aligned_vector(expected_returns, assets, label="expected_returns")
        detail["expected_return"] = forecasts.to_numpy(dtype=float)
        detail["expected_return_contribution"] = values * forecasts.to_numpy(dtype=float)
        summary["expected_annual_return"] = float(values @ forecasts.to_numpy(dtype=float))
    if pair_betas is not None:
        betas = _aligned_vector(pair_betas, assets, label="pair_betas")
        detail["pair_beta"] = betas.to_numpy(dtype=float)
        detail["beta_contribution"] = values * betas.to_numpy(dtype=float)
        summary["expected_market_beta"] = float(values @ betas.to_numpy(dtype=float))
    if target_risk_budgets is not None:
        budgets = _aligned_vector(
            target_risk_budgets, assets, label="target_risk_budgets"
        )
        if (budgets < 0).any() or float(budgets.sum()) <= 0:
            raise AllocationError("target_risk_budgets must be non-negative with positive sum")
        budgets = budgets / budgets.sum()
        detail["target_risk_budget"] = budgets.to_numpy(dtype=float)
        detail["risk_budget_deviation"] = risk_fraction - budgets.to_numpy(dtype=float)
        summary["maximum_absolute_risk_budget_deviation"] = float(
            np.nanmax(np.abs(risk_fraction - budgets.to_numpy(dtype=float)))
        )
    return detail, summary


def _constraint_diagnostics(
    weights: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    caps: Sequence[_LinearCap],
    *,
    binding_tolerance: float = 1e-6,
) -> pd.DataFrame:
    values = weights.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(weights.index):
        rows.extend(
            [
                {
                    "constraint_type": "pair_lower_bound",
                    "label": str(pair),
                    "allocation": values[index],
                    "limit": float(lower.iloc[index]),
                    "slack": values[index] - float(lower.iloc[index]),
                    "binding": values[index] - float(lower.iloc[index]) <= binding_tolerance,
                },
                {
                    "constraint_type": "pair_upper_bound",
                    "label": str(pair),
                    "allocation": values[index],
                    "limit": float(upper.iloc[index]),
                    "slack": float(upper.iloc[index]) - values[index],
                    "binding": float(upper.iloc[index]) - values[index] <= binding_tolerance,
                },
            ]
        )
    for item in caps:
        allocation = float(item.incidence @ values)
        if item.cap is not None:
            slack = item.cap - allocation
            rows.append(
                {
                    "constraint_type": item.kind + "_upper_cap",
                    "label": item.label,
                    "allocation": allocation,
                    "limit": item.cap,
                    "slack": slack,
                    "binding": slack <= binding_tolerance,
                }
            )
        if item.lower is not None:
            lower_slack = allocation - item.lower
            rows.append(
                {
                    "constraint_type": item.kind + "_lower_cap",
                    "label": item.label,
                    "allocation": allocation,
                    "limit": item.lower,
                    "slack": lower_slack,
                    "binding": lower_slack <= binding_tolerance,
                }
            )
    return pd.DataFrame(rows)


def allocate_pair_sleeves(
    returns: pd.DataFrame,
    *,
    method: str = "equal_risk_contribution",
    expected_returns: Mapping[str, float] | pd.Series | None = None,
    pair_betas: Mapping[str, float] | pd.Series | None = None,
    target_risk_budgets: Mapping[str, float] | pd.Series | None = None,
    pair_metadata: pd.DataFrame | None = None,
    lower_bounds: float | Mapping[str, float] | pd.Series = 0.0,
    upper_bounds: float | Mapping[str, float] | pd.Series = 1.0,
    sector_caps: Mapping[str, float] | None = None,
    name_incidence_caps: float | Mapping[str, float] | None = None,
    name_incidence: pd.DataFrame | None = None,
    group_caps: float | Mapping[str, float] | None = None,
    group_incidence: pd.DataFrame | None = None,
    linear_exposures: pd.DataFrame | None = None,
    linear_exposure_bounds: Mapping[str, float | tuple[float, float]] | None = None,
    pair_gross_exposures: Mapping[str, float] | pd.Series | None = None,
    gross_exposure_bounds: float | tuple[float, float] | None = None,
    pair_net_exposures: Mapping[str, float] | pd.Series | None = None,
    net_exposure_bounds: float | tuple[float, float] | None = None,
    market_beta_bounds: float | tuple[float, float] | None = None,
    factor_betas: pd.DataFrame | Mapping[str, Mapping[str, float]] | None = None,
    factor_beta_bounds: Mapping[str, float | tuple[float, float]] | None = None,
    target_return: float | None = None,
    annualization: int = 252,
    covariance_estimator: str = "ledoit_wolf",
    minimum_observations: int = 3,
    mean_shrinkage: float = 0.50,
    risk_free_rate: float = 0.0,
    l2_penalty: float = 0.01,
    fallback_method: str | None = "inverse_volatility",
) -> AllocationResult:
    """Optimize long-only allocations across complete pair sleeves.

    The covariance and expected-return inputs must be estimated using information
    available before the allocation date.  To avoid selection leakage, call this
    function in a walk-forward schedule and lag each resulting weight vector by
    at least one trading session.

    ``target_return`` is an annual expected-return floor applied to the same
    shrunk forecasts used by the optimizer. Pair-level gross and net exposures,
    market/factor betas, shared-name incidence, arbitrary group incidence, and
    sector caps are enforced as explicit linear constraints. A scalar gross
    bound means ``[0, bound]``; scalar net and beta bounds mean symmetric bands.

    If the requested numerical objective fails, ``fallback_method`` is attempted
    and disclosed in the returned result. Infeasible constraints always raise
    :class:`AllocationError`; no cap is silently relaxed.
    """
    requested = _canonical_method(method)
    fallback = _canonical_method(fallback_method) if fallback_method is not None else None
    if fallback == requested:
        fallback = None
    if not math.isfinite(risk_free_rate):
        raise AllocationError("risk_free_rate must be finite")
    if not math.isfinite(l2_penalty) or l2_penalty < 0:
        raise AllocationError("l2_penalty must be finite and non-negative")
    if target_return is not None:
        try:
            target_return = float(target_return)
        except (TypeError, ValueError) as exc:
            raise AllocationError("target_return must be numeric") from exc
        if not math.isfinite(target_return):
            raise AllocationError("target_return must be finite")

    clean = _clean_returns(returns, minimum_observations=minimum_observations)
    assets = clean.columns
    lower = _aligned_vector(lower_bounds, assets, label="lower_bounds")
    upper = _aligned_vector(upper_bounds, assets, label="upper_bounds")
    if (lower < 0).any() or (upper > 1).any() or (lower > upper).any():
        raise AllocationError("pair bounds must satisfy 0 <= lower <= upper <= 1")
    if float(lower.sum()) > 1.0 + 1e-10 or float(upper.sum()) < 1.0 - 1e-10:
        raise AllocationError("pair bounds cannot sum to a fully invested portfolio")

    linear_caps = _build_linear_caps(
        assets,
        pair_metadata,
        sector_caps=sector_caps,
        name_incidence_caps=name_incidence_caps,
        name_incidence=name_incidence,
        group_caps=group_caps,
        group_incidence=group_incidence,
        linear_exposures=linear_exposures,
        linear_exposure_bounds=linear_exposure_bounds,
    )
    if group_incidence is not None and group_caps is None:
        raise AllocationError("group_caps are required when group_incidence is supplied")
    _append_vector_bound(
        linear_caps,
        assets,
        pair_gross_exposures,
        gross_exposure_bounds,
        kind="portfolio_gross_exposure",
        label="gross",
        scalar_is_upper=True,
        require_nonnegative_exposures=True,
    )
    _append_vector_bound(
        linear_caps,
        assets,
        pair_net_exposures,
        net_exposure_bounds,
        kind="portfolio_net_exposure",
        label="net",
    )
    if market_beta_bounds is not None:
        _append_vector_bound(
            linear_caps,
            assets,
            pair_betas,
            market_beta_bounds,
            kind="market_beta",
            label="market",
        )
    _append_matrix_bounds(
        linear_caps,
        assets,
        factor_betas,
        factor_beta_bounds,
        kind="factor_beta",
        label="factor_betas",
    )
    lower_values = lower.to_numpy(dtype=float)
    upper_values = upper.to_numpy(dtype=float)

    covariance = estimate_covariance(
        clean,
        annualization=annualization,
        estimator=covariance_estimator,
        minimum_observations=minimum_observations,
    )
    covariance_values = covariance.to_numpy(dtype=float)
    raw_forecasts, shrunk_forecasts = _prepare_forecasts(
        clean,
        expected_returns,
        assets,
        annualization=annualization,
        mean_shrinkage=mean_shrinkage,
    )
    if target_return is not None:
        linear_caps.append(
            _LinearCap(
                "expected_return",
                "annual_expected_return",
                shrunk_forecasts.to_numpy(dtype=float),
                None,
                target_return,
            )
        )
    for item in linear_caps:
        minimum_allocation = float(item.incidence @ lower_values)
        # This shortcut is valid only for non-negative incidence. Signed factor,
        # beta, and net-exposure rows can be reduced by allocating residual weight
        # to a sleeve with a negative loading, so their feasibility is left to the
        # complete constrained solve below.
        if (
            item.cap is not None
            and np.all(item.incidence >= 0)
            and minimum_allocation > item.cap + 1e-10
        ):
            raise AllocationError(
                f"{item.kind} cap for {item.label!r} is below mandatory lower-bound "
                f"allocation ({item.cap:.6g} < {minimum_allocation:.6g})"
            )
    inverse_target = _inverse_volatility_target(covariance_values)
    if target_risk_budgets is None:
        risk_budgets = pd.Series(1.0 / len(assets), index=assets, dtype=float)
    else:
        risk_budgets = _aligned_vector(
            target_risk_budgets, assets, label="target_risk_budgets"
        )
        if (risk_budgets < 0).any() or float(risk_budgets.sum()) <= 0:
            raise AllocationError(
                "target_risk_budgets must be non-negative with positive sum"
            )
        risk_budgets = risk_budgets / risk_budgets.sum()
    feasible, feasible_result = _feasible_start(
        inverse_target, lower_values, upper_values, linear_caps
    )

    def solve(method_name: str) -> OptimizeResult:
        if method_name == "inverse_volatility":
            return minimize(
                lambda weights: float(np.sum(np.square(weights - inverse_target))),
                feasible,
                jac=lambda weights: 2.0 * (weights - inverse_target),
                method="SLSQP",
                bounds=list(zip(lower_values, upper_values, strict=True)),
                constraints=_constraints_for_slsqp(linear_caps),
                options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
            )
        return minimize(
            _objective(
                method_name,
                covariance_values,
                shrunk_forecasts.to_numpy(dtype=float),
                risk_free_rate=risk_free_rate,
                l2_penalty=l2_penalty,
                risk_budgets=risk_budgets.to_numpy(dtype=float),
            ),
            feasible,
            method="SLSQP",
            bounds=list(zip(lower_values, upper_values, strict=True)),
            constraints=_constraints_for_slsqp(linear_caps),
            options={"ftol": 1e-12, "maxiter": 2_000, "disp": False},
        )

    result = solve(requested)
    candidate = np.asarray(result.x, dtype=float)
    effective = requested
    fallback_used = False
    requested_message = str(result.message)
    if not result.success or not _is_feasible(
        candidate, lower_values, upper_values, linear_caps
    ):
        if fallback is None:
            raise AllocationError(
                f"{requested} optimization failed and no fallback was allowed: {result.message}"
            )
        fallback_result = solve(fallback)
        fallback_candidate = np.asarray(fallback_result.x, dtype=float)
        if not fallback_result.success or not _is_feasible(
            fallback_candidate, lower_values, upper_values, linear_caps
        ):
            raise AllocationError(
                f"{requested} optimization failed ({result.message}); fallback {fallback} "
                f"also failed ({fallback_result.message})"
            )
        result = fallback_result
        candidate = fallback_candidate
        effective = fallback
        fallback_used = True

    candidate[np.abs(candidate) < 1e-14] = 0.0
    candidate /= candidate.sum()
    if not _is_feasible(candidate, lower_values, upper_values, linear_caps):
        raise AllocationError("post-solve normalization violated an allocation constraint")
    weights = pd.Series(candidate, index=assets, name="weight")
    detail, summary = allocation_diagnostics(
        weights,
        covariance,
        expected_returns=shrunk_forecasts,
        pair_betas=pair_betas,
        target_risk_budgets=risk_budgets,
    )
    constraint_detail = _constraint_diagnostics(
        weights, lower, upper, linear_caps
    )
    exposure_diagnostics: dict[str, float] = {}
    for item in linear_caps:
        realized = float(item.incidence @ candidate)
        exposure_diagnostics[f"{item.kind}:{item.label}"] = realized
        if item.kind == "portfolio_gross_exposure":
            summary["portfolio_gross_exposure"] = realized
        elif item.kind == "portfolio_net_exposure":
            summary["portfolio_net_exposure"] = realized
        elif item.kind == "expected_return" and item.lower is not None:
            summary["target_annual_return"] = float(item.lower)
            summary["target_return_slack"] = realized - float(item.lower)
    factor_exposure_diagnostics = {
        item.label: float(item.incidence @ candidate)
        for item in linear_caps
        if item.kind == "factor_beta"
    }
    summary.update(
        {
            "requested_method": requested,
            "effective_method": effective,
            "fallback_used": fallback_used,
            "covariance_estimator": covariance.attrs.get("estimator"),
            "covariance_shrinkage": covariance.attrs.get("shrinkage"),
            "covariance_observations": covariance.attrs.get("observations"),
            "raw_forecast_cross_sectional_mean": float(raw_forecasts.mean()),
            "mean_shrinkage": mean_shrinkage,
            "binding_constraint_count": int(constraint_detail["binding"].sum()),
            "linear_constraint_exposures": exposure_diagnostics,
            "factor_beta_exposures": factor_exposure_diagnostics,
        }
    )
    message = (
        f"requested {requested} failed ({requested_message}); used {effective}: {result.message}"
        if fallback_used
        else str(result.message)
    )
    return AllocationResult(
        requested_method=requested,
        effective_method=effective,
        weights=weights,
        covariance=covariance,
        expected_returns=shrunk_forecasts,
        weight_diagnostics=detail,
        constraint_diagnostics=constraint_detail,
        diagnostics=summary,
        optimization_success=bool(result.success),
        optimization_message=message,
    )


def build_allocation_candidates(
    returns: pd.DataFrame,
    *,
    methods: Sequence[str] = SUPPORTED_METHODS,
    **kwargs: Any,
) -> dict[str, AllocationResult]:
    """Build a deterministic method-comparison set with identical inputs/caps."""
    if not methods:
        raise AllocationError("methods cannot be empty")
    candidates: dict[str, AllocationResult] = {}
    for method in methods:
        canonical = _canonical_method(method)
        if canonical in candidates:
            raise AllocationError(f"duplicate allocation method: {canonical}")
        candidates[canonical] = allocate_pair_sleeves(
            returns,
            method=canonical,
            **kwargs,
        )
    return candidates
