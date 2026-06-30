"""Replicate-derived noise floor for the Seed/VFA dataset (AGENTS.md §5).

Purpose
-------
Estimate the irreducible RMSE from within-replicate variability. This is the
headline result of Exp-A and the reference line on every performance figure.

Definition (AGENTS.md §5)
--------------------------
For each replicate group g with n_g >= 2 observations of the target y:

    s²_g  = var(y_g, ddof=1)          # within-group sample variance
    w_g   = n_g - 1                   # pooling weight
    RMSE_floor = sqrt( Σ w_g·s²_g / Σ w_g )

This is the pooled within-group standard deviation — the minimum RMSE any
model could achieve if it predicted the group mean exactly.

Confidence interval
-------------------
Bootstrap over *groups* (not rows): draw B samples of the same number of
groups with replacement, compute the floor for each, report the α/2 and
1−α/2 percentiles. Resampling groups preserves the group structure and
reflects between-group variability in within-replicate variance.

Leakage flag
------------
``flag_below_floor(rmse, rmse_floor)`` returns True iff ``rmse < rmse_floor``,
i.e. the reported performance is better than the noise limit. The check is
strict (no tolerance) because the floor itself already carries uncertainty
through the CI.

Inputs / outputs
----------------
:func:`compute_floor` consumes the grouped DataFrame (``seed_with_groups.csv``)
and the name of the target and replicate-group columns. It returns a
:class:`NoiseFloorResult` with all information needed for the CSV and sidebar.

Invariants
----------
* The formula uses ``ddof=1`` for unbiased variance estimates.
* Weighting by ``n_g - 1`` makes the pooled estimate exactly equivalent to a
  one-way ANOVA within-group SS / df, correct for balanced and unbalanced
  designs alike.
* The bootstrap seed must come from ``config.yaml`` (AGENTS.md §1.5).
* Hardcoding the expected floor value (≈0.23) is forbidden; it is verified,
  not asserted (AGENTS.md §5 "verify, do not hardcode").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class GroupVarianceStats:
    """Within-group variance statistics for a single replicate group.

    Attributes
    ----------
    group_id:
        The replicate_group_id label.
    n_obs:
        Number of observations in this group (n_g).
    weight:
        Pooling weight = n_g − 1.
    variance:
        Within-group sample variance (ddof=1).
    """

    group_id: str
    n_obs: int
    weight: float
    variance: float


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile bootstrap confidence interval for the noise floor.

    Attributes
    ----------
    lower:
        Lower bound (α/2 quantile of bootstrap distribution).
    upper:
        Upper bound (1 − α/2 quantile).
    ci_level:
        Nominal confidence level (e.g. 0.95).
    n_bootstrap:
        Number of bootstrap resamples used.
    """

    lower: float
    upper: float
    ci_level: float
    n_bootstrap: int


@dataclass(frozen=True)
class NoiseFloorResult:
    """Complete noise-floor estimate with uncertainty (AGENTS.md §5).

    Attributes
    ----------
    rmse_floor:
        Point estimate of the replicate-derived noise floor (in the same units
        as the target, i.e. g/L VFA).
    n_groups_used:
        Number of replicate groups with n_g >= 2 that contributed to the
        estimate.
    n_obs_used:
        Total observations across all contributing groups.
    ci:
        Bootstrap confidence interval over groups.
    group_stats:
        Per-group variance statistics, ordered by group_id. Preserved for
        auditability (rules §6 scientific reliability).
    """

    rmse_floor: float
    n_groups_used: int
    n_obs_used: int
    ci: BootstrapCI
    group_stats: tuple[GroupVarianceStats, ...]


def _pool_variances(variances: np.ndarray, weights: np.ndarray) -> float:
    """Compute the pooled within-group RMSE from per-group variances and weights.

    Parameters
    ----------
    variances:
        1-D array of per-group within-group sample variances (ddof=1).
    weights:
        1-D array of pooling weights (n_g − 1 for each group).

    Returns
    -------
    float
        sqrt(weighted mean of variances) = pooled within-group std deviation.

    Raises
    ------
    ValueError
        If there are no groups or total weight is zero.
    """
    if len(variances) == 0 or weights.sum() == 0:
        raise ValueError(
            "Cannot compute pooled RMSE: no groups with n_g >= 2 or all weights zero."
        )
    return float(np.sqrt(np.dot(weights, variances) / weights.sum()))


def compute_floor(
    df: pd.DataFrame,
    *,
    target_col: str,
    group_col: str,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> NoiseFloorResult:
    """Compute the replicate-derived noise floor and its bootstrap CI.

    Parameters
    ----------
    df:
        Grouped dataset (e.g. ``seed_with_groups.csv``). Must contain at least
        ``target_col`` and ``group_col``.
    target_col:
        Name of the regression target column (``"Cret total VFAs"``).
    group_col:
        Name of the replicate-group column (``"replicate_group_id"``).
    n_bootstrap:
        Number of bootstrap resamples over groups. Should be >= 5000 for
        stable 95% CI bounds.
    ci_level:
        Two-sided confidence level (e.g. 0.95 → 2.5th and 97.5th percentiles).
    seed:
        RNG seed for reproducibility (from ``config.yaml``).

    Returns
    -------
    NoiseFloorResult

    Raises
    ------
    KeyError
        If ``target_col`` or ``group_col`` are not present in ``df``.
    ValueError
        If no group has n_g >= 2 (cannot estimate within-group variance).
    """
    if target_col not in df.columns:
        raise KeyError(f"Target column not found: '{target_col}'.")
    if group_col not in df.columns:
        raise KeyError(f"Group column not found: '{group_col}'.")

    # Collect per-group statistics; only groups with >= 2 observations contribute.
    stats: list[GroupVarianceStats] = []
    for gid, grp in df.groupby(group_col, sort=True):
        vals = grp[target_col].to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        stats.append(
            GroupVarianceStats(
                group_id=str(gid),
                n_obs=len(vals),
                weight=float(len(vals) - 1),
                variance=float(np.var(vals, ddof=1)),
            )
        )

    if not stats:
        raise ValueError(
            f"No replicate group in column '{group_col}' has n_g >= 2. "
            "Cannot estimate within-group variance."
        )

    variances = np.array([s.variance for s in stats])
    weights = np.array([s.weight for s in stats])

    rmse_floor = _pool_variances(variances, weights)
    n_groups_used = len(stats)
    n_obs_used = int(sum(s.n_obs for s in stats))

    ci = _bootstrap_ci(
        variances=variances,
        weights=weights,
        n_bootstrap=n_bootstrap,
        ci_level=ci_level,
        seed=seed,
    )

    return NoiseFloorResult(
        rmse_floor=rmse_floor,
        n_groups_used=n_groups_used,
        n_obs_used=n_obs_used,
        ci=ci,
        group_stats=tuple(stats),
    )


def _bootstrap_ci(
    variances: np.ndarray,
    weights: np.ndarray,
    *,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> BootstrapCI:
    """Percentile bootstrap CI over groups (one resample unit = one group).

    Draws ``n_bootstrap`` samples of the same number of groups *with
    replacement*, computes the pooled floor for each, and returns percentile
    bounds. Resampling at the group level (not the row level) reflects
    between-group variability in within-replicate variance.
    """
    rng = np.random.default_rng(seed)
    n_groups = len(variances)
    bootstrap_floors: np.ndarray = np.empty(n_bootstrap, dtype=float)

    for i in range(n_bootstrap):
        idx = rng.integers(0, n_groups, size=n_groups)
        bootstrap_floors[i] = _pool_variances(variances[idx], weights[idx])

    alpha = 1.0 - ci_level
    lower = float(np.percentile(bootstrap_floors, 100.0 * alpha / 2.0))
    upper = float(np.percentile(bootstrap_floors, 100.0 * (1.0 - alpha / 2.0)))

    return BootstrapCI(
        lower=lower,
        upper=upper,
        ci_level=ci_level,
        n_bootstrap=n_bootstrap,
    )


def flag_below_floor(rmse: float, rmse_floor: float) -> bool:
    """Return True iff ``rmse`` is strictly below the noise floor.

    This is the leakage-check predicate from AGENTS.md §5:

        leakage_suspected = (RMSE < RMSE_floor)

    The comparison is strict (no tolerance band) because ``rmse_floor``
    already carries estimation uncertainty through the bootstrap CI. Callers
    that want a softer threshold should compare against ``ci.lower`` instead.

    Parameters
    ----------
    rmse:
        RMSE of the model or protocol being evaluated.
    rmse_floor:
        Noise-floor point estimate from :func:`compute_floor`.
    """
    return float(rmse) < float(rmse_floor)


def rmse_from_r2_and_std(r2: float, std_test: float) -> float:
    """Reconstruct the implied RMSE from a published (R², test-target-std) pair.

    Derivation: R² = 1 − RMSE² / Var(y_test), so RMSE = std_test · sqrt(1 − R²).

    Used by ``run_noise_floor.py`` to evaluate the published-numbers sidebar
    (AGENTS.md §5) without re-running any model.

    Parameters
    ----------
    r2:
        Published R² value (must be in [0, 1]).
    std_test:
        Standard deviation of the test-set targets.

    Raises
    ------
    ValueError
        If ``r2`` is outside [0, 1] or ``std_test`` is non-positive.
    """
    if not 0.0 <= r2 <= 1.0:
        raise ValueError(f"R² must be in [0, 1], got {r2}.")
    if std_test <= 0.0:
        raise ValueError(f"std_test must be positive, got {std_test}.")
    return float(std_test * np.sqrt(1.0 - r2))


def to_record(result: NoiseFloorResult) -> dict[str, object]:
    """Flatten a :class:`NoiseFloorResult` into a single-row dict for CSV output.

    Column names match the ``results/tables/noise_floor.csv`` schema
    (AGENTS.md §5, §9).
    """
    ci_pct = int(round(result.ci.ci_level * 100))
    return {
        "RMSE_floor": result.rmse_floor,
        f"ci_lower_{ci_pct}": result.ci.lower,
        f"ci_upper_{ci_pct}": result.ci.upper,
        "ci_level": result.ci.ci_level,
        "n_bootstrap": result.ci.n_bootstrap,
        "n_groups_used": result.n_groups_used,
        "n_obs_used": result.n_obs_used,
    }
