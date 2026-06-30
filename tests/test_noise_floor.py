"""Acceptance tests for the noise-floor layer (AGENTS.md §11, test_noise_floor).

Covers:
- ``test_floor_positive_and_reasonable``: 0.1 < RMSE_floor < 0.4 (§11).
- ``test_below_floor_flag``: fires iff RMSE < floor (§11).
- Formula correctness on known synthetic examples.
- Bootstrap CI encloses the point estimate and is positive.
- Determinism: two calls with the same seed produce identical results.
- Published sidebar: best-augmented R²=0.937 → RMSE implied < floor (§5).
- Error paths: missing columns, no groups with n_g >= 2.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from seed_vfa import noise_floor as nf
from seed_vfa.config import load_config
from seed_vfa.data import TARGET
from seed_vfa.groups import REPLICATE_GROUP_ID


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def grouped() -> pd.DataFrame:
    """Load the real grouped dataset produced by run_data_audit.py."""
    config = load_config()
    path = config.paths.seed_with_groups
    if not path.is_file():
        pytest.skip(
            f"{path} not found — run scripts/run_data_audit.py before this test."
        )
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def real_result(grouped: pd.DataFrame) -> nf.NoiseFloorResult:
    """Compute the floor from the real dataset (cached at module scope)."""
    config = load_config()
    return nf.compute_floor(
        grouped,
        target_col=TARGET,
        group_col=REPLICATE_GROUP_ID,
        n_bootstrap=config.noise_floor.n_bootstrap,
        ci_level=config.noise_floor.ci_level,
        seed=config.seed,
    )


# ---------------------------------------------------------------------------
# §11 mandatory tests
# ---------------------------------------------------------------------------

def test_floor_positive_and_reasonable(real_result: nf.NoiseFloorResult) -> None:
    """AGENTS.md §11: 0.1 < RMSE_floor < 0.4."""
    config = load_config()
    floor = real_result.rmse_floor
    assert floor > config.noise_floor.floor_min, (
        f"RMSE_floor={floor:.4f} is not above the minimum "
        f"({config.noise_floor.floor_min})."
    )
    assert floor < config.noise_floor.floor_max, (
        f"RMSE_floor={floor:.4f} is not below the maximum "
        f"({config.noise_floor.floor_max})."
    )


def test_below_floor_flag_fires_iff_below(real_result: nf.NoiseFloorResult) -> None:
    """AGENTS.md §11: flag fires iff RMSE < floor, strictly (no tolerance)."""
    floor = real_result.rmse_floor

    # Strictly below: flag must be True.
    assert nf.flag_below_floor(floor - 1e-9, floor) is True

    # Exactly equal: flag must be False (strict inequality).
    assert nf.flag_below_floor(floor, floor) is False

    # Strictly above: flag must be False.
    assert nf.flag_below_floor(floor + 1e-9, floor) is False


# ---------------------------------------------------------------------------
# Formula correctness on controlled synthetic examples
# ---------------------------------------------------------------------------

def _make_df(
    group_pairs: list[tuple[float, float]],
    singletons: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal DataFrame with replicate_group_id and target columns."""
    rows: list[dict[str, object]] = []
    for gid, (y1, y2) in enumerate(group_pairs):
        rows.append({"replicate_group_id": f"G{gid:02d}", "target": y1})
        rows.append({"replicate_group_id": f"G{gid:02d}", "target": y2})
    for sid, y in enumerate(singletons or []):
        rows.append({"replicate_group_id": f"S{sid:02d}", "target": y})
    return pd.DataFrame(rows)


def test_formula_two_groups_equal_variance() -> None:
    """Two pairs each with s²=1.0, equal weights → floor = 1.0.

    For a pair (a, b), s² = (a−b)² / 2. We need s²=1 → |a−b|=sqrt(2).
    """
    delta = math.sqrt(2.0)
    df = _make_df([(0.0, delta), (1.0, 1.0 + delta)])
    result = nf.compute_floor(
        df, target_col="target", group_col="replicate_group_id",
        n_bootstrap=100, ci_level=0.95, seed=0,
    )
    assert result.n_groups_used == 2
    assert result.n_obs_used == 4
    assert abs(result.rmse_floor - 1.0) < 1e-12


def test_formula_unequal_weights() -> None:
    """Three obs in one group (weight=2), two in another (weight=1).

    group A: [0, 1, 2] → s² = 1.0, weight = 2
    group B: [0, 4]    → s² = 8.0, weight = 1
    pooled = sqrt((2·1 + 1·8) / 3) = sqrt(10/3)
    """
    rows = [
        {"replicate_group_id": "A", "target": 0.0},
        {"replicate_group_id": "A", "target": 1.0},
        {"replicate_group_id": "A", "target": 2.0},
        {"replicate_group_id": "B", "target": 0.0},
        {"replicate_group_id": "B", "target": 4.0},
    ]
    df = pd.DataFrame(rows)
    result = nf.compute_floor(
        df, target_col="target", group_col="replicate_group_id",
        n_bootstrap=100, ci_level=0.95, seed=0,
    )
    expected = float(np.sqrt(10.0 / 3.0))
    assert abs(result.rmse_floor - expected) < 1e-12
    assert result.n_groups_used == 2
    assert result.n_obs_used == 5


def test_singletons_excluded() -> None:
    """Singleton groups must not contribute to the floor estimate."""
    df = _make_df([(0.0, 1.0)], singletons=[99.9])
    result = nf.compute_floor(
        df, target_col="target", group_col="replicate_group_id",
        n_bootstrap=50, ci_level=0.95, seed=0,
    )
    assert result.n_groups_used == 1
    assert result.n_obs_used == 2


def test_real_dataset_counts(real_result: nf.NoiseFloorResult) -> None:
    """Real dataset: 33 replicate pairs → n_groups_used=33, n_obs_used=66."""
    assert real_result.n_groups_used == 33
    assert real_result.n_obs_used == 66


# ---------------------------------------------------------------------------
# Bootstrap CI properties
# ---------------------------------------------------------------------------

def test_bootstrap_ci_encloses_point_estimate(real_result: nf.NoiseFloorResult) -> None:
    """The bootstrap CI must bracket the point estimate."""
    ci = real_result.ci
    assert ci.lower <= real_result.rmse_floor <= ci.upper, (
        f"CI [{ci.lower:.4f}, {ci.upper:.4f}] does not enclose "
        f"RMSE_floor={real_result.rmse_floor:.4f}."
    )


def test_bootstrap_ci_positive(real_result: nf.NoiseFloorResult) -> None:
    assert real_result.ci.lower > 0.0


def test_bootstrap_deterministic(grouped: pd.DataFrame) -> None:
    """Two calls with the same seed return bit-for-bit identical results."""
    config = load_config()
    kwargs: dict[str, object] = dict(
        target_col=TARGET,
        group_col=REPLICATE_GROUP_ID,
        n_bootstrap=500,   # reduced for speed; determinism is seed-independent
        ci_level=config.noise_floor.ci_level,
        seed=config.seed,
    )
    r1 = nf.compute_floor(grouped, **kwargs)  # type: ignore[arg-type]
    r2 = nf.compute_floor(grouped, **kwargs)  # type: ignore[arg-type]
    assert r1.rmse_floor == r2.rmse_floor
    assert r1.ci.lower == r2.ci.lower
    assert r1.ci.upper == r2.ci.upper


# ---------------------------------------------------------------------------
# Published-numbers sidebar (AGENTS.md §5)
# ---------------------------------------------------------------------------

def test_sidebar_best_augmented_below_floor(real_result: nf.NoiseFloorResult) -> None:
    """Source paper best augmented R²=0.937 → implied RMSE ≈ 0.21 < floor."""
    rmse_implied = nf.rmse_from_r2_and_std(0.937, 0.835)
    assert nf.flag_below_floor(rmse_implied, real_result.rmse_floor), (
        f"Expected RMSE_implied={rmse_implied:.4f} < RMSE_floor="
        f"{real_result.rmse_floor:.4f}."
    )


def test_sidebar_baseline_above_floor(real_result: nf.NoiseFloorResult) -> None:
    """Source paper baseline R²=0.885 → implied RMSE ≈ 0.283 > floor."""
    rmse_implied = nf.rmse_from_r2_and_std(0.885, 0.835)
    assert not nf.flag_below_floor(rmse_implied, real_result.rmse_floor), (
        f"Baseline RMSE_implied={rmse_implied:.4f} was unexpectedly below "
        f"RMSE_floor={real_result.rmse_floor:.4f}."
    )


def test_rmse_from_r2_and_std_known_value() -> None:
    """sqrt(1 − 0.885) · 0.835 ≈ 0.283 (source paper baseline round-trip)."""
    result = nf.rmse_from_r2_and_std(0.885, 0.835)
    assert abs(result - 0.283) < 0.001, f"Got {result:.4f}, expected ≈0.283"


def test_rmse_from_r2_and_std_invalid_r2() -> None:
    with pytest.raises(ValueError, match="R²"):
        nf.rmse_from_r2_and_std(1.1, 1.0)


def test_rmse_from_r2_and_std_invalid_std() -> None:
    with pytest.raises(ValueError, match="std_test"):
        nf.rmse_from_r2_and_std(0.9, 0.0)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_missing_target_col_raises() -> None:
    df = pd.DataFrame({"group": ["A", "A"], "other": [1.0, 2.0]})
    with pytest.raises(KeyError, match="Target column"):
        nf.compute_floor(
            df, target_col="target", group_col="group",
            n_bootstrap=10, ci_level=0.95, seed=0,
        )


def test_missing_group_col_raises() -> None:
    df = pd.DataFrame({"target": [1.0, 2.0]})
    with pytest.raises(KeyError, match="Group column"):
        nf.compute_floor(
            df, target_col="target", group_col="group",
            n_bootstrap=10, ci_level=0.95, seed=0,
        )


def test_all_singletons_raises() -> None:
    """All singletons → no within-group variance estimable → ValueError."""
    df = pd.DataFrame({"group": ["A", "B", "C"], "target": [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match="n_g >= 2"):
        nf.compute_floor(
            df, target_col="target", group_col="group",
            n_bootstrap=10, ci_level=0.95, seed=0,
        )
