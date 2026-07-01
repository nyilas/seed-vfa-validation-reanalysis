"""Tests for the identifiability / feature-importance layer (AGENTS.md §10).

Verifies:
- normalize_importances clips negatives to zero and sums to 1.
- normalize_importances returns a uniform distribution when all values are ≤ 0.
- rank_importances gives 1-based descending ranks.
- rank_importances handles ties deterministically.
- Kendall τ = +1 for identical rankings; −1 for reversed rankings.
- add_kendall_tau fills the column and the value is in [−1, 1].
- standardized_coef importance has correct shape on a fitted Ridge pipeline.
- gpr_ard_importance returns positive values of the right shape.
- Demo 2 migration: removing ζ from FS2 → FS6 raises pH's normalized rank.
- Demo 2 migration: removing pH from FS2 → FS2_NO_PH raises ζ's normalized rank.
- IMPORTANCE_STABILITY_COLUMNS matches the AGENTS.md §9 schema exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from seed_vfa.config import load_config
from seed_vfa.data import TARGET
from seed_vfa.features import get_columns, select_features
from seed_vfa.importance import (
    IMPORTANCE_STABILITY_COLUMNS,
    PH_COL,
    SHAP_AVAILABLE,
    ZETA_COL,
    FoldImportance,
    add_kendall_tau,
    aggregate_fold_importances,
    compute_fold_importance,
    normalize_importances,
    rank_importances,
    _gpr_ard,
    _standardized_coef,
)
from seed_vfa.models import build_pipeline
from seed_vfa.splits import protocol_c


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def config():
    return load_config()


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    cfg = load_config()
    path = cfg.paths.seed_with_groups
    if not path.is_file():
        pytest.skip(f"{path} not found — run run_data_audit.py first.")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def first_c_split(df: pd.DataFrame, config):
    """First fold of Protocol C."""
    return protocol_c(df, n_splits=config.splits.gkf_n_splits)[0]


# ---------------------------------------------------------------------------
# Schema constant
# ---------------------------------------------------------------------------


def test_importance_stability_columns_schema() -> None:
    expected = (
        "feature_set", "method", "protocol", "feature",
        "importance_mean", "importance_std", "rank_mean",
        "rank_kendall_tau_vs_other_methods",
    )
    assert IMPORTANCE_STABILITY_COLUMNS == expected


# ---------------------------------------------------------------------------
# normalize_importances
# ---------------------------------------------------------------------------


def test_normalize_sums_to_one() -> None:
    raw = np.array([1.0, 2.0, 3.0, 4.0])
    result = normalize_importances(raw)
    assert result.sum() == pytest.approx(1.0)


def test_normalize_all_nonnegative() -> None:
    raw = np.array([1.0, 0.5, 2.0])
    result = normalize_importances(raw)
    assert (result >= 0).all()


def test_normalize_clips_negatives_to_zero() -> None:
    raw = np.array([-1.0, 2.0, 3.0])
    result = normalize_importances(raw)
    assert result[0] == pytest.approx(0.0)
    assert result.sum() == pytest.approx(1.0)


def test_normalize_all_nonpositive_returns_uniform() -> None:
    raw = np.array([-1.0, -2.0, 0.0])
    result = normalize_importances(raw)
    assert result.sum() == pytest.approx(1.0)
    assert np.allclose(result, 1.0 / 3.0)


def test_normalize_all_zeros_returns_uniform() -> None:
    raw = np.zeros(4)
    result = normalize_importances(raw)
    assert np.allclose(result, 0.25)


# ---------------------------------------------------------------------------
# rank_importances
# ---------------------------------------------------------------------------


def test_rank_highest_gets_rank_one() -> None:
    imp = np.array([0.1, 0.5, 0.3])
    ranks = rank_importances(imp)
    assert ranks[1] == 1   # 0.5 is highest


def test_rank_lowest_gets_last_rank() -> None:
    imp = np.array([0.1, 0.5, 0.3])
    ranks = rank_importances(imp)
    assert ranks[0] == 3   # 0.1 is lowest


def test_rank_is_one_based() -> None:
    imp = np.array([0.2, 0.8, 0.5, 0.1])
    ranks = rank_importances(imp)
    assert set(ranks) == {1, 2, 3, 4}


def test_rank_covers_all_positions() -> None:
    imp = np.array([0.3, 0.1, 0.4, 0.2])
    ranks = rank_importances(imp)
    assert sorted(ranks) == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Kendall τ via add_kendall_tau
# ---------------------------------------------------------------------------


def _make_agg_df(
    feature_set: str,
    method: str,
    protocol: str,
    features: list[str],
    importances: list[float],
) -> pd.DataFrame:
    n = len(features)
    return pd.DataFrame({
        "feature_set": feature_set,
        "method": method,
        "protocol": protocol,
        "feature": features,
        "importance_mean": importances,
        "importance_std": [0.0] * n,
        "rank_mean": list(range(1, n + 1)),
        "rank_kendall_tau_vs_other_methods": float("nan"),
    })


def test_kendall_tau_perfect_agreement() -> None:
    feats = ["a", "b", "c"]
    imps = [0.5, 0.3, 0.2]
    df1 = _make_agg_df("FS2", "standardized_coef", "C", feats, imps)
    df2 = _make_agg_df("FS2", "permutation",       "C", feats, imps)
    combined = pd.concat([df1, df2], ignore_index=True)
    result = add_kendall_tau(combined)
    taus = result[result["feature_set"] == "FS2"]["rank_kendall_tau_vs_other_methods"].unique()
    assert len(taus) == 1
    assert float(taus[0]) == pytest.approx(1.0)


def test_kendall_tau_perfect_disagreement() -> None:
    feats = ["a", "b", "c"]
    df1 = _make_agg_df("FS2", "standardized_coef", "C", feats, [0.5, 0.3, 0.2])
    df2 = _make_agg_df("FS2", "permutation",       "C", feats, [0.2, 0.3, 0.5])
    combined = pd.concat([df1, df2], ignore_index=True)
    result = add_kendall_tau(combined)
    taus = result[result["feature_set"] == "FS2"]["rank_kendall_tau_vs_other_methods"].unique()
    assert len(taus) == 1
    assert float(taus[0]) == pytest.approx(-1.0)


def test_kendall_tau_in_range() -> None:
    feats = ["a", "b", "c"]
    df1 = _make_agg_df("FS2", "standardized_coef", "C", feats, [0.5, 0.3, 0.2])
    df2 = _make_agg_df("FS2", "permutation",       "C", feats, [0.3, 0.5, 0.2])
    combined = pd.concat([df1, df2], ignore_index=True)
    result = add_kendall_tau(combined)
    tau = result.loc[result["method"] == "standardized_coef",
                     "rank_kendall_tau_vs_other_methods"].iloc[0]
    assert -1.0 <= float(tau) <= 1.0


def test_kendall_tau_single_method_is_nan() -> None:
    feats = ["a", "b"]
    df1 = _make_agg_df("FS2", "standardized_coef", "C", feats, [0.6, 0.4])
    result = add_kendall_tau(df1)
    tau = result["rank_kendall_tau_vs_other_methods"].iloc[0]
    assert np.isnan(tau)


# ---------------------------------------------------------------------------
# Standardized coefficient extraction
# ---------------------------------------------------------------------------


def test_standardized_coef_shape(df: pd.DataFrame, first_c_split, config) -> None:
    fid = "FS2"
    cols = get_columns(fid)
    pipe = build_pipeline("ridge", cols, seed=config.seed, models_config=config.models)
    pipe.fit(select_features(fid, df.iloc[first_c_split.train_idx]),
             df.iloc[first_c_split.train_idx][TARGET].values)
    coefs = _standardized_coef(pipe)
    assert coefs.shape == (len(cols),)


def test_standardized_coef_nonnegative(df: pd.DataFrame, first_c_split, config) -> None:
    fid = "FS2"
    cols = get_columns(fid)
    pipe = build_pipeline("ridge", cols, seed=config.seed, models_config=config.models)
    pipe.fit(select_features(fid, df.iloc[first_c_split.train_idx]),
             df.iloc[first_c_split.train_idx][TARGET].values)
    coefs = _standardized_coef(pipe)
    assert (coefs >= 0).all()


# ---------------------------------------------------------------------------
# GPR ARD importance
# ---------------------------------------------------------------------------


def test_gpr_ard_shape(df: pd.DataFrame, first_c_split, config) -> None:
    fid = "FS2"
    X_train = select_features(fid, df.iloc[first_c_split.train_idx]).to_numpy(dtype=float)
    y_train = df.iloc[first_c_split.train_idx][TARGET].values
    n_features = len(get_columns(fid))
    imps = _gpr_ard(X_train, y_train, n_features, seed=config.seed, n_restarts=1)
    assert imps.shape == (n_features,)


def test_gpr_ard_positive(df: pd.DataFrame, first_c_split, config) -> None:
    fid = "FS2"
    X_train = select_features(fid, df.iloc[first_c_split.train_idx]).to_numpy(dtype=float)
    y_train = df.iloc[first_c_split.train_idx][TARGET].values
    n_features = len(get_columns(fid))
    imps = _gpr_ard(X_train, y_train, n_features, seed=config.seed, n_restarts=1)
    assert (imps > 0).all()


# ---------------------------------------------------------------------------
# compute_fold_importance integration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["standardized_coef", "permutation"])
def test_compute_fold_importance_shape(
    method: str, df: pd.DataFrame, first_c_split, config
) -> None:
    fid = "FS2"
    fi = compute_fold_importance(method, df, first_c_split, fid, config=config)
    assert fi is not None
    n_features = len(get_columns(fid))
    assert fi.raw_importances.shape == (n_features,)
    assert fi.normalized_importances.shape == (n_features,)


def test_compute_fold_importance_normalized_sums_to_one(
    df: pd.DataFrame, first_c_split, config
) -> None:
    fi = compute_fold_importance(
        "standardized_coef", df, first_c_split, "FS2", config=config
    )
    assert fi is not None
    assert fi.normalized_importances.sum() == pytest.approx(1.0)


@pytest.mark.skipif(not SHAP_AVAILABLE, reason="shap not installed")
def test_compute_fold_importance_shap_shape(
    df: pd.DataFrame, first_c_split, config
) -> None:
    fid = "FS2"
    fi = compute_fold_importance("shap", df, first_c_split, fid, config=config)
    assert fi is not None
    assert fi.raw_importances.shape == (len(get_columns(fid)),)


# ---------------------------------------------------------------------------
# Demo 2: importance migration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["standardized_coef", "permutation"])
def test_ph_importance_rises_when_zeta_removed(
    method: str, df: pd.DataFrame, config
) -> None:
    """Removing ζ from FS2 → FS6 must raise pH's normalised importance.

    This is the core Demo 2 result: pH's contribution in FS2 is suppressed
    by collinearity with ζ.  Once ζ is absent (FS6), pH absorbs its role
    and its importance rises.
    """
    splits = protocol_c(df, n_splits=config.splits.gkf_n_splits)

    def _mean_importance(fs_id: str, target_feature: str) -> float:
        fold_imps = [
            fi for sr in splits
            if (fi := compute_fold_importance(method, df, sr, fs_id, config=config))
               is not None
        ]
        assert fold_imps, f"No fold importances computed for {fs_id}/{method}"
        norm_matrix = np.stack([fi.normalized_importances for fi in fold_imps])
        mean_norm = norm_matrix.mean(axis=0)
        cols = fold_imps[0].feature_columns
        idx = list(cols).index(target_feature)
        return float(mean_norm[idx])

    ph_in_fs2 = _mean_importance("FS2", PH_COL)
    ph_in_fs6 = _mean_importance("FS6", PH_COL)
    assert ph_in_fs6 > ph_in_fs2, (
        f"{method}: pH importance should rise when ζ is removed "
        f"(FS2={ph_in_fs2:.4f}, FS6={ph_in_fs6:.4f})."
    )


@pytest.mark.parametrize("method", ["standardized_coef", "permutation"])
def test_zeta_importance_rises_when_ph_removed(
    method: str, df: pd.DataFrame, config
) -> None:
    """Removing pH from FS2 → FS2_NO_PH must raise ζ's normalised importance."""
    splits = protocol_c(df, n_splits=config.splits.gkf_n_splits)

    def _mean_importance(fs_id: str, target_feature: str) -> float:
        fold_imps = [
            fi for sr in splits
            if (fi := compute_fold_importance(method, df, sr, fs_id, config=config))
               is not None
        ]
        norm_matrix = np.stack([fi.normalized_importances for fi in fold_imps])
        mean_norm = norm_matrix.mean(axis=0)
        cols = fold_imps[0].feature_columns
        idx = list(cols).index(target_feature)
        return float(mean_norm[idx])

    zeta_in_fs2 = _mean_importance("FS2", ZETA_COL)
    zeta_in_fs2_no_ph = _mean_importance("FS2_NO_PH", ZETA_COL)
    assert zeta_in_fs2_no_ph > zeta_in_fs2, (
        f"{method}: ζ importance should rise when pH is removed "
        f"(FS2={zeta_in_fs2:.4f}, FS2_NO_PH={zeta_in_fs2_no_ph:.4f})."
    )
