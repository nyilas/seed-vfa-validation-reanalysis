"""Feature importance for the identifiability experiment (AGENTS.md §10, Exp-C).

Purpose
-------
Demo 2: pH and ζ (zeta potential) are physically coupled and statistically
aliased (r ≈ −0.61).  When both are present (FS2), four importance methods
assign *arbitrary* relative rankings to them.  Removing one (FS6 = no-ζ;
FS2_NO_PH = no-pH) causes the other's importance to migrate upward —
demonstrating that the reported ranking is an artifact of which aliased
variable was retained, not of the physics.

Methods (AGENTS.md §7 / §10)
------------------------------
==================  ===========  ============================================
Method ID           Model        Importance measure
==================  ===========  ============================================
standardized_coef   Ridge        |coefficient| in StandardScaler feature space
permutation         Ridge        mean R² decrease under feature shuffling
shap                CatBoost     mean |SHAP value| over test samples
gpr_ard             GPR (ARD)    1 / ARD length-scale  (sensitivity proxy)
==================  ===========  ============================================

The GPR is a **diagnostic probe only** (AGENTS.md §0): it shows the ARD
mechanism is confused by aliasing — it is not a contribution model.

SHAP is optional (AGENTS.md §2).  :data:`SHAP_AVAILABLE` is set at import
time; the method is silently skipped if the library is absent.

Normalization and ranking
--------------------------
Before cross-method comparison, raw importances are normalized to sum to 1
(negative permutation values are clipped to 0 before the sum).  Ranks are
1-based descending (rank 1 = most important feature).

Kendall τ stability
-------------------
For each feature set, pairwise Kendall τ between the fold-mean importance
rankings is computed across all method pairs.  Each method's
``rank_kendall_tau_vs_other_methods`` is the mean τ against all other methods:
a value near 1 means broad agreement; near 0 or negative means the aliased
pair produces method disagreement.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import kendalltau as _kendalltau
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.inspection import permutation_importance as _sk_perm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import shap as _shap
    SHAP_AVAILABLE: bool = True
except ImportError:  # pragma: no cover
    SHAP_AVAILABLE = False

# ---------------------------------------------------------------------------
# Method identifiers
# ---------------------------------------------------------------------------

METHOD_STANDARDIZED_COEF: str = "standardized_coef"
METHOD_PERMUTATION: str = "permutation"
METHOD_SHAP: str = "shap"
METHOD_GPR_ARD: str = "gpr_ard"

ALL_IMPORTANCE_METHODS: tuple[str, ...] = (
    METHOD_STANDARDIZED_COEF,
    METHOD_PERMUTATION,
    METHOD_SHAP,
    METHOD_GPR_ARD,
)

# The aliased feature pair (Demo 2 focus).
ZETA_COL: str = "Zeta potential [mV]"
PH_COL: str = "pH"
ALIASED_PAIR: frozenset[str] = frozenset({ZETA_COL, PH_COL})

# Per-fold permutation repeats.
_PERMUTATION_N_REPEATS: int = 10

# GPR optimizer restarts: more → better global optimum, slower.
_GPR_N_RESTARTS: int = 5

# Output column names matching AGENTS.md §9 schema.
IMPORTANCE_STABILITY_COLUMNS: tuple[str, ...] = (
    "feature_set",
    "method",
    "protocol",
    "feature",
    "importance_mean",
    "importance_std",
    "rank_mean",
    "rank_kendall_tau_vs_other_methods",
)

# ---------------------------------------------------------------------------
# Data type
# ---------------------------------------------------------------------------


@dataclass
class FoldImportance:
    """Raw and normalised importances for one (method, fold) evaluation.

    Attributes
    ----------
    feature_set_id:
        One of ``"FS2"``, ``"FS6"``, ``"FS2_NO_PH"``.
    method:
        One of :data:`ALL_IMPORTANCE_METHODS`.
    fold_key:
        ``(split_id, fold)`` from the :class:`~seed_vfa.splits.SplitResult`.
    feature_columns:
        Ordered feature names; same order as the importance arrays.
    raw_importances:
        Unsigned magnitude for coef / SHAP / GPR; signed decrease for
        permutation.  Shape: ``(n_features,)``.
    normalized_importances:
        Clip-then-scale to sum to 1.  Shape: ``(n_features,)``.
    """

    feature_set_id: str
    method: str
    fold_key: tuple[int, int]
    feature_columns: tuple[str, ...]
    raw_importances: np.ndarray
    normalized_importances: np.ndarray


# ---------------------------------------------------------------------------
# Utility: normalization and ranking
# ---------------------------------------------------------------------------


def normalize_importances(raw: np.ndarray) -> np.ndarray:
    """Clip negative values to 0 then rescale to sum to 1.

    Negative permutation importances mean a feature actively hurts the
    model.  Clipping to zero before normalization treats them as
    "not informative" rather than penalising the sum.

    Parameters
    ----------
    raw:
        1-D array of raw importance values (any sign).

    Returns
    -------
    numpy.ndarray
        Non-negative array summing to 1.  Returns a uniform distribution
        if all values are zero or negative.
    """
    clipped = np.maximum(raw, 0.0)
    total = float(clipped.sum())
    if total == 0.0:
        return np.full(len(raw), 1.0 / len(raw))
    return clipped / total


def rank_importances(importances: np.ndarray) -> np.ndarray:
    """Return 1-based ranks (rank 1 = most important feature).

    Ties broken by stable sort (original order preserved).

    Parameters
    ----------
    importances:
        1-D array of importance values.

    Returns
    -------
    numpy.ndarray of int
        Array of the same length with ranks 1 … n.
    """
    order = np.argsort(-importances, kind="stable")
    ranks = np.empty(len(importances), dtype=int)
    ranks[order] = np.arange(1, len(importances) + 1)
    return ranks


# ---------------------------------------------------------------------------
# Internal importance extractors
# ---------------------------------------------------------------------------


def _standardized_coef(pipeline: Pipeline) -> np.ndarray:
    """Return |coef| from a fitted Ridge pipeline (standardised feature space)."""
    return np.abs(pipeline.named_steps["model"].coef_)


def _permutation(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    seed: int,
    n_repeats: int,
) -> np.ndarray:
    """Mean R² decrease under feature shuffling (signed; can be negative)."""
    result = _sk_perm(
        pipeline, X_test, y_test,
        n_repeats=n_repeats,
        random_state=seed,
        scoring="r2",
    )
    return result.importances_mean


def _shap_importance(pipeline: Pipeline, X_test: pd.DataFrame) -> np.ndarray:
    """Mean |SHAP value| over test samples from a fitted CatBoost pipeline."""
    prep = pipeline.named_steps["prep"]
    model = pipeline.named_steps["model"]
    X_transformed = prep.transform(X_test)
    explainer = _shap.TreeExplainer(model)
    shap_values = explainer(X_transformed).values   # (n_samples, n_features)
    return np.abs(shap_values).mean(axis=0)


def _gpr_ard(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    *,
    seed: int,
    n_restarts: int,
) -> np.ndarray:
    """Fit GPR with ARD RBF kernel; return 1/length_scale as importance proxy.

    A shorter ARD length-scale means the posterior varies more rapidly with
    changes in that feature → the model is more *sensitive* to it → higher
    importance.  ``importance_i = 1 / length_scale_i``.

    Features are z-scored internally (StandardScaler fit on ``X_train``).
    Length-scale bounds ``[1e-2, 1e2]`` prevent the optimiser from collapsing
    to pathological extremes.

    Notes
    -----
    This function may produce convergence warnings during kernel hyperparameter
    optimisation; these are suppressed because they are expected on small
    datasets and do not indicate a code error.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    kernel = (
        RBF(
            length_scale=np.ones(n_features),
            length_scale_bounds=(1e-2, 1e2),
        )
        + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    )
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-6,
        n_restarts_optimizer=n_restarts,
        random_state=seed,
        normalize_y=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gpr.fit(X_scaled, y_train)

    length_scales: np.ndarray = gpr.kernel_.k1.length_scale
    return 1.0 / length_scales


# ---------------------------------------------------------------------------
# High-level fold evaluation
# ---------------------------------------------------------------------------


def compute_fold_importance(
    method: str,
    df: pd.DataFrame,
    split_result,
    feature_set_id: str,
    *,
    config,
) -> FoldImportance | None:
    """Compute importances for one (method, fold) combination.

    Builds and fits the appropriate model *inside the training fold only*,
    then extracts importances.  Returns ``None`` when the method is
    unavailable (e.g. SHAP not installed).

    Parameters
    ----------
    method:
        One of :data:`ALL_IMPORTANCE_METHODS`.
    df:
        Full grouped DataFrame (``seed_with_groups.csv``).
    split_result:
        A :class:`~seed_vfa.splits.SplitResult` with ``train_idx`` and
        ``test_idx``.
    feature_set_id:
        One of ``"FS2"``, ``"FS6"``, ``"FS2_NO_PH"``.
    config:
        Loaded :class:`~seed_vfa.config.Config`.

    Returns
    -------
    FoldImportance or None
    """
    # Deferred imports to avoid circular dependencies at module level.
    from .data import TARGET  # noqa: PLC0415
    from .features import get_columns, select_features  # noqa: PLC0415
    from .models import build_pipeline  # noqa: PLC0415

    feature_cols = get_columns(feature_set_id)
    train_rows = df.iloc[split_result.train_idx]
    test_rows = df.iloc[split_result.test_idx]
    X_train = select_features(feature_set_id, train_rows)
    y_train = train_rows[TARGET].to_numpy(dtype=float)
    X_test = select_features(feature_set_id, test_rows)
    y_test = test_rows[TARGET].to_numpy(dtype=float)

    if method == METHOD_STANDARDIZED_COEF:
        pipe = build_pipeline(
            "ridge", feature_cols, seed=config.seed, models_config=config.models
        )
        pipe.fit(X_train, y_train)
        raw = _standardized_coef(pipe)

    elif method == METHOD_PERMUTATION:
        pipe = build_pipeline(
            "ridge", feature_cols, seed=config.seed, models_config=config.models
        )
        pipe.fit(X_train, y_train)
        raw = _permutation(
            pipe, X_test, y_test,
            seed=config.seed, n_repeats=_PERMUTATION_N_REPEATS,
        )

    elif method == METHOD_SHAP:
        if not SHAP_AVAILABLE:
            return None
        pipe = build_pipeline(
            "catboost", feature_cols, seed=config.seed, models_config=config.models
        )
        pipe.fit(X_train, y_train)
        raw = _shap_importance(pipe, X_test)

    elif method == METHOD_GPR_ARD:
        raw = _gpr_ard(
            X_train.to_numpy(dtype=float),
            y_train,
            len(feature_cols),
            seed=config.seed,
            n_restarts=_GPR_N_RESTARTS,
        )

    else:
        raise ValueError(
            f"Unknown importance method {method!r}. "
            f"Valid: {ALL_IMPORTANCE_METHODS}"
        )

    return FoldImportance(
        feature_set_id=feature_set_id,
        method=method,
        fold_key=(split_result.split_id, split_result.fold),
        feature_columns=feature_cols,
        raw_importances=raw,
        normalized_importances=normalize_importances(raw),
    )


# ---------------------------------------------------------------------------
# Aggregation across folds
# ---------------------------------------------------------------------------


def aggregate_fold_importances(
    fold_importances: list[FoldImportance],
    *,
    protocol: str,
) -> pd.DataFrame:
    """Aggregate fold-level importances into mean, std, and mean rank.

    Returns a DataFrame with one row per feature.
    The ``rank_kendall_tau_vs_other_methods`` column is left as ``NaN`` and
    is filled later by :func:`add_kendall_tau`.

    Parameters
    ----------
    fold_importances:
        Non-empty list of :class:`FoldImportance` objects sharing the same
        ``feature_set_id`` and ``method``.
    protocol:
        Protocol label written into the ``protocol`` column (e.g. ``"C"``).

    Returns
    -------
    pandas.DataFrame
        Schema: ``feature_set, method, protocol, feature, importance_mean,
        importance_std, rank_mean, rank_kendall_tau_vs_other_methods``.
    """
    if not fold_importances:
        raise ValueError("fold_importances must not be empty.")

    feature_set_id = fold_importances[0].feature_set_id
    method = fold_importances[0].method
    feature_columns = fold_importances[0].feature_columns

    # shape: (n_folds, n_features)
    norm_matrix = np.stack(
        [fi.normalized_importances for fi in fold_importances], axis=0
    )
    rank_matrix = np.stack(
        [rank_importances(fi.normalized_importances) for fi in fold_importances],
        axis=0,
    )

    return pd.DataFrame(
        {
            "feature_set": feature_set_id,
            "method": method,
            "protocol": protocol,
            "feature": list(feature_columns),
            "importance_mean": norm_matrix.mean(axis=0),
            "importance_std": norm_matrix.std(axis=0, ddof=0),
            "rank_mean": rank_matrix.mean(axis=0).astype(float),
            "rank_kendall_tau_vs_other_methods": np.nan,
        }
    )


def add_kendall_tau(agg_df: pd.DataFrame) -> pd.DataFrame:
    """Fill ``rank_kendall_tau_vs_other_methods`` in an aggregated table.

    For each ``(feature_set, protocol)`` group, computes pairwise Kendall τ
    between methods using their ``importance_mean`` arrays.  Each method's
    value is the mean τ against all other methods.

    A high τ (≈1) means the method's ranking agrees with the consensus.
    A low τ on the aliased pair indicates method disagreement driven by
    the pH/ζ aliasing (Demo 2 key result).

    Parameters
    ----------
    agg_df:
        Concatenated output of :func:`aggregate_fold_importances` for all
        methods and feature sets.

    Returns
    -------
    pandas.DataFrame
        Same DataFrame with the ``rank_kendall_tau_vs_other_methods``
        column populated.
    """
    out = agg_df.copy()

    for (fs_id, proto), group in out.groupby(["feature_set", "protocol"]):
        methods = group["method"].unique()
        if len(methods) < 2:
            continue

        method_imp: dict[str, np.ndarray] = {}
        for m in methods:
            method_imp[m] = group.loc[group["method"] == m, "importance_mean"].to_numpy()

        tau_sum: dict[str, float] = {m: 0.0 for m in methods}
        tau_cnt: dict[str, int] = {m: 0 for m in methods}

        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1:]:
                tau, _ = _kendalltau(method_imp[m1], method_imp[m2])
                if not np.isnan(tau):
                    tau_sum[m1] += tau
                    tau_cnt[m1] += 1
                    tau_sum[m2] += tau
                    tau_cnt[m2] += 1

        for m in methods:
            tau_val = tau_sum[m] / tau_cnt[m] if tau_cnt[m] > 0 else np.nan
            mask = (
                (out["feature_set"] == fs_id)
                & (out["protocol"] == proto)
                & (out["method"] == m)
            )
            out.loc[mask, "rank_kendall_tau_vs_other_methods"] = tau_val

    return out
