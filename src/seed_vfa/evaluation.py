"""Fold-level evaluation, prediction collection, and metric aggregation (AGENTS.md §9).

Purpose
-------
Implement the three-step evaluation pipeline that AGENTS.md §1 (invariants 6–8)
requires:

1. For each fold, compute fold predictions (y_true, y_pred) and fold-level metrics
   (R², RMSE, MAE, split-size counts) in a single call so nothing is discarded.
2. Save per-row predictions with the exact §9 schema to ``all_predictions.csv``.
3. Aggregate fold-level metrics into mean + std across folds for
   ``model_metrics.csv``.

Aggregate metrics are ALWAYS derived from fold-level metrics, never recomputed on
a fresh pass over predictions — satisfying AGENTS.md §1 invariant 6.

Schemas (AGENTS.md §9)
----------------------
Predictions (``all_predictions.csv``)::

    row_id, condition_id, replicate_group_id, membrane_id, feed_type,
    protocol, split_id, fold, model, feature_set, y_true, y_pred, residual

Model metrics (``model_metrics.csv``)::

    protocol, model, feature_set,
    R2_mean, R2_std, RMSE_mean, RMSE_std, MAE_mean, MAE_std,
    n_train, n_test, n_conditions_train, n_conditions_test,
    n_domains_train, n_domains_test, below_floor
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import TARGET
from .groups import CONDITION_ID, FEED_TYPE, MEMBRANE_ID, REPLICATE_GROUP_ID, ROW_ID
from .noise_floor import flag_below_floor
from .splits import SplitResult

# ---------------------------------------------------------------------------
# Schema constants (AGENTS.md §9)
# ---------------------------------------------------------------------------

PREDICTIONS_COLUMNS: tuple[str, ...] = (
    ROW_ID,
    CONDITION_ID,
    REPLICATE_GROUP_ID,
    MEMBRANE_ID,
    FEED_TYPE,
    "protocol",
    "split_id",
    "fold",
    "model",
    "feature_set",
    "y_true",
    "y_pred",
    "residual",
)

METRICS_COLUMNS: tuple[str, ...] = (
    "protocol",
    "model",
    "feature_set",
    "R2_mean",
    "R2_std",
    "RMSE_mean",
    "RMSE_std",
    "MAE_mean",
    "MAE_std",
    "n_train",
    "n_test",
    "n_conditions_train",
    "n_conditions_test",
    "n_domains_train",
    "n_domains_test",
    "below_floor",
)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldMetrics:
    """Metrics for a single train/test fold.

    All fields are populated by :func:`evaluate_fold` in one pass over the
    same y_true/y_pred arrays used to build the predictions DataFrame,
    satisfying AGENTS.md §1 invariant 6.
    """

    r2: float
    rmse: float
    mae: float
    n_train: int
    n_test: int
    n_conditions_train: int
    n_conditions_test: int
    n_domains_train: int
    n_domains_test: int


@dataclass(frozen=True)
class AggregatedMetrics:
    """Cross-fold aggregated metrics for one (protocol, model, feature_set) combination.

    Matches the ``model_metrics.csv`` schema in AGENTS.md §9 exactly.
    All ``_mean`` / ``_std`` fields use population std (ddof=0) across folds.
    Integer fields (n_train, n_test, …) are the rounded mean across folds.
    """

    protocol: str
    model: str
    feature_set: str
    R2_mean: float
    R2_std: float
    RMSE_mean: float
    RMSE_std: float
    MAE_mean: float
    MAE_std: float
    n_train: int
    n_test: int
    n_conditions_train: int
    n_conditions_test: int
    n_domains_train: int
    n_domains_test: int
    below_floor: bool


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


def evaluate_fold(
    df: pd.DataFrame,
    split_result: SplitResult,
    y_pred: np.ndarray,
    *,
    model: str,
    feature_set: str,
    target_col: str = TARGET,
) -> tuple[pd.DataFrame, FoldMetrics]:
    """Compute per-row predictions and fold metrics from a single trained fold.

    Parameters
    ----------
    df:
        Full grouped DataFrame (``seed_with_groups.csv``) with all metadata
        columns.
    split_result:
        The :class:`~seed_vfa.splits.SplitResult` that produced this fold.
    y_pred:
        Predictions from ``pipeline.predict(X_test)``, aligned to
        ``df.iloc[split_result.test_idx]``.
    model:
        Model ID string (e.g. ``"ridge"``).
    feature_set:
        Feature set ID string (e.g. ``"FS2"``).
    target_col:
        Target column name. Defaults to :data:`~seed_vfa.data.TARGET`.

    Returns
    -------
    predictions_df:
        DataFrame with exactly :data:`PREDICTIONS_COLUMNS` columns and one row
        per test observation.
    fold_metrics:
        :class:`FoldMetrics` computed from the same y_true/y_pred arrays.

    Notes
    -----
    Both outputs are derived from the same y_true array in one pass.  The
    caller must save ``predictions_df`` before using ``fold_metrics`` to satisfy
    AGENTS.md §1 invariant 6: aggregate metrics are derived from saved
    predictions, not discarded alongside them.
    """
    test_rows = df.iloc[split_result.test_idx]
    train_rows = df.iloc[split_result.train_idx]

    y_true: np.ndarray = test_rows[target_col].to_numpy(dtype=float)

    predictions_df = pd.DataFrame(
        {
            ROW_ID: test_rows[ROW_ID].values,
            CONDITION_ID: test_rows[CONDITION_ID].values,
            REPLICATE_GROUP_ID: test_rows[REPLICATE_GROUP_ID].values,
            MEMBRANE_ID: test_rows[MEMBRANE_ID].values,
            FEED_TYPE: test_rows[FEED_TYPE].values,
            "protocol": split_result.protocol,
            "split_id": split_result.split_id,
            "fold": split_result.fold,
            "model": model,
            "feature_set": feature_set,
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_true - y_pred,
        }
    )

    fold_metrics = FoldMetrics(
        r2=float(r2_score(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        mae=float(mean_absolute_error(y_true, y_pred)),
        n_train=len(split_result.train_idx),
        n_test=len(split_result.test_idx),
        n_conditions_train=int(train_rows[CONDITION_ID].nunique()),
        n_conditions_test=int(test_rows[CONDITION_ID].nunique()),
        n_domains_train=int(train_rows[MEMBRANE_ID].nunique()),
        n_domains_test=int(test_rows[MEMBRANE_ID].nunique()),
    )

    return predictions_df, fold_metrics


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_metrics(
    fold_metrics_list: Sequence[FoldMetrics],
    *,
    protocol: str,
    model: str,
    feature_set: str,
    rmse_floor: float,
) -> AggregatedMetrics:
    """Aggregate per-fold metrics into a single :class:`AggregatedMetrics` record.

    Parameters
    ----------
    fold_metrics_list:
        One :class:`FoldMetrics` per fold, as returned by :func:`evaluate_fold`.
        Must not be empty.
    protocol:
        Protocol ID (e.g. ``"C"``).
    model:
        Model ID (e.g. ``"ridge"``).
    feature_set:
        Feature set ID (e.g. ``"FS2"``).
    rmse_floor:
        ``NoiseFloorResult.rmse_floor``; used to set ``below_floor``.

    Returns
    -------
    AggregatedMetrics
        Mean and population std (ddof=0) across all folds.  Integer count
        fields are rounded means.  ``below_floor`` is applied to
        ``RMSE_mean``.

    Raises
    ------
    ValueError
        If ``fold_metrics_list`` is empty.
    """
    if not fold_metrics_list:
        raise ValueError("fold_metrics_list must not be empty.")

    r2s = np.array([m.r2 for m in fold_metrics_list], dtype=float)
    rmses = np.array([m.rmse for m in fold_metrics_list], dtype=float)
    maes = np.array([m.mae for m in fold_metrics_list], dtype=float)
    rmse_mean = float(np.mean(rmses))

    def _imean(vals: list[int]) -> int:
        return int(round(float(np.mean(vals))))

    return AggregatedMetrics(
        protocol=protocol,
        model=model,
        feature_set=feature_set,
        R2_mean=float(np.mean(r2s)),
        R2_std=float(np.std(r2s, ddof=0)),
        RMSE_mean=rmse_mean,
        RMSE_std=float(np.std(rmses, ddof=0)),
        MAE_mean=float(np.mean(maes)),
        MAE_std=float(np.std(maes, ddof=0)),
        n_train=_imean([m.n_train for m in fold_metrics_list]),
        n_test=_imean([m.n_test for m in fold_metrics_list]),
        n_conditions_train=_imean([m.n_conditions_train for m in fold_metrics_list]),
        n_conditions_test=_imean([m.n_conditions_test for m in fold_metrics_list]),
        n_domains_train=_imean([m.n_domains_train for m in fold_metrics_list]),
        n_domains_test=_imean([m.n_domains_test for m in fold_metrics_list]),
        below_floor=flag_below_floor(rmse_mean, rmse_floor),
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def metrics_to_record(agg: AggregatedMetrics) -> dict[str, object]:
    """Convert :class:`AggregatedMetrics` to a flat dict for CSV serialisation.

    Key order matches :data:`METRICS_COLUMNS` exactly.
    """
    return {
        "protocol": agg.protocol,
        "model": agg.model,
        "feature_set": agg.feature_set,
        "R2_mean": agg.R2_mean,
        "R2_std": agg.R2_std,
        "RMSE_mean": agg.RMSE_mean,
        "RMSE_std": agg.RMSE_std,
        "MAE_mean": agg.MAE_mean,
        "MAE_std": agg.MAE_std,
        "n_train": agg.n_train,
        "n_test": agg.n_test,
        "n_conditions_train": agg.n_conditions_train,
        "n_conditions_test": agg.n_conditions_test,
        "n_domains_train": agg.n_domains_train,
        "n_domains_test": agg.n_domains_test,
        "below_floor": agg.below_floor,
    }
