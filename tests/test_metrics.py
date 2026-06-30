"""Tests for the evaluation and prediction-saving layer (AGENTS.md §9, §11).

Verifies:
- R², RMSE, MAE are computed correctly on known synthetic examples.
- residual == y_true - y_pred for every row.
- evaluate_fold returns a DataFrame with exactly PREDICTIONS_COLUMNS.
- evaluate_fold row count equals len(test_idx).
- n_train / n_test in FoldMetrics match the split indices.
- n_conditions_train / n_conditions_test are correct.
- n_domains_train / n_domains_test are correct.
- aggregate_metrics mean and std are correct on known FoldMetrics lists.
- below_floor is applied to RMSE_mean, not per-fold RMSE.
- metrics_to_record keys match METRICS_COLUMNS exactly.
- §11 mandatory: test_reproducible — two runs same seed → identical predictions
  and metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seed_vfa.config import load_config
from seed_vfa.data import TARGET
from seed_vfa.evaluation import (
    METRICS_COLUMNS,
    PREDICTIONS_COLUMNS,
    FoldMetrics,
    aggregate_metrics,
    evaluate_fold,
    metrics_to_record,
)
from seed_vfa.features import get_columns, select_features
from seed_vfa.groups import CONDITION_ID, MEMBRANE_ID
from seed_vfa.models import build_pipeline
from seed_vfa.splits import SplitResult, protocol_a


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
def one_split(df: pd.DataFrame, config) -> SplitResult:
    splits = protocol_a(df, test_size=config.splits.test_size, seed=config.seed)
    return splits[0]


@pytest.fixture(scope="module")
def ridge_fold(
    df: pd.DataFrame, one_split: SplitResult, config
) -> tuple[pd.DataFrame, FoldMetrics]:
    """Fit Ridge/FS2 on protocol-A fold; return (predictions_df, fold_metrics)."""
    fid = "FS2"
    pipe = build_pipeline("ridge", get_columns(fid), seed=config.seed, models_config=config.models)
    pipe.fit(select_features(fid, df.iloc[one_split.train_idx]),
             df.iloc[one_split.train_idx][TARGET].values)
    y_pred = pipe.predict(select_features(fid, df.iloc[one_split.test_idx]))
    return evaluate_fold(df, one_split, y_pred, model="ridge", feature_set=fid)


# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------


def test_predictions_columns_schema() -> None:
    expected = (
        "row_id", "condition_id", "replicate_group_id", "membrane_id", "feed_type",
        "protocol", "split_id", "fold", "model", "feature_set",
        "y_true", "y_pred", "residual",
    )
    assert PREDICTIONS_COLUMNS == expected


def test_metrics_columns_schema() -> None:
    expected = (
        "protocol", "model", "feature_set",
        "R2_mean", "R2_std", "RMSE_mean", "RMSE_std", "MAE_mean", "MAE_std",
        "n_train", "n_test",
        "n_conditions_train", "n_conditions_test",
        "n_domains_train", "n_domains_test",
        "below_floor",
    )
    assert METRICS_COLUMNS == expected


# ---------------------------------------------------------------------------
# Metric computation correctness (synthetic)
# ---------------------------------------------------------------------------


def _mini_df(y: np.ndarray) -> pd.DataFrame:
    n = len(y)
    return pd.DataFrame({
        "row_id": list(range(n)),
        "condition_id": list(range(n)),
        "replicate_group_id": list(range(n)),
        "membrane_id": [225] * n,
        "feed_type": ["simple"] * n,
        TARGET: y,
    })


def _mini_split(n: int, n_train: int) -> SplitResult:
    return SplitResult(
        protocol="A", split_id=0, fold=0,
        train_idx=np.arange(n_train),
        test_idx=np.arange(n_train, n),
    )


def test_r2_rmse_mae_perfect_predictions() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    df_mini = _mini_df(y)
    sr = _mini_split(n=4, n_train=2)
    y_pred = y[sr.test_idx]
    _, fm = evaluate_fold(df_mini, sr, y_pred=y_pred, model="dummy", feature_set="FS1")
    assert fm.r2 == pytest.approx(1.0)
    assert fm.rmse == pytest.approx(0.0, abs=1e-12)
    assert fm.mae == pytest.approx(0.0, abs=1e-12)


def test_r2_rmse_mae_known_values() -> None:
    # test rows y_true=[3,5], y_pred=[2,6]: errors=[-1,+1]
    # RMSE = sqrt((1+1)/2) = 1.0, MAE = 1.0
    y_all = np.array([1.0, 2.0, 3.0, 5.0])
    df_mini = _mini_df(y_all)
    sr = _mini_split(n=4, n_train=2)
    y_pred = np.array([2.0, 6.0])
    _, fm = evaluate_fold(df_mini, sr, y_pred=y_pred, model="dummy", feature_set="FS1")
    assert fm.rmse == pytest.approx(1.0, rel=1e-9)
    assert fm.mae == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# evaluate_fold DataFrame structure
# ---------------------------------------------------------------------------


def test_predictions_df_columns_exact(ridge_fold) -> None:
    preds_df, _ = ridge_fold
    assert tuple(preds_df.columns) == PREDICTIONS_COLUMNS


def test_predictions_df_row_count(one_split: SplitResult, ridge_fold) -> None:
    preds_df, _ = ridge_fold
    assert len(preds_df) == len(one_split.test_idx)


def test_residual_equals_y_true_minus_y_pred(ridge_fold) -> None:
    preds_df, _ = ridge_fold
    np.testing.assert_allclose(
        preds_df["residual"].values,
        preds_df["y_true"].values - preds_df["y_pred"].values,
    )


def test_n_train_n_test_match_split(one_split: SplitResult, ridge_fold) -> None:
    _, fm = ridge_fold
    assert fm.n_train == len(one_split.train_idx)
    assert fm.n_test == len(one_split.test_idx)


def test_n_conditions_test_correct(df: pd.DataFrame, one_split: SplitResult, ridge_fold) -> None:
    _, fm = ridge_fold
    expected = int(df.iloc[one_split.test_idx][CONDITION_ID].nunique())
    assert fm.n_conditions_test == expected


def test_n_conditions_train_correct(df: pd.DataFrame, one_split: SplitResult, ridge_fold) -> None:
    _, fm = ridge_fold
    expected = int(df.iloc[one_split.train_idx][CONDITION_ID].nunique())
    assert fm.n_conditions_train == expected


def test_n_domains_test_correct(df: pd.DataFrame, one_split: SplitResult, ridge_fold) -> None:
    _, fm = ridge_fold
    expected = int(df.iloc[one_split.test_idx][MEMBRANE_ID].nunique())
    assert fm.n_domains_test == expected


def test_n_domains_train_correct(df: pd.DataFrame, one_split: SplitResult, ridge_fold) -> None:
    _, fm = ridge_fold
    expected = int(df.iloc[one_split.train_idx][MEMBRANE_ID].nunique())
    assert fm.n_domains_train == expected


def test_predictions_no_nan(ridge_fold) -> None:
    preds_df, _ = ridge_fold
    assert not preds_df.isnull().any().any()


def test_predictions_row_ids_subset_of_dataset(df: pd.DataFrame, ridge_fold) -> None:
    preds_df, _ = ridge_fold
    valid = set(df["row_id"].values)
    assert set(preds_df["row_id"].values).issubset(valid)


def test_protocol_label_propagated(one_split: SplitResult, ridge_fold) -> None:
    preds_df, _ = ridge_fold
    assert (preds_df["protocol"] == one_split.protocol).all()


def test_model_label_propagated(ridge_fold) -> None:
    preds_df, _ = ridge_fold
    assert (preds_df["model"] == "ridge").all()


def test_feature_set_label_propagated(ridge_fold) -> None:
    preds_df, _ = ridge_fold
    assert (preds_df["feature_set"] == "FS2").all()


# ---------------------------------------------------------------------------
# aggregate_metrics correctness
# ---------------------------------------------------------------------------


def _fm(r2: float, rmse: float, mae: float) -> FoldMetrics:
    return FoldMetrics(r2=r2, rmse=rmse, mae=mae,
                       n_train=59, n_test=20,
                       n_conditions_train=36, n_conditions_test=10,
                       n_domains_train=4, n_domains_test=4)


def test_aggregate_single_fold_std_is_zero() -> None:
    agg = aggregate_metrics([_fm(0.8, 0.3, 0.25)],
                            protocol="A", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    assert agg.R2_std == pytest.approx(0.0, abs=1e-12)
    assert agg.RMSE_std == pytest.approx(0.0, abs=1e-12)


def test_aggregate_single_fold_mean_equals_value() -> None:
    agg = aggregate_metrics([_fm(0.8, 0.3, 0.25)],
                            protocol="A", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    assert agg.R2_mean == pytest.approx(0.8)
    assert agg.RMSE_mean == pytest.approx(0.3)
    assert agg.MAE_mean == pytest.approx(0.25)


def test_aggregate_two_folds_mean() -> None:
    agg = aggregate_metrics([_fm(0.6, 0.4, 0.3), _fm(0.8, 0.2, 0.1)],
                            protocol="C", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    assert agg.R2_mean == pytest.approx(0.7, rel=1e-9)
    assert agg.RMSE_mean == pytest.approx(0.3, rel=1e-9)
    assert agg.MAE_mean == pytest.approx(0.2, rel=1e-9)


def test_aggregate_two_folds_population_std() -> None:
    # population std of [0.6, 0.8] = sqrt(((0.1)^2 + (0.1)^2) / 2) = 0.1
    agg = aggregate_metrics([_fm(0.6, 0.4, 0.3), _fm(0.8, 0.2, 0.1)],
                            protocol="C", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    assert agg.R2_std == pytest.approx(0.1, rel=1e-9)
    assert agg.RMSE_std == pytest.approx(0.1, rel=1e-9)


def test_aggregate_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate_metrics([], protocol="A", model="dummy", feature_set="FS1",
                          rmse_floor=0.23)


def test_aggregate_labels_propagate() -> None:
    agg = aggregate_metrics([_fm(0.5, 0.35, 0.2)],
                            protocol="D", model="catboost", feature_set="FS3",
                            rmse_floor=0.23)
    assert agg.protocol == "D"
    assert agg.model == "catboost"
    assert agg.feature_set == "FS3"


def test_n_train_is_rounded_integer_mean() -> None:
    fm1 = FoldMetrics(r2=0.8, rmse=0.3, mae=0.2,
                      n_train=58, n_test=21,
                      n_conditions_train=35, n_conditions_test=11,
                      n_domains_train=4, n_domains_test=4)
    fm2 = FoldMetrics(r2=0.8, rmse=0.3, mae=0.2,
                      n_train=62, n_test=17,
                      n_conditions_train=35, n_conditions_test=11,
                      n_domains_train=4, n_domains_test=4)
    agg = aggregate_metrics([fm1, fm2], protocol="C", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    assert agg.n_train == 60  # mean([58, 62]) = 60.0 → int 60
    assert isinstance(agg.n_train, int)
    assert isinstance(agg.n_test, int)


# ---------------------------------------------------------------------------
# below_floor flag
# ---------------------------------------------------------------------------


def test_below_floor_applied_to_rmse_mean() -> None:
    # fold RMSEs [0.20, 0.24] → mean = 0.22 < floor=0.23 → below_floor=True
    agg = aggregate_metrics([_fm(0.9, 0.20, 0.15), _fm(0.9, 0.24, 0.15)],
                            protocol="A", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    assert agg.RMSE_mean == pytest.approx(0.22, rel=1e-9)
    assert agg.below_floor is True


def test_below_floor_false_when_rmse_above_floor() -> None:
    agg = aggregate_metrics([_fm(0.5, 0.30, 0.25)],
                            protocol="A", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    assert agg.below_floor is False


def test_below_floor_strict_not_equal() -> None:
    # RMSE_mean == rmse_floor is NOT below the floor (strict <)
    agg = aggregate_metrics([_fm(0.9, 0.23, 0.15)],
                            protocol="A", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    assert agg.below_floor is False


# ---------------------------------------------------------------------------
# metrics_to_record serialisation
# ---------------------------------------------------------------------------


def test_metrics_to_record_keys_match_schema() -> None:
    agg = aggregate_metrics([_fm(0.7, 0.30, 0.20)],
                            protocol="A", model="ridge", feature_set="FS2",
                            rmse_floor=0.23)
    record = metrics_to_record(agg)
    assert tuple(record.keys()) == METRICS_COLUMNS


def test_metrics_to_record_values_round_trip() -> None:
    agg = aggregate_metrics([_fm(0.8, 0.25, 0.15)],
                            protocol="B", model="catboost", feature_set="FS3",
                            rmse_floor=0.23)
    record = metrics_to_record(agg)
    assert record["protocol"] == "B"
    assert record["model"] == "catboost"
    assert record["feature_set"] == "FS3"
    assert record["R2_mean"] == pytest.approx(0.8)
    assert record["RMSE_mean"] == pytest.approx(0.25)
    assert isinstance(record["below_floor"], bool)


# ---------------------------------------------------------------------------
# §11 mandatory: test_reproducible
# ---------------------------------------------------------------------------


def test_reproducible(df: pd.DataFrame, config) -> None:
    """§11 mandatory — two runs with the same seed produce bitwise-identical results.

    Two independent pipelines built with the same seed, fit on the same
    protocol-A fold, must yield identical predictions DataFrames and
    FoldMetrics objects.  Verifies that the evaluation pipeline is
    deterministic and that config.seed controls all stochastic elements
    (AGENTS.md §1 invariant 5).
    """
    fid = "FS2"
    cols = get_columns(fid)
    split = protocol_a(df, test_size=config.splits.test_size, seed=config.seed)[0]
    X_train = select_features(fid, df.iloc[split.train_idx])
    y_train = df.iloc[split.train_idx][TARGET].values
    X_test = select_features(fid, df.iloc[split.test_idx])

    def _run() -> tuple[pd.DataFrame, FoldMetrics]:
        pipe = build_pipeline("ridge", cols, seed=config.seed, models_config=config.models)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        return evaluate_fold(df, split, y_pred, model="ridge", feature_set=fid)

    preds1, metrics1 = _run()
    preds2, metrics2 = _run()

    pd.testing.assert_frame_equal(preds1, preds2, check_exact=True)
    assert metrics1 == metrics2, (
        f"FoldMetrics differ between runs:\n  run 1: {metrics1}\n  run 2: {metrics2}"
    )
