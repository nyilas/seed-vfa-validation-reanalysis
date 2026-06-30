"""Tests for the model factory layer (AGENTS.md §7).

Verifies:
- Every model ID produces an sklearn-compatible Pipeline.
- sklearn.base.clone() succeeds on every pipeline (required for CV).
- The pipeline has exactly two named steps: 'prep' and 'model'.
- The 'prep' step contains a StandardScaler (possibly inside a ColumnTransformer).
- The 'prep' step is NOT fitted before pipeline.fit() is called.
- pipeline.predict() raises NotFittedError before fit.
- All model + feature-set combinations fit and predict without error.
- Stochastic models (catboost, random_forest) are deterministic with a fixed seed.
- Two pipelines built with different seeds produce different predictions.
- DummyRegressor always predicts the training mean regardless of features.
- Unknown model IDs raise ModelError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from seed_vfa.config import load_config
from seed_vfa.features import ALL_FEATURE_SET_IDS, get_columns, select_features
from seed_vfa.models import (
    ALL_MODEL_IDS,
    ModelError,
    build_pipeline,
    get_estimator_from_pipeline,
    get_preprocessor_from_pipeline,
)


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
def train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Small deterministic 75/25 split for fit/predict smoke tests."""
    n = len(df)
    split = int(n * 0.75)
    y = df["Cret total VFAs"].values
    return df.iloc[:split], df.iloc[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

def test_all_model_ids_known(config) -> None:
    for mid in ALL_MODEL_IDS:
        pipe = build_pipeline(mid, get_columns("FS2"), seed=config.seed, models_config=config.models)
        assert isinstance(pipe, Pipeline)


# ---------------------------------------------------------------------------
# Pipeline structure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mid", ALL_MODEL_IDS)
def test_pipeline_has_prep_and_model_steps(mid: str, config) -> None:
    pipe = build_pipeline(mid, get_columns("FS1"), seed=config.seed, models_config=config.models)
    assert "prep" in pipe.named_steps, f"{mid}: missing 'prep' step."
    assert "model" in pipe.named_steps, f"{mid}: missing 'model' step."
    assert len(pipe.steps) == 2, f"{mid}: expected 2 steps, got {len(pipe.steps)}."


@pytest.mark.parametrize("mid", ALL_MODEL_IDS)
def test_scaler_is_inside_pipeline(mid: str, config) -> None:
    """StandardScaler must be a component of the 'prep' step, not a standalone object."""
    pipe = build_pipeline(mid, get_columns("FS1"), seed=config.seed, models_config=config.models)
    prep = get_preprocessor_from_pipeline(pipe)
    # For all-numeric feature sets, prep IS a StandardScaler.
    # For feature sets with feed_type, prep is a ColumnTransformer that contains one.
    has_scaler = isinstance(prep, StandardScaler) or (
        isinstance(prep, ColumnTransformer)
        and any(isinstance(t, StandardScaler) for _, t, _ in prep.transformers)
    )
    assert has_scaler, (
        f"{mid}: 'prep' step ({type(prep).__name__}) contains no StandardScaler."
    )


@pytest.mark.parametrize("mid", ALL_MODEL_IDS)
def test_scaler_inside_pipeline_with_categorical_feature_set(mid: str, config) -> None:
    """When feed_type is in the feature set, a ColumnTransformer must wrap the scaler."""
    pipe = build_pipeline(mid, get_columns("FS3"), seed=config.seed, models_config=config.models)
    prep = get_preprocessor_from_pipeline(pipe)
    assert isinstance(prep, ColumnTransformer), (
        f"{mid} with FS3 (has feed_type): expected ColumnTransformer, "
        f"got {type(prep).__name__}."
    )
    scaler_names = [name for name, t, _ in prep.transformers if isinstance(t, StandardScaler)]
    assert len(scaler_names) > 0, (
        f"{mid} with FS3: ColumnTransformer contains no StandardScaler."
    )


@pytest.mark.parametrize("mid", ALL_MODEL_IDS)
def test_prep_step_not_fitted_before_fit(mid: str, config) -> None:
    """The preprocessing step must not be fitted at pipeline construction time."""
    pipe = build_pipeline(mid, get_columns("FS1"), seed=config.seed, models_config=config.models)
    prep = get_preprocessor_from_pipeline(pipe)
    # sklearn fitted transformers expose n_features_in_; unfitted ones do not.
    assert not hasattr(prep, "n_features_in_"), (
        f"{mid}: 'prep' step already has n_features_in_ — it was fitted before "
        "the training fold, violating AGENTS.md §1 invariant 3."
    )


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mid", ALL_MODEL_IDS)
def test_clone_succeeds(mid: str, config) -> None:
    """clone() must work for every pipeline (required by sklearn CV internals)."""
    pipe = build_pipeline(mid, get_columns("FS2"), seed=config.seed, models_config=config.models)
    cloned = clone(pipe)
    assert isinstance(cloned, Pipeline)


@pytest.mark.parametrize("mid", ALL_MODEL_IDS)
def test_predict_before_fit_raises(mid: str, config) -> None:
    pipe = build_pipeline(mid, get_columns("FS2"), seed=config.seed, models_config=config.models)
    X_dummy = pd.DataFrame(
        {"Zeta potential [mV]": [1.0], "pH": [7.0], "Pressure [bar]": [5.0],
         "PWP [LMH/bar]": [10.0], "Monovalent anion feed [mmol/L]": [0.0]}
    )
    with pytest.raises(NotFittedError):
        pipe.predict(X_dummy)


# ---------------------------------------------------------------------------
# Fit / predict across model × feature-set combinations
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mid", ALL_MODEL_IDS)
@pytest.mark.parametrize("fid", ALL_FEATURE_SET_IDS)
def test_fit_predict_all_combinations(
    mid: str, fid: str, df: pd.DataFrame, train_test, config
) -> None:
    X_tr, X_te, y_tr, y_te = train_test
    X_tr_f = select_features(fid, X_tr)
    X_te_f = select_features(fid, X_te)
    pipe = build_pipeline(mid, get_columns(fid), seed=config.seed, models_config=config.models)
    pipe.fit(X_tr_f, y_tr)
    preds = pipe.predict(X_te_f)
    assert preds.shape == y_te.shape, (
        f"{mid} + {fid}: prediction shape {preds.shape} != target shape {y_te.shape}."
    )
    assert not np.any(np.isnan(preds)), f"{mid} + {fid}: predictions contain NaN."
    assert not np.any(np.isinf(preds)), f"{mid} + {fid}: predictions contain Inf."


# ---------------------------------------------------------------------------
# DummyRegressor behaviour
# ---------------------------------------------------------------------------

def test_dummy_predicts_training_mean(
    df: pd.DataFrame, train_test, config
) -> None:
    """DummyRegressor must always predict the training mean."""
    X_tr, X_te, y_tr, y_te = train_test
    X_tr_f = select_features("FS1", X_tr)
    X_te_f = select_features("FS1", X_te)
    pipe = build_pipeline("dummy", get_columns("FS1"), seed=config.seed, models_config=config.models)
    pipe.fit(X_tr_f, y_tr)
    preds = pipe.predict(X_te_f)
    expected = float(np.mean(y_tr))
    assert np.allclose(preds, expected), (
        f"DummyRegressor predicted {preds[:3]} but training mean is {expected:.4f}."
    )


def test_dummy_predictions_independent_of_features(
    df: pd.DataFrame, train_test, config
) -> None:
    """Predictions from Dummy must be identical regardless of feature set."""
    X_tr, X_te, y_tr, y_te = train_test
    preds: list[np.ndarray] = []
    for fid in ["FS1", "FS2", "FS3", "FS5"]:
        pipe = build_pipeline("dummy", get_columns(fid), seed=config.seed, models_config=config.models)
        pipe.fit(select_features(fid, X_tr), y_tr)
        preds.append(pipe.predict(select_features(fid, X_te)))
    for p in preds[1:]:
        assert np.allclose(preds[0], p), "DummyRegressor gave different predictions for different feature sets."


# ---------------------------------------------------------------------------
# Stochastic model determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mid", ["catboost", "random_forest"])
def test_same_seed_identical_predictions(
    mid: str, df: pd.DataFrame, train_test, config
) -> None:
    """Two pipelines built with the same seed must produce identical predictions."""
    X_tr, X_te, y_tr, y_te = train_test
    fid = "FS2"
    X_tr_f = select_features(fid, X_tr)
    X_te_f = select_features(fid, X_te)
    cols = get_columns(fid)

    pipe1 = build_pipeline(mid, cols, seed=config.seed, models_config=config.models)
    pipe2 = build_pipeline(mid, cols, seed=config.seed, models_config=config.models)
    pipe1.fit(X_tr_f, y_tr)
    pipe2.fit(X_tr_f, y_tr)

    assert np.array_equal(pipe1.predict(X_te_f), pipe2.predict(X_te_f)), (
        f"{mid}: two pipelines with seed={config.seed} produced different predictions."
    )


@pytest.mark.parametrize("mid", ["catboost", "random_forest"])
def test_different_seeds_different_predictions(
    mid: str, df: pd.DataFrame, train_test, config
) -> None:
    """Different seeds should produce different predictions (verifies seed is used)."""
    X_tr, X_te, y_tr, y_te = train_test
    fid = "FS2"
    X_tr_f = select_features(fid, X_tr)
    X_te_f = select_features(fid, X_te)
    cols = get_columns(fid)

    pipe1 = build_pipeline(mid, cols, seed=0, models_config=config.models)
    pipe2 = build_pipeline(mid, cols, seed=99999, models_config=config.models)
    pipe1.fit(X_tr_f, y_tr)
    pipe2.fit(X_tr_f, y_tr)

    assert not np.array_equal(pipe1.predict(X_te_f), pipe2.predict(X_te_f)), (
        f"{mid}: pipelines with seeds 0 and 99999 produced identical predictions — "
        "the seed is likely not being applied."
    )


# ---------------------------------------------------------------------------
# Estimator access helpers
# ---------------------------------------------------------------------------

def test_get_estimator_from_pipeline(config) -> None:
    from sklearn.linear_model import Ridge
    pipe = build_pipeline("ridge", get_columns("FS2"), seed=config.seed, models_config=config.models)
    est = get_estimator_from_pipeline(pipe)
    assert isinstance(est, Ridge)


def test_get_preprocessor_from_pipeline(config) -> None:
    pipe = build_pipeline("ridge", get_columns("FS1"), seed=config.seed, models_config=config.models)
    prep = get_preprocessor_from_pipeline(pipe)
    assert isinstance(prep, StandardScaler)


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------

def test_unknown_model_id_raises(config) -> None:
    with pytest.raises(ModelError, match="Unknown model"):
        build_pipeline("gpr", get_columns("FS2"), seed=config.seed, models_config=config.models)
