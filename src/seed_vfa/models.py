"""Diagnostic model factory for the Seed/VFA reanalysis (AGENTS.md §7).

Purpose
-------
Build sklearn-compatible ``Pipeline`` objects for the six model roles defined
in §7.  Each pipeline bundles preprocessing (scaling and optional categorical
encoding) *inside* the estimator, so no transformer can be fitted before the
training fold is defined (AGENTS.md §1, invariant 3).

These models are diagnostic instruments to demonstrate the effect of
validation protocol, not competition entries.  Hyperparameters are modest
fixed references from ``config.yaml``, not tuned results.

Models (AGENTS.md §7)
----------------------
=================  ============  ============================================
ID                 Estimator     Role
=================  ============  ============================================
``dummy``          DummyRegressor(strategy="mean")  Trivial baseline (≈0.72
                                 RMSE line on every figure).
``ridge``          Ridge(alpha)  Collapses under extrapolation (full feature
                                 set) or degrades gracefully (parsimonious);
                                 the feature set selects the scientific role.
``catboost``       CatBoostRegressor  Source-paper reference; NOT the
                                 protagonist.
``random_forest``  RandomForest  Non-linear control; does not extrapolate
                                 beyond seen range.
=================  ============  ============================================

Preprocessing inside every pipeline
-------------------------------------
All numeric features are scaled with ``StandardScaler`` (fit inside the
training fold).  The categorical ``feed_type`` column, when present, is
encoded with ``OrdinalEncoder(categories=[['simple','complex']])``
(simple → 0.0, complex → 1.0) *before* the scaler, inside the same
``ColumnTransformer``.  No transformer is ever fitted on the full dataset.

``build_pipeline`` detects the presence of ``feed_type`` in ``feature_columns``
and selects the appropriate preprocessor automatically.  The resulting
pipeline always exposes the ``prep`` and ``model`` named steps.

Determinism
-----------
Every stochastic estimator (``catboost``, ``random_forest``) takes its seed
from ``config.seed`` (AGENTS.md §1, invariant 5).  Re-running with the same
seed reproduces identical predictions bit-for-bit.

Usage
-----
::

    from seed_vfa.config import load_config
    from seed_vfa.features import get_columns, select_features
    from seed_vfa.models import build_pipeline

    config = load_config()
    df = ...
    X = select_features("FS2", df)
    y = df["Cret total VFAs"].values
    pipe = build_pipeline("ridge", get_columns("FS2"), seed=config.seed, models_config=config.models)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
"""

from __future__ import annotations

from typing import Literal

from catboost import CatBoostRegressor
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import ModelsConfig

# ---------------------------------------------------------------------------
# Model identifiers
# ---------------------------------------------------------------------------

ModelID = Literal["dummy", "ridge", "catboost", "random_forest"]

ALL_MODEL_IDS: tuple[ModelID, ...] = (
    "dummy",
    "ridge",
    "catboost",
    "random_forest",
)

# ---------------------------------------------------------------------------
# Categorical column handling
# ---------------------------------------------------------------------------

# The only categorical feature that can appear in a feature set (AGENTS.md §8).
_FEED_TYPE_COL: str = "feed_type"

# Explicit category order makes encoding deterministic and documented.
# simple (no added ions) → 0.0 ; complex (ions present) → 1.0.
_FEED_TYPE_CATEGORIES: list[list[str]] = [["simple", "complex"]]


class ModelError(ValueError):
    """Raised when an unknown model ID is requested."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_preprocessor(feature_columns: list[str] | tuple[str, ...]) -> ColumnTransformer | StandardScaler:
    """Build the preprocessing step appropriate for the given feature columns.

    If ``feed_type`` is absent: return a plain ``StandardScaler`` (all columns
    are numeric).

    If ``feed_type`` is present: return a ``ColumnTransformer`` that scales the
    numeric columns and ordinally encodes ``feed_type``.  ``remainder='drop'``
    ensures no column is silently passed through or duplicated.

    The returned transformer is *not* fitted; it will be fitted inside the
    training fold when ``Pipeline.fit`` is called.
    """
    cols = list(feature_columns)
    if _FEED_TYPE_COL not in cols:
        return StandardScaler()

    num_cols = [c for c in cols if c != _FEED_TYPE_COL]
    cat_cols = [_FEED_TYPE_COL]

    transformers = []
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))
    transformers.append(
        (
            "cat",
            OrdinalEncoder(
                categories=_FEED_TYPE_CATEGORIES,
                # Unseen categories at predict time raise an error rather than
                # silently producing NaN (fail loudly, rules §10).
                handle_unknown="error",
            ),
            cat_cols,
        )
    )
    return ColumnTransformer(transformers, remainder="drop")


def _require_model_id(model_id: str) -> None:
    if model_id not in ALL_MODEL_IDS:
        raise ModelError(
            f"Unknown model '{model_id}'. Valid IDs: {list(ALL_MODEL_IDS)}."
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pipeline(
    model_id: str,
    feature_columns: list[str] | tuple[str, ...],
    *,
    seed: int,
    models_config: ModelsConfig,
) -> Pipeline:
    """Build a preprocessing + estimator ``Pipeline`` for ``model_id``.

    The pipeline always contains exactly two named steps:

    * ``"prep"`` — a ``StandardScaler`` (all-numeric) or ``ColumnTransformer``
      (numeric + ``feed_type``), *not* pre-fitted.
    * ``"model"`` — the sklearn estimator.

    Parameters
    ----------
    model_id:
        One of ``ALL_MODEL_IDS``.
    feature_columns:
        The columns that will be present in ``X`` at fit/predict time
        (i.e. ``get_columns(feature_set_id)``).  Used to build the correct
        preprocessor.
    seed:
        Global RNG seed from ``config.seed``.  Used only by stochastic models
        (CatBoost, RandomForest).  DummyRegressor and Ridge ignore it.
    models_config:
        Hyperparameter values from ``config.models``.

    Returns
    -------
    sklearn.pipeline.Pipeline
        An unfitted pipeline ready for use inside a CV fold.

    Raises
    ------
    ModelError
        If ``model_id`` is not in ``ALL_MODEL_IDS``.
    """
    _require_model_id(model_id)
    prep = _make_preprocessor(feature_columns)
    estimator = _build_estimator(model_id, seed=seed, cfg=models_config)
    return Pipeline([("prep", prep), ("model", estimator)])


def _build_estimator(
    model_id: str,
    *,
    seed: int,
    cfg: ModelsConfig,
) -> DummyRegressor | Ridge | CatBoostRegressor | RandomForestRegressor:
    """Return the bare estimator for ``model_id`` (no preprocessing wrapper)."""
    if model_id == "dummy":
        return DummyRegressor(strategy="mean")

    if model_id == "ridge":
        return Ridge(alpha=cfg.ridge_alpha)

    if model_id == "catboost":
        return CatBoostRegressor(
            iterations=cfg.catboost_iterations,
            depth=cfg.catboost_depth,
            learning_rate=cfg.catboost_learning_rate,
            loss_function="RMSE",
            random_seed=seed,
            # verbose=0 suppresses per-iteration output in scripts.
            verbose=0,
            # thread_count=1 forces single-threaded execution for full
            # determinism; multi-threaded CatBoost can produce platform-
            # dependent floating-point rounding differences.
            thread_count=1,
            # Allow sklearn to set_params on this estimator (needed for CV).
            allow_writing_files=False,
        )

    if model_id == "random_forest":
        return RandomForestRegressor(
            n_estimators=cfg.random_forest_n_estimators,
            # n_jobs=1 ensures bit-for-bit determinism; parallel execution can
            # reorder floating-point accumulation across platforms.
            n_jobs=1,
            random_state=seed,
        )

    # Unreachable after _require_model_id, but satisfies type checkers.
    raise ModelError(f"Unhandled model_id: {model_id!r}")  # pragma: no cover


def get_estimator_from_pipeline(pipeline: Pipeline) -> object:
    """Return the estimator step from a pipeline built by :func:`build_pipeline`.

    Convenience accessor for tests and feature-importance extraction that need
    the bare estimator (e.g. ``coef_``, ``feature_importances_``).
    """
    return pipeline.named_steps["model"]


def get_preprocessor_from_pipeline(pipeline: Pipeline) -> ColumnTransformer | StandardScaler:
    """Return the preprocessing step from a pipeline built by :func:`build_pipeline`."""
    return pipeline.named_steps["prep"]  # type: ignore[return-value]
