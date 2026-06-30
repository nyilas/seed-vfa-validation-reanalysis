"""Leakage-invariant acceptance tests (AGENTS.md §11, test_no_leakage).

Three §11 mandatory tests:

``test_replicates_never_split``
    For grouped protocols C and D (and optional E), no ``replicate_group_id``
    appears in both train and test within any fold. This is AGENTS.md §1
    invariant 2. Failure here means the grouped protocol silently tested a
    model on conditions it saw during training.

``test_no_global_fit``
    The split layer must not transform feature values. Verified structurally:
    (a) SplitResult objects carry only integer index arrays, not feature
    matrices; (b) the grouped dataset is not mutated by any split-generation
    call; (c) split indices are plain integers, not any transformed quantity.
    Note: when models.py is implemented, this test should be extended to also
    assert that sklearn scalers/selectors are fit inside the training fold only
    (via Pipeline), not on the full dataset before splitting.

``test_no_synthetic_in_test``
    Every test index produced by every protocol must be a valid positional
    index into the grouped dataset (0 … n−1). A synthetic row would carry an
    index beyond this range.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seed_vfa import splits
from seed_vfa.config import load_config
from seed_vfa.groups import REPLICATE_GROUP_ID


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    config = load_config()
    path = config.paths.seed_with_groups
    if not path.is_file():
        pytest.skip(f"{path} not found — run run_data_audit.py first.")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def config():
    return load_config()


# ---------------------------------------------------------------------------
# §11: test_replicates_never_split
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("protocol_label,gen_fn_name", [
    ("C", "protocol_c"),
    ("D", "protocol_d"),
    ("E", "protocol_e"),
])
def test_replicates_never_split(
    df: pd.DataFrame, config, protocol_label: str, gen_fn_name: str
) -> None:
    """AGENTS.md §11: no replicate_group_id spans train/test in grouped protocols."""
    if protocol_label == "C":
        all_splits = splits.protocol_c(df, n_splits=config.splits.gkf_n_splits)
    elif protocol_label == "D":
        all_splits = splits.protocol_d(df)
    else:
        all_splits = splits.protocol_e(df)

    for sr in all_splits:
        train_rids = set(df.iloc[sr.train_idx][REPLICATE_GROUP_ID])
        test_rids = set(df.iloc[sr.test_idx][REPLICATE_GROUP_ID])
        crossing = train_rids & test_rids
        assert len(crossing) == 0, (
            f"Protocol {protocol_label} fold {sr.fold}: "
            f"replicate_group_id(s) {crossing} appear in both train and test. "
            "This violates AGENTS.md §1 invariant 2."
        )


# ---------------------------------------------------------------------------
# §11: test_no_global_fit
# ---------------------------------------------------------------------------

def test_no_global_fit(df: pd.DataFrame, config) -> None:
    """Split generators must not modify feature values or fit any transformer.

    Checks three structural properties:
    1. SplitResult.train_idx and test_idx are integer-dtype arrays (indices,
       not transformed feature values).
    2. The DataFrame is not mutated by any split-generation call.
    3. All indices are within the valid positional range [0, len(df) - 1].

    When models.py is implemented, extend this test to verify that all sklearn
    preprocessing/selection estimators are fitted inside Pipeline.fit() on the
    training fold only, not on the full dataset (AGENTS.md §1 invariant 3).
    """
    # Snapshot a subset of raw feature values before generating any split.
    feature_snapshot = df.iloc[:, :5].values.copy()

    all_generated: list[splits.SplitResult] = [
        *splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size),
        *splits.protocol_b(df, n_splits=3, test_size=config.splits.test_size, seed=config.seed),
        *splits.protocol_c(df, n_splits=config.splits.gkf_n_splits),
        *splits.protocol_d(df),
        *splits.protocol_e(df),
    ]

    # 1. DataFrame feature values must be unchanged.
    assert np.array_equal(df.iloc[:, :5].values, feature_snapshot), (
        "Feature values were modified by split generation — global fit suspected."
    )

    n = len(df)
    for sr in all_generated:
        # 2. Index arrays must have integer dtype.
        assert np.issubdtype(sr.train_idx.dtype, np.integer), (
            f"Protocol {sr.protocol} train_idx has non-integer dtype "
            f"{sr.train_idx.dtype}. SplitResult must carry raw indices only."
        )
        assert np.issubdtype(sr.test_idx.dtype, np.integer), (
            f"Protocol {sr.protocol} test_idx has non-integer dtype "
            f"{sr.test_idx.dtype}."
        )

        # 3. All indices within valid range.
        assert sr.train_idx.min() >= 0 and sr.train_idx.max() < n, (
            f"Protocol {sr.protocol}: train_idx out of range [0, {n-1}]."
        )
        assert sr.test_idx.min() >= 0 and sr.test_idx.max() < n, (
            f"Protocol {sr.protocol}: test_idx out of range [0, {n-1}]."
        )


# ---------------------------------------------------------------------------
# §11: test_no_synthetic_in_test
# ---------------------------------------------------------------------------

def test_no_synthetic_in_test(df: pd.DataFrame, config) -> None:
    """All test indices must map to real rows in the grouped dataset (§1.4).

    A synthetic row (which is out of scope for v1) would require appending rows
    beyond index n−1. This test guards against that by asserting every test
    index is a valid positional index into df.
    """
    n = len(df)
    all_splits_list: list[splits.SplitResult] = [
        *splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size),
        *splits.protocol_b(df, n_splits=5, test_size=config.splits.test_size, seed=config.seed),
        *splits.protocol_c(df, n_splits=config.splits.gkf_n_splits),
        *splits.protocol_d(df),
        *splits.protocol_e(df),
    ]
    for sr in all_splits_list:
        invalid = sr.test_idx[(sr.test_idx < 0) | (sr.test_idx >= n)]
        assert len(invalid) == 0, (
            f"Protocol {sr.protocol} fold {sr.fold}: "
            f"test_idx contains {len(invalid)} out-of-range value(s): {invalid}. "
            "Only real dataset rows (0 … n−1) may appear in the test set."
        )


# ---------------------------------------------------------------------------
# Sanity: random protocols DO split replicates (object of study, not a bug)
# ---------------------------------------------------------------------------

def test_random_protocols_do_split_replicates(df: pd.DataFrame, config) -> None:
    """Verify that random protocols produce the expected replicate leakage.

    This is not a failure condition — it is *the* condition that Protocols A/B
    are designed to demonstrate. If this test fails, the whole scientific
    argument about protocol-induced leakage collapses.
    """
    result_a = splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size)
    train_rids = set(df.iloc[result_a[0].train_idx][REPLICATE_GROUP_ID])
    test_rids = set(df.iloc[result_a[0].test_idx][REPLICATE_GROUP_ID])
    crossing = train_rids & test_rids
    assert len(crossing) > 0, (
        "Protocol A (random) produced zero replicate overlap. "
        "With 33 replicate pairs and 25% test fraction this should be impossible. "
        "Check the dataset structure."
    )
