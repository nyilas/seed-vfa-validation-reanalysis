"""Tests for the split-generator layer (AGENTS.md §6).

Verifies:
- Each protocol produces the expected number of splits / folds.
- Every split is a valid partition: train ∪ test = all rows, train ∩ test = ∅.
- Approximate test-set size for protocols A and B.
- Grouped protocols (C, D, E) produce zero condition/replicate overlap.
- Random protocols (A, B) do produce replicate overlap (structural expectation).
- Protocol D yields exactly 4 folds, one per membrane.
- Protocol E yields exactly 2 folds, one per feed regime.
- Leakage-audit table has correct columns, row count, and boolean flags.
- Determinism: same seed → identical split index arrays.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seed_vfa import splits
from seed_vfa.config import load_config
from seed_vfa.groups import (
    CONDITION_ID,
    FEED_TYPE,
    MEMBRANE_ID,
    REPLICATE_GROUP_ID,
)


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
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_partition(sr: splits.SplitResult, n: int) -> None:
    """Assert train and test form a valid, non-overlapping cover of all n rows."""
    all_idx = np.sort(np.concatenate([sr.train_idx, sr.test_idx]))
    assert np.array_equal(all_idx, np.arange(n)), (
        f"Protocol {sr.protocol} fold {sr.fold}: train + test do not cover all rows."
    )
    assert len(np.intersect1d(sr.train_idx, sr.test_idx)) == 0, (
        f"Protocol {sr.protocol} fold {sr.fold}: train and test overlap."
    )


def _replicate_overlap(df: pd.DataFrame, sr: splits.SplitResult) -> int:
    tr_rids = set(df.iloc[sr.train_idx][REPLICATE_GROUP_ID])
    te_rids = set(df.iloc[sr.test_idx][REPLICATE_GROUP_ID])
    return len(tr_rids & te_rids)


def _condition_overlap(df: pd.DataFrame, sr: splits.SplitResult) -> int:
    tr = set(df.iloc[sr.train_idx][CONDITION_ID])
    te = set(df.iloc[sr.test_idx][CONDITION_ID])
    return len(tr & te)


# ---------------------------------------------------------------------------
# Protocol A
# ---------------------------------------------------------------------------

def test_protocol_a_single_split(df: pd.DataFrame, config) -> None:
    result = splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size)
    assert len(result) == 1
    sr = result[0]
    assert sr.protocol == "A"
    assert sr.split_id == 0
    assert sr.fold == 0


def test_protocol_a_valid_partition(df: pd.DataFrame, config) -> None:
    result = splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size)
    _assert_valid_partition(result[0], len(df))


def test_protocol_a_approximate_test_size(df: pd.DataFrame, config) -> None:
    result = splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size)
    frac = len(result[0].test_idx) / len(df)
    assert abs(frac - config.splits.test_size) < 0.05, (
        f"Protocol A test fraction {frac:.3f} deviates from {config.splits.test_size}."
    )


def test_protocol_a_deterministic(df: pd.DataFrame, config) -> None:
    r1 = splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size)
    r2 = splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size)
    assert np.array_equal(r1[0].test_idx, r2[0].test_idx)


# ---------------------------------------------------------------------------
# Protocol B
# ---------------------------------------------------------------------------

def test_protocol_b_split_count(df: pd.DataFrame, config) -> None:
    result = splits.protocol_b(
        df, n_splits=config.splits.n_shuffle_splits,
        test_size=config.splits.test_size, seed=config.seed,
    )
    assert len(result) == config.splits.n_shuffle_splits


def test_protocol_b_split_ids(df: pd.DataFrame, config) -> None:
    result = splits.protocol_b(
        df, n_splits=config.splits.n_shuffle_splits,
        test_size=config.splits.test_size, seed=config.seed,
    )
    for i, sr in enumerate(result):
        assert sr.protocol == "B"
        assert sr.split_id == i
        assert sr.fold == 0


def test_protocol_b_all_valid_partitions(df: pd.DataFrame, config) -> None:
    result = splits.protocol_b(
        df, n_splits=10,  # reduced for speed; correctness doesn't depend on n
        test_size=config.splits.test_size, seed=config.seed,
    )
    for sr in result:
        _assert_valid_partition(sr, len(df))


def test_protocol_b_has_replicate_overlap(df: pd.DataFrame, config) -> None:
    """B is a row-level random split; replicates WILL be split (object of study)."""
    result = splits.protocol_b(
        df, n_splits=10, test_size=config.splits.test_size, seed=config.seed,
    )
    # Expect overlap in the large majority of splits on this dataset.
    overlapping = sum(1 for sr in result if _replicate_overlap(df, sr) > 0)
    assert overlapping > len(result) // 2, (
        "Expected most protocol-B splits to have replicate overlap; "
        f"only {overlapping}/{len(result)} did."
    )


def test_protocol_b_deterministic(df: pd.DataFrame, config) -> None:
    r1 = splits.protocol_b(
        df, n_splits=5, test_size=config.splits.test_size, seed=config.seed,
    )
    r2 = splits.protocol_b(
        df, n_splits=5, test_size=config.splits.test_size, seed=config.seed,
    )
    for s1, s2 in zip(r1, r2):
        assert np.array_equal(s1.test_idx, s2.test_idx)


# ---------------------------------------------------------------------------
# Protocol C
# ---------------------------------------------------------------------------

def test_protocol_c_fold_count(df: pd.DataFrame, config) -> None:
    result = splits.protocol_c(df, n_splits=config.splits.gkf_n_splits)
    assert len(result) == config.splits.gkf_n_splits


def test_protocol_c_fold_ids(df: pd.DataFrame, config) -> None:
    result = splits.protocol_c(df, n_splits=config.splits.gkf_n_splits)
    for fold_idx, sr in enumerate(result):
        assert sr.protocol == "C"
        assert sr.split_id == 0
        assert sr.fold == fold_idx


def test_protocol_c_valid_partitions(df: pd.DataFrame, config) -> None:
    for sr in splits.protocol_c(df, n_splits=config.splits.gkf_n_splits):
        _assert_valid_partition(sr, len(df))


def test_protocol_c_zero_replicate_overlap(df: pd.DataFrame, config) -> None:
    """GroupKFold by condition_id must never split a replicate group (§1.2)."""
    for sr in splits.protocol_c(df, n_splits=config.splits.gkf_n_splits):
        overlap = _replicate_overlap(df, sr)
        assert overlap == 0, (
            f"Protocol C fold {sr.fold}: replicate_overlap={overlap} (must be 0)."
        )


def test_protocol_c_zero_condition_overlap(df: pd.DataFrame, config) -> None:
    """No condition appears in both train and test in any fold."""
    for sr in splits.protocol_c(df, n_splits=config.splits.gkf_n_splits):
        overlap = _condition_overlap(df, sr)
        assert overlap == 0, (
            f"Protocol C fold {sr.fold}: condition_overlap={overlap} (must be 0)."
        )


def test_protocol_c_all_conditions_covered(df: pd.DataFrame, config) -> None:
    """Union of all test folds must cover every condition exactly once."""
    test_conditions: list[str] = []
    for sr in splits.protocol_c(df, n_splits=config.splits.gkf_n_splits):
        test_conditions.extend(df.iloc[sr.test_idx][CONDITION_ID].tolist())
    assert sorted(test_conditions) == sorted(df[CONDITION_ID].tolist())


# ---------------------------------------------------------------------------
# Protocol D
# ---------------------------------------------------------------------------

def test_protocol_d_fold_count(df: pd.DataFrame) -> None:
    result = splits.protocol_d(df)
    n_membranes = df[MEMBRANE_ID].nunique()
    assert len(result) == n_membranes


def test_protocol_d_one_membrane_per_fold(df: pd.DataFrame) -> None:
    result = splits.protocol_d(df)
    membrane_values = sorted(df[MEMBRANE_ID].unique())
    held_out = []
    for sr in result:
        test_mems = df.iloc[sr.test_idx][MEMBRANE_ID].unique()
        assert len(test_mems) == 1, (
            f"Protocol D fold {sr.fold}: test set contains >1 membrane: {test_mems}."
        )
        held_out.append(int(test_mems[0]))
    assert sorted(held_out) == membrane_values


def test_protocol_d_valid_partitions(df: pd.DataFrame) -> None:
    for sr in splits.protocol_d(df):
        _assert_valid_partition(sr, len(df))


def test_protocol_d_zero_replicate_overlap(df: pd.DataFrame) -> None:
    """LOGO by membrane never splits replicates (replicates are within-membrane)."""
    for sr in splits.protocol_d(df):
        overlap = _replicate_overlap(df, sr)
        assert overlap == 0, (
            f"Protocol D fold {sr.fold}: replicate_overlap={overlap} (must be 0)."
        )


def test_protocol_d_zero_condition_overlap(df: pd.DataFrame) -> None:
    for sr in splits.protocol_d(df):
        overlap = _condition_overlap(df, sr)
        assert overlap == 0, (
            f"Protocol D fold {sr.fold}: condition_overlap={overlap} (must be 0)."
        )


# ---------------------------------------------------------------------------
# Protocol E
# ---------------------------------------------------------------------------

def test_protocol_e_fold_count(df: pd.DataFrame) -> None:
    result = splits.protocol_e(df)
    assert len(result) == df[FEED_TYPE].nunique()


def test_protocol_e_one_feed_type_per_fold(df: pd.DataFrame) -> None:
    result = splits.protocol_e(df)
    held_out: list[str] = []
    for sr in result:
        test_feeds = df.iloc[sr.test_idx][FEED_TYPE].unique()
        assert len(test_feeds) == 1, (
            f"Protocol E fold {sr.fold}: test set contains >1 feed type: {test_feeds}."
        )
        held_out.append(str(test_feeds[0]))
    assert sorted(held_out) == sorted(df[FEED_TYPE].unique())


def test_protocol_e_valid_partitions(df: pd.DataFrame) -> None:
    for sr in splits.protocol_e(df):
        _assert_valid_partition(sr, len(df))


def test_protocol_e_zero_replicate_overlap(df: pd.DataFrame) -> None:
    for sr in splits.protocol_e(df):
        overlap = _replicate_overlap(df, sr)
        assert overlap == 0, (
            f"Protocol E fold {sr.fold}: replicate_overlap={overlap} (must be 0)."
        )


# ---------------------------------------------------------------------------
# compute_overlap_stats
# ---------------------------------------------------------------------------

def test_overlap_stats_all_zero_on_grouped(df: pd.DataFrame, config) -> None:
    """GroupKFold folds must have zero overlap on all three axes."""
    for sr in splits.protocol_c(df, n_splits=config.splits.gkf_n_splits):
        stats = splits.compute_overlap_stats(df, sr.train_idx, sr.test_idx)
        assert stats.condition_overlap == 0
        assert stats.replicate_overlap == 0


def test_overlap_stats_nonzero_on_random(df: pd.DataFrame, config) -> None:
    """Random split (A) must have non-zero overlap on this dataset."""
    result = splits.protocol_a(df, seed=config.seed, test_size=config.splits.test_size)
    stats = splits.compute_overlap_stats(df, result[0].train_idx, result[0].test_idx)
    assert stats.replicate_overlap > 0, (
        "Expected non-zero replicate_overlap for a random split."
    )
    assert stats.membrane_overlap > 0, (
        "Expected non-zero membrane_overlap for a random split."
    )


# ---------------------------------------------------------------------------
# build_leakage_audit
# ---------------------------------------------------------------------------

def test_leakage_audit_columns(df: pd.DataFrame, config) -> None:
    result = splits.protocol_b(
        df, n_splits=5, test_size=config.splits.test_size, seed=config.seed,
    )
    audit = splits.build_leakage_audit(df, result)
    expected_cols = {
        "split_id", "fold", "protocol",
        "condition_overlap", "replicate_overlap", "membrane_overlap",
        "leakage_warning",
    }
    assert expected_cols.issubset(set(audit.columns))


def test_leakage_audit_row_count(df: pd.DataFrame, config) -> None:
    n = 10
    result = splits.protocol_b(
        df, n_splits=n, test_size=config.splits.test_size, seed=config.seed,
    )
    audit = splits.build_leakage_audit(df, result)
    assert len(audit) == n


def test_leakage_audit_warning_iff_replicate_overlap(
    df: pd.DataFrame, config
) -> None:
    result = splits.protocol_b(
        df, n_splits=10, test_size=config.splits.test_size, seed=config.seed,
    )
    audit = splits.build_leakage_audit(df, result)
    for _, row in audit.iterrows():
        expected_warning = row["replicate_overlap"] > 0
        assert row["leakage_warning"] == expected_warning, (
            f"leakage_warning mismatch at split_id={row['split_id']}: "
            f"replicate_overlap={row['replicate_overlap']}, "
            f"leakage_warning={row['leakage_warning']}."
        )


def test_leakage_audit_grouped_no_warnings(df: pd.DataFrame, config) -> None:
    """Grouped protocols must produce zero warnings in the leakage audit."""
    for gen_fn, label in [
        (lambda: splits.protocol_c(df, n_splits=config.splits.gkf_n_splits), "C"),
        (lambda: splits.protocol_d(df), "D"),
        (lambda: splits.protocol_e(df), "E"),
    ]:
        audit = splits.build_leakage_audit(df, gen_fn())
        warnings = audit["leakage_warning"].sum()
        assert warnings == 0, (
            f"Protocol {label} produced {warnings} leakage warning(s); expected 0."
        )
