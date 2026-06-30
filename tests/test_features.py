"""Tests for the feature-set registry (AGENTS.md §8).

Verifies:
- Registry completeness: all eight IDs are registered.
- Column counts match the AGENTS.md §8 specification exactly.
- All columns in every feature set are present in the real dataset.
- Target and group-ID columns never appear in any feature set.
- FS6 = FS2 minus ζ; FS2_NO_PH = FS2 minus pH (derived-pair invariants).
- FS3, FS5, FS7 include feed_type (categorical column); FS1/FS2/FS4/FS6/FS2_NO_PH do not.
- select_features returns correct shape and column order.
- validate_columns fails loudly when a column is missing from df.
- get_definition and get_columns raise FeatureSetError on unknown IDs.
- No duplicate column names within any feature set.
- select_features never includes the target or group-ID columns.
"""

from __future__ import annotations

import pandas as pd
import pytest

from seed_vfa import features
from seed_vfa.config import load_config
from seed_vfa.data import TARGET
from seed_vfa.features import (
    ALL_FEATURE_SET_IDS,
    FeatureSetError,
    get_columns,
    get_definition,
    select_features,
    validate_columns,
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


# ---------------------------------------------------------------------------
# Registry completeness
# ---------------------------------------------------------------------------

def test_all_ids_registered() -> None:
    for fid in ALL_FEATURE_SET_IDS:
        defn = get_definition(fid)
        assert defn.feature_set_id == fid


def test_registry_has_no_extra_ids() -> None:
    assert set(features.FEATURE_SETS.keys()) == set(ALL_FEATURE_SET_IDS)


# ---------------------------------------------------------------------------
# Column counts (AGENTS.md §8)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fid,expected_n", [
    ("FS1", 14),   # all 14 raw measurement columns
    ("FS2", 5),    # ζ, pH, pressure, PWP, monovalent anion feed
    ("FS3", 4),    # pH, pressure, temperature, feed_type
    ("FS4", 7),    # MWCO, roughness, ζ, contact angle, MgSO4 rej, NaCl rej, PWP
    ("FS5", 4),    # pH, pressure, feed_type, MWCO
    ("FS6", 4),    # FS2 minus ζ
    ("FS7", 8),    # pH, T, P, feed_type, 4 feed-ion concentrations
    ("FS2_NO_PH", 4),  # FS2 minus pH
])
def test_column_count(fid: str, expected_n: int) -> None:
    cols = get_columns(fid)
    assert len(cols) == expected_n, (
        f"{fid}: expected {expected_n} columns, got {len(cols)}: {cols}"
    )


# ---------------------------------------------------------------------------
# All columns present in the real dataset
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fid", ALL_FEATURE_SET_IDS)
def test_all_columns_in_dataset(fid: str, df: pd.DataFrame) -> None:
    """validate_columns must not raise for any registered feature set."""
    validate_columns(fid, df)   # raises FeatureSetError if any column is missing


# ---------------------------------------------------------------------------
# Target and group-ID columns never in any feature set
# ---------------------------------------------------------------------------

_NON_FEATURE = {
    TARGET, "row_id", "condition_id", "replicate_group_id",
    "domain_id", "is_duplicate_removed",
}


@pytest.mark.parametrize("fid", ALL_FEATURE_SET_IDS)
def test_no_target_or_group_id_columns(fid: str) -> None:
    cols = set(get_columns(fid))
    leaked = cols & _NON_FEATURE
    assert len(leaked) == 0, (
        f"{fid} contains non-feature column(s): {leaked}"
    )


# ---------------------------------------------------------------------------
# FS1 exact content
# ---------------------------------------------------------------------------

def test_fs1_is_all_raw_columns() -> None:
    assert get_columns("FS1") == features.ALL_RAW_COLUMNS


# ---------------------------------------------------------------------------
# FS6 and FS2_NO_PH are strict subsets of FS2
# ---------------------------------------------------------------------------

def test_fs6_is_fs2_minus_zeta() -> None:
    fs2 = set(get_columns("FS2"))
    fs6 = set(get_columns("FS6"))
    zeta = "Zeta potential [mV]"
    assert zeta in fs2
    assert zeta not in fs6
    assert fs6 == fs2 - {zeta}, (
        f"FS6 should be FS2 minus ζ.\n  FS2: {sorted(fs2)}\n  FS6: {sorted(fs6)}"
    )


def test_fs2_no_ph_is_fs2_minus_ph() -> None:
    fs2 = set(get_columns("FS2"))
    fs2_no_ph = set(get_columns("FS2_NO_PH"))
    ph = "pH"
    assert ph in fs2
    assert ph not in fs2_no_ph
    assert fs2_no_ph == fs2 - {ph}, (
        f"FS2_NO_PH should be FS2 minus pH.\n  FS2: {sorted(fs2)}\n  FS2_NO_PH: {sorted(fs2_no_ph)}"
    )


def test_fs6_and_fs2_no_ph_same_size() -> None:
    """Both derived sets drop exactly one feature from FS2."""
    assert len(get_columns("FS6")) == len(get_columns("FS2_NO_PH"))


# ---------------------------------------------------------------------------
# feed_type presence / absence
# ---------------------------------------------------------------------------

_FEED_TYPE = "feed_type"

@pytest.mark.parametrize("fid", ["FS3", "FS5", "FS7"])
def test_feed_type_present(fid: str) -> None:
    assert _FEED_TYPE in get_columns(fid), (
        f"{fid} should contain 'feed_type' (categorical) but does not."
    )


@pytest.mark.parametrize("fid", ["FS1", "FS2", "FS4", "FS6", "FS2_NO_PH"])
def test_feed_type_absent(fid: str) -> None:
    assert _FEED_TYPE not in get_columns(fid), (
        f"{fid} should not contain 'feed_type' but does."
    )


# ---------------------------------------------------------------------------
# No duplicate column names within a feature set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fid", ALL_FEATURE_SET_IDS)
def test_no_duplicate_columns(fid: str) -> None:
    cols = get_columns(fid)
    assert len(cols) == len(set(cols)), (
        f"{fid} has duplicate column names: {[c for c in cols if cols.count(c) > 1]}"
    )


# ---------------------------------------------------------------------------
# select_features correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fid", ALL_FEATURE_SET_IDS)
def test_select_features_shape(fid: str, df: pd.DataFrame) -> None:
    result = select_features(fid, df)
    expected_cols = get_columns(fid)
    assert result.shape == (len(df), len(expected_cols)), (
        f"{fid}: shape {result.shape} != expected ({len(df)}, {len(expected_cols)})"
    )


@pytest.mark.parametrize("fid", ALL_FEATURE_SET_IDS)
def test_select_features_column_order(fid: str, df: pd.DataFrame) -> None:
    result = select_features(fid, df)
    assert tuple(result.columns) == get_columns(fid), (
        f"{fid}: column order mismatch.\n"
        f"  expected: {get_columns(fid)}\n"
        f"  got:      {tuple(result.columns)}"
    )


def test_select_features_returns_copy(df: pd.DataFrame) -> None:
    """Mutating the returned DataFrame must not affect the original."""
    result = select_features("FS2", df)
    original_val = df.iloc[0, df.columns.get_loc("Zeta potential [mV]")]
    result.iloc[0, 0] = original_val + 999.0
    assert df.iloc[0, df.columns.get_loc("Zeta potential [mV]")] == original_val


@pytest.mark.parametrize("fid", ALL_FEATURE_SET_IDS)
def test_select_features_no_target(fid: str, df: pd.DataFrame) -> None:
    result = select_features(fid, df)
    assert TARGET not in result.columns


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_get_definition_unknown_id() -> None:
    with pytest.raises(FeatureSetError, match="Unknown feature set"):
        get_definition("FS99")


def test_get_columns_unknown_id() -> None:
    with pytest.raises(FeatureSetError, match="Unknown feature set"):
        get_columns("NOT_A_FEATURE_SET")


def test_validate_columns_missing_column(df: pd.DataFrame) -> None:
    """validate_columns must raise FeatureSetError and name the missing column."""
    broken = df.drop(columns=["Zeta potential [mV]"])
    with pytest.raises(FeatureSetError, match="Zeta potential"):
        validate_columns("FS2", broken)


def test_validate_columns_lists_all_missing(df: pd.DataFrame) -> None:
    """All missing columns should be reported in one error, not one-at-a-time."""
    broken = df.drop(columns=["Zeta potential [mV]", "pH"])
    with pytest.raises(FeatureSetError) as exc_info:
        validate_columns("FS2", broken)
    msg = str(exc_info.value)
    assert "Zeta potential" in msg
    assert "pH" in msg


def test_select_features_unknown_id(df: pd.DataFrame) -> None:
    with pytest.raises(FeatureSetError):
        select_features("FS_INVALID", df)
