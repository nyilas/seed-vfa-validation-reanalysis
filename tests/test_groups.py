"""Acceptance tests for the data-audit layer (AGENTS.md §11, test_groups).

These tests are an independent oracle: the expected counts (79 rows / 46
conditions / 4 membranes / 2 feed regimes, one duplicate removed) are hardcoded
from the data contract (AGENTS.md §4) rather than read from ``config.yaml``, so
a wrong config cannot make the suite pass trivially.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from seed_vfa import data, groups
from seed_vfa.config import load_config
from seed_vfa.groups import DataContractError

# Ground-truth invariants (AGENTS.md §4), independent of config.yaml.
RAW_N_ROWS = 80
N_ROWS = 79
N_CONDITIONS = 46
N_MEMBRANES = 4
N_FEED_TYPES = 2
N_DUPLICATES_REMOVED = 1
EXPECTED_MWCO = {225, 250, 300, 400}
EXPECTED_FEED_TYPES = {"simple", "complex"}


@pytest.fixture(scope="module")
def grouped() -> pd.DataFrame:
    """Build the grouped dataset through the real pipeline (load -> dedup -> groups)."""
    config = load_config()
    raw = data.load_raw(config.paths.raw_dataset, expected_n_rows=RAW_N_ROWS)
    dedup = data.deduplicate(raw, expected_n_removed=N_DUPLICATES_REMOVED)
    return groups.build_groups(dedup)


def test_raw_has_one_exact_duplicate() -> None:
    """The raw file must contain exactly one removable exact-duplicate row."""
    config = load_config()
    raw = data.load_raw(config.paths.raw_dataset, expected_n_rows=RAW_N_ROWS)
    assert int(raw.duplicated(keep="first").sum()) == N_DUPLICATES_REMOVED


def test_row_count_after_dedup(grouped: pd.DataFrame) -> None:
    assert len(grouped) == N_ROWS


def test_condition_count(grouped: pd.DataFrame) -> None:
    assert grouped[groups.CONDITION_ID].nunique() == N_CONDITIONS


def test_replicate_groups_match_conditions(grouped: pd.DataFrame) -> None:
    """Replicate groups partition the data identically to condition_id (§4)."""
    assert grouped[groups.REPLICATE_GROUP_ID].nunique() == N_CONDITIONS
    # Every replicate group maps to exactly one condition and vice versa.
    pairing = grouped.groupby(groups.REPLICATE_GROUP_ID, observed=True)[
        groups.CONDITION_ID
    ].nunique()
    assert (pairing == 1).all()


def test_membrane_count_and_values(grouped: pd.DataFrame) -> None:
    assert grouped[groups.MEMBRANE_ID].nunique() == N_MEMBRANES
    assert set(grouped[groups.MEMBRANE_ID].unique()) == EXPECTED_MWCO


def test_feed_type_count_and_values(grouped: pd.DataFrame) -> None:
    assert grouped[groups.FEED_TYPE].nunique() == N_FEED_TYPES
    assert set(grouped[groups.FEED_TYPE].unique()) == EXPECTED_FEED_TYPES


def test_domain_is_alias_of_membrane(grouped: pd.DataFrame) -> None:
    assert grouped[groups.DOMAIN_ID].equals(grouped[groups.MEMBRANE_ID])


def test_duplicate_removed_flag(grouped: pd.DataFrame) -> None:
    """Exactly one surviving twin is flagged is_duplicate_removed (§4)."""
    assert int(grouped[groups.IS_DUPLICATE_REMOVED].sum()) == N_DUPLICATES_REMOVED


def test_row_id_is_contiguous(grouped: pd.DataFrame) -> None:
    assert np.array_equal(
        grouped[groups.ROW_ID].to_numpy(), np.arange(N_ROWS)
    )


def test_feed_type_simple_iff_no_feed_ions(grouped: pd.DataFrame) -> None:
    """feed_type == 'simple' exactly when all feed-ion columns are zero (§4)."""
    no_feed_ions = (grouped[list(data.FEED_ION_COLUMNS)] == 0).all(axis=1)
    is_simple = grouped[groups.FEED_TYPE] == groups.FEED_SIMPLE
    assert is_simple.equals(no_feed_ions.rename(None))


def test_validate_contract_passes(grouped: pd.DataFrame) -> None:
    config = load_config()
    # Should not raise.
    groups.validate_contract(grouped, config.data_contract)


def test_validate_contract_fails_loudly(grouped: pd.DataFrame) -> None:
    """A corrupted frame (a row dropped) must raise DataContractError."""
    config = load_config()
    broken = grouped.iloc[:-1].copy()
    with pytest.raises(DataContractError):
        groups.validate_contract(broken, config.data_contract)
