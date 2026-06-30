"""Derived grouping columns and data-contract validation (AGENTS.md §4).

Purpose
-------
Given the de-duplicated frame from :mod:`seed_vfa.data`, build the derived
columns that every downstream split, model and metric depends on, then assert
the §4 data contract and fail loudly on any violation.

Derived columns (AGENTS.md §4)
------------------------------
======================  =========================================================
``row_id``              stable integer index after de-duplication (0..n-1)
``membrane_id``         MWCO [Da] value, one of {225, 250, 300, 400}
``feed_type``           ``"simple"`` iff all feed-ion columns == 0, else
                        ``"complex"``
``condition_id``        unique key of (membrane, pH, temperature, pressure,
                        feed-ion vector); 46 distinct
``replicate_group_id``  one group per ``condition_id`` (replicates of the same
                        condition); identical partition to ``condition_id``
``domain_id``           alias of ``membrane_id`` (domain == membrane)
``is_duplicate_removed``  ``True`` on the surviving twin of the removed exact
                        duplicate
======================  =========================================================

Invariants
----------
``condition_id`` and ``replicate_group_id`` are assigned with ``sort=True`` so
the labels depend only on the data values, not on row order — making the
mapping reproducible (AGENTS.md §1, invariant 5). The validation step enforces
79 rows / 46 conditions / 4 membranes / 2 feed regimes (AGENTS.md §4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DataContract
from .data import (
    CONDITION_KEY_COLUMNS,
    FEED_ION_COLUMNS,
    MWCO_COLUMN,
    DeduplicationResult,
)

# Derived column names, defined once to avoid string drift across the codebase.
ROW_ID: str = "row_id"
MEMBRANE_ID: str = "membrane_id"
FEED_TYPE: str = "feed_type"
CONDITION_ID: str = "condition_id"
REPLICATE_GROUP_ID: str = "replicate_group_id"
DOMAIN_ID: str = "domain_id"
IS_DUPLICATE_REMOVED: str = "is_duplicate_removed"

DERIVED_COLUMNS: tuple[str, ...] = (
    ROW_ID,
    MEMBRANE_ID,
    FEED_TYPE,
    CONDITION_ID,
    REPLICATE_GROUP_ID,
    DOMAIN_ID,
    IS_DUPLICATE_REMOVED,
)

FEED_SIMPLE: str = "simple"
FEED_COMPLEX: str = "complex"


class DataContractError(ValueError):
    """Raised when the de-duplicated dataset violates the AGENTS.md §4 contract."""


def build_groups(dedup: DeduplicationResult) -> pd.DataFrame:
    """Attach the §4 derived columns to the de-duplicated frame.

    Parameters
    ----------
    dedup:
        Result of :func:`seed_vfa.data.deduplicate`. Its ``clean`` frame is
        copied (not mutated) and its ``duplicate_twin_mask`` populates
        ``is_duplicate_removed``.

    Returns
    -------
    pandas.DataFrame
        ``clean`` plus the derived columns of :data:`DERIVED_COLUMNS`.
    """
    df = dedup.clean.copy()
    n = len(df)

    if dedup.duplicate_twin_mask.shape[0] != n:
        raise ValueError(
            "duplicate_twin_mask length does not match the clean frame: "
            f"{dedup.duplicate_twin_mask.shape[0]} vs {n}."
        )

    # row_id: stable index after de-duplication.
    df[ROW_ID] = np.arange(n, dtype=int)

    # membrane_id: the MWCO value itself (domain == membrane). Stored as int.
    df[MEMBRANE_ID] = df[MWCO_COLUMN].astype(int)
    df[DOMAIN_ID] = df[MEMBRANE_ID]

    # feed_type: "simple" iff every feed-ion concentration is exactly 0.
    is_simple = (df[list(FEED_ION_COLUMNS)] == 0).all(axis=1)
    df[FEED_TYPE] = np.where(is_simple, FEED_SIMPLE, FEED_COMPLEX)

    # condition_id: dense code over the condition key, ordered by the sorted key
    # tuple so the labels are independent of row order (reproducibility).
    condition_code = df.groupby(
        list(CONDITION_KEY_COLUMNS), sort=True
    ).ngroup()
    df[CONDITION_ID] = condition_code.map(lambda c: f"C{c:02d}")

    # replicate_group_id: one group per condition (replicates share the same
    # condition). Identical partition to condition_id by construction (§4).
    df[REPLICATE_GROUP_ID] = condition_code.map(lambda c: f"R{c:02d}")

    # is_duplicate_removed: True on the surviving twin of the dropped duplicate.
    df[IS_DUPLICATE_REMOVED] = dedup.duplicate_twin_mask

    return df


def validate_contract(df: pd.DataFrame, contract: DataContract) -> None:
    """Assert the §4 data contract, raising :class:`DataContractError` if violated.

    All violations are collected and reported together so a single run surfaces
    every problem (rules §10: clear, contextualized failures).

    Parameters
    ----------
    df:
        Frame produced by :func:`build_groups`.
    contract:
        Expected counts and allowed values from ``config.yaml`` (AGENTS.md §4).

    Raises
    ------
    DataContractError
        If any invariant fails.
    """
    violations: list[str] = []

    missing = [c for c in DERIVED_COLUMNS if c not in df.columns]
    if missing:
        # Without the derived columns the remaining checks are meaningless.
        raise DataContractError(f"Missing derived columns: {missing}.")

    if len(df) != contract.n_rows:
        violations.append(f"row count: expected {contract.n_rows}, got {len(df)}.")

    n_conditions = int(df[CONDITION_ID].nunique())
    if n_conditions != contract.n_conditions:
        violations.append(
            f"condition_id count: expected {contract.n_conditions}, got {n_conditions}."
        )

    n_replicate_groups = int(df[REPLICATE_GROUP_ID].nunique())
    if n_replicate_groups != contract.n_conditions:
        violations.append(
            "replicate_group_id count must equal condition_id count "
            f"({contract.n_conditions}), got {n_replicate_groups}."
        )

    n_membranes = int(df[MEMBRANE_ID].nunique())
    if n_membranes != contract.n_membranes:
        violations.append(
            f"membrane_id count: expected {contract.n_membranes}, got {n_membranes}."
        )

    allowed_mwco = set(contract.membrane_mwco)
    unexpected_mwco = set(df[MEMBRANE_ID].unique()) - allowed_mwco
    if unexpected_mwco:
        violations.append(
            f"membrane_id values outside {sorted(allowed_mwco)}: "
            f"{sorted(unexpected_mwco)}."
        )

    n_feed_types = int(df[FEED_TYPE].nunique())
    if n_feed_types != contract.n_feed_types:
        violations.append(
            f"feed_type count: expected {contract.n_feed_types}, got {n_feed_types}."
        )

    allowed_feed = set(contract.feed_types)
    unexpected_feed = set(df[FEED_TYPE].unique()) - allowed_feed
    if unexpected_feed:
        violations.append(
            f"feed_type values outside {sorted(allowed_feed)}: "
            f"{sorted(unexpected_feed)}."
        )

    n_twin = int(df[IS_DUPLICATE_REMOVED].sum())
    if n_twin != contract.n_exact_duplicates:
        violations.append(
            f"is_duplicate_removed flags: expected {contract.n_exact_duplicates}, "
            f"got {n_twin}."
        )

    expected_row_ids = np.arange(len(df))
    if not np.array_equal(df[ROW_ID].to_numpy(), expected_row_ids):
        violations.append("row_id must be a contiguous 0..n-1 integer index.")

    if violations:
        raise DataContractError(
            "Data contract violated (AGENTS.md §4):\n  - "
            + "\n  - ".join(violations)
        )
