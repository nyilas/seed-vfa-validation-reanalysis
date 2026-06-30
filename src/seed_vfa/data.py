"""Raw data loading and de-duplication for the Seed/VFA dataset.

Purpose
-------
Load ``data/raw/Seed_Dataset.csv`` and remove the single known exact-duplicate
row *before any split* (AGENTS.md §1, invariant 1). This module owns the raw
CSV schema (column names and their semantic groupings) and produces the
de-duplicated frame written to ``seed_clean.csv``. Derived grouping columns are
added separately in :mod:`seed_vfa.groups`.

Invariants
----------
* The raw file has exactly ``raw_n_rows`` rows and the columns in
  :data:`RAW_COLUMNS`.
* Exactly ``n_exact_duplicates`` exact-duplicate rows are removed; for each
  removed row its surviving "first occurrence" twin is recorded so that the
  ``is_duplicate_removed`` flag (AGENTS.md §4) can mark it downstream.
* No information is dropped or imputed beyond the exact-duplicate removal.

This module performs no scaling, selection, or imputation: those must happen
inside training folds only (AGENTS.md §1, invariant 3).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --- Raw CSV schema (data/raw/Seed_Dataset.csv) ------------------------------
# These column names are intrinsic to the fixed input dataset, not tunable
# parameters; they are defined once here and reused by groups.py.

TARGET: str = "Cret total VFAs"

MWCO_COLUMN: str = "MWCO [Da]"
PH_COLUMN: str = "pH"
TEMPERATURE_COLUMN: str = "Temperature [°C]"
PRESSURE_COLUMN: str = "Pressure [bar]"

# Feed-ion concentration columns. feed_type == "simple" iff all four are 0
# (AGENTS.md §4).
FEED_ION_COLUMNS: tuple[str, ...] = (
    "Monovalent cation feed [mmol/L]",
    "Divalent cation feed [mmol/L]",
    "Monovalent anion feed [mmol/L]",
    "Divalent anion feed [mmol/L]",
)

# condition_id key = (membrane, pH, temperature, pressure, feed-ion vector)
# (AGENTS.md §4). Order is fixed for deterministic, reproducible keys.
CONDITION_KEY_COLUMNS: tuple[str, ...] = (
    MWCO_COLUMN,
    PH_COLUMN,
    TEMPERATURE_COLUMN,
    PRESSURE_COLUMN,
    *FEED_ION_COLUMNS,
)

RAW_COLUMNS: tuple[str, ...] = (
    MWCO_COLUMN,
    "Average surface roughness[nm]",
    "Zeta potential [mV]",
    "Static contact angle [°]",
    "MgSO4 rejection [%]",
    "NaCl rejection [%]",
    PH_COLUMN,
    TEMPERATURE_COLUMN,
    PRESSURE_COLUMN,
    "PWP [LMH/bar]",
    *FEED_ION_COLUMNS,
    TARGET,
)


@dataclass(frozen=True)
class DeduplicationResult:
    """Outcome of removing exact-duplicate rows.

    Attributes
    ----------
    clean:
        De-duplicated frame with a fresh ``RangeIndex`` (0..n-1).
    n_removed:
        Number of rows dropped (each a non-first occurrence of an exact match).
    removed_raw_index:
        Original (pre-dedup) integer positions of the removed rows. Recorded so
        the audit log can state exactly which rows were dropped (AGENTS.md §1.1).
    duplicate_twin_mask:
        Boolean array aligned to ``clean``: ``True`` on each surviving "first
        occurrence" whose later duplicate was removed. Used by groups.py to set
        the ``is_duplicate_removed`` column (AGENTS.md §4).
    """

    clean: pd.DataFrame
    n_removed: int
    removed_raw_index: tuple[int, ...]
    duplicate_twin_mask: np.ndarray


def load_raw(path: str | Path, *, expected_n_rows: int | None = None) -> pd.DataFrame:
    """Load the raw Seed dataset CSV and validate its schema.

    Parameters
    ----------
    path:
        Path to ``Seed_Dataset.csv``.
    expected_n_rows:
        If given, assert the raw frame has exactly this many rows (fail loudly
        otherwise). Pass ``config.data_contract.raw_n_rows``.

    Returns
    -------
    pandas.DataFrame
        Raw data with a default ``RangeIndex``.

    Raises
    ------
    FileNotFoundError
        If the CSV does not exist.
    ValueError
        If the columns or (when checked) the row count do not match the
        expected schema.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)

    actual_columns = tuple(df.columns)
    if actual_columns != RAW_COLUMNS:
        raise ValueError(
            "Raw dataset columns do not match the expected schema.\n"
            f"  expected: {RAW_COLUMNS}\n"
            f"  actual:   {actual_columns}"
        )

    if expected_n_rows is not None and len(df) != expected_n_rows:
        raise ValueError(
            f"Raw dataset row count mismatch: expected {expected_n_rows}, "
            f"got {len(df)}."
        )

    return df


def deduplicate(
    df: pd.DataFrame, *, expected_n_removed: int | None = None
) -> DeduplicationResult:
    """Remove exact-duplicate rows, keeping the first occurrence of each.

    Implements AGENTS.md §1 invariant 1: the exact duplicate is removed *before
    any split*, the row count goes 80 → 79, and which row was removed is
    recorded.

    Parameters
    ----------
    df:
        Raw frame from :func:`load_raw` (default ``RangeIndex`` assumed, so
        ``df.index`` gives original 0-based row positions).
    expected_n_removed:
        If given, assert exactly this many rows are removed (fail loudly
        otherwise). Pass ``config.data_contract.n_exact_duplicates``.

    Returns
    -------
    DeduplicationResult
    """
    # Rows flagged True are non-first occurrences -> the ones to drop.
    removed_mask = df.duplicated(keep="first")
    n_removed = int(removed_mask.sum())

    if expected_n_removed is not None and n_removed != expected_n_removed:
        raise ValueError(
            f"Unexpected number of exact-duplicate rows: expected "
            f"{expected_n_removed}, found {n_removed}."
        )

    removed_raw_index = tuple(int(i) for i in df.index[removed_mask])

    # Surviving "first occurrence" twins: members of a duplicated group that are
    # NOT being removed. duplicated(keep=False) marks every member of any group
    # with >1 identical rows.
    all_members_mask = df.duplicated(keep=False)
    twin_mask_raw = all_members_mask & ~removed_mask

    clean = df.loc[~removed_mask].reset_index(drop=True)
    # twin_mask_raw is aligned to the kept rows in order, so dropping the removed
    # rows and resetting the index keeps it aligned to `clean`.
    duplicate_twin_mask = twin_mask_raw.loc[~removed_mask].to_numpy(dtype=bool)

    return DeduplicationResult(
        clean=clean,
        n_removed=n_removed,
        removed_raw_index=removed_raw_index,
        duplicate_twin_mask=duplicate_twin_mask,
    )


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Write ``df`` to CSV (no index column), creating parent dirs as needed.

    Returns the resolved output path.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path
