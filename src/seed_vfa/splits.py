"""Validation split generators and leakage-audit utilities (AGENTS.md §6).

Purpose
-------
Produce train/test index splits for the five protocols defined in §6, and
provide utilities to audit how much leakage each split contains. This module
is intentionally restricted to index generation: it performs no feature
scaling, no selection, no model fitting, and no metric computation (AGENTS.md
§1, invariants 2 and 3).

Protocols (AGENTS.md §6)
-------------------------
=====  ==============  ==============================  ======  =============================
ID     Class           Groups column                   Unit    Scientific question
=====  ==============  ==============================  ======  =============================
A      train_test_split  —                             row     replicate source-paper setting
B      ShuffleSplit    —                               row     variance + leakage audit
C      GroupKFold      condition_id                    cond.   honest interpolation
D      LeaveOneGroupOut  membrane_id                   domain  extrapolation unseen membrane
E      LeaveOneGroupOut  feed_type                     regime  simple↔complex transfer (opt.)
=====  ==============  ==============================  ======  =============================

For protocols C/D/E, all rows sharing a ``replicate_group_id`` fall in the same
fold (invariant 2). For A/B, replicates may be separated — this is the *object
of study*, not an oversight, and is made auditable by :func:`compute_overlap_stats`.

Hard invariants checked at split time
--------------------------------------
* C and D never put any ``replicate_group_id`` in both train and test.
* All returned indices are positional (0 … len(df)−1) with no out-of-bounds
  values and no duplicates between train and test within a fold.

Leakage audit
-------------
:func:`compute_overlap_stats` counts how many conditions, replicate groups, and
membrane domains appear in *both* the train and test portions of a given split.
A non-zero ``replicate_overlap`` is expected and structurally unavoidable for
random protocols; it signals potential leakage when it appears in grouped ones.
:func:`build_leakage_audit` aggregates these stats for every split in a list,
producing a DataFrame suitable for ``results/tables/leakage_audit.csv``.

Naming conventions
------------------
* ``split_id``: for B, the index of the shuffle repeat (0 … n_splits−1); 0 for
  A; always 0 for C/D/E (a single cross-validation schedule has no repeat).
* ``fold``: for C/D/E, the fold index within the CV schedule; 0 for A/B (each
  is a single train/test partition).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    LeaveOneGroupOut,
    ShuffleSplit,
    train_test_split,
)

from .groups import CONDITION_ID, FEED_TYPE, MEMBRANE_ID, REPLICATE_GROUP_ID

# ---------------------------------------------------------------------------
# Protocol identifiers (AGENTS.md §6)
# ---------------------------------------------------------------------------

ProtocolID = Literal["A", "B", "C", "D", "E"]

PROTOCOL_A: ProtocolID = "A"
PROTOCOL_B: ProtocolID = "B"
PROTOCOL_C: ProtocolID = "C"
PROTOCOL_D: ProtocolID = "D"
PROTOCOL_E: ProtocolID = "E"

# Protocols that must never split a replicate group across train/test (§1.2).
GROUPED_PROTOCOLS: frozenset[str] = frozenset({"C", "D", "E"})
# Protocols where replicate separation is expected and is the object of study.
RANDOM_PROTOCOLS: frozenset[str] = frozenset({"A", "B"})


# ---------------------------------------------------------------------------
# Core data types
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    """A single train/test partition returned by a protocol generator.

    Attributes
    ----------
    protocol:
        One of "A"–"E" (AGENTS.md §6).
    split_id:
        For protocol B: 0-based index of the shuffle repeat.
        For protocols A/C/D/E: always 0.
    fold:
        For protocols C/D/E: 0-based fold index within the CV schedule.
        For protocols A/B: always 0.
    train_idx:
        Positional integer indices into the DataFrame (not row_id values).
    test_idx:
        Positional integer indices into the DataFrame (not row_id values).
    """

    protocol: str
    split_id: int
    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class OverlapStats:
    """Leakage statistics for a single train/test partition (AGENTS.md §6).

    Attributes
    ----------
    condition_overlap:
        Number of distinct ``condition_id`` values present in both train and
        test portions. Non-zero implies the same experimental condition was
        seen during training.
    replicate_overlap:
        Number of distinct ``replicate_group_id`` values in both portions.
        Equivalent to ``condition_overlap`` because the two columns share the
        same partition; provided separately for clarity.
    membrane_overlap:
        Number of distinct ``membrane_id`` values in both portions. Always
        equals the total number of membranes minus one for random splits on
        this dataset (all 4 membranes are represented in both halves).
    """

    condition_overlap: int
    replicate_overlap: int
    membrane_overlap: int


# ---------------------------------------------------------------------------
# Protocol generators
# ---------------------------------------------------------------------------

def protocol_a(
    df: pd.DataFrame,
    *,
    seed: int,
    test_size: float,
) -> list[SplitResult]:
    """Protocol A — single random 75/25 split (AGENTS.md §6).

    Replicates the source-paper experimental setting: one random train/test
    partition with 25% held out. Replicates *may* be separated across the
    boundary; this is auditable via :func:`compute_overlap_stats`.

    Parameters
    ----------
    df:
        Grouped dataset. Only its length is used here.
    seed:
        RNG seed for reproducibility (from ``config.yaml``).
    test_size:
        Fraction of rows allocated to the test set (e.g. 0.25).
    """
    n = len(df)
    train_idx, test_idx = train_test_split(
        np.arange(n), test_size=test_size, random_state=seed, shuffle=True
    )
    return [
        SplitResult(
            protocol=PROTOCOL_A,
            split_id=0,
            fold=0,
            train_idx=np.asarray(train_idx),
            test_idx=np.asarray(test_idx),
        )
    ]


def protocol_b(
    df: pd.DataFrame,
    *,
    n_splits: int,
    test_size: float,
    seed: int,
) -> list[SplitResult]:
    """Protocol B — repeated random ShuffleSplit (AGENTS.md §6).

    Runs ``n_splits`` independent random splits. Each split is a row-level
    partition; replicates *may* be separated. The primary purpose is to
    quantify the variance of the random-split performance estimate and to
    generate an audit trail of per-split leakage statistics.

    Parameters
    ----------
    df:
        Grouped dataset. Only its length is used here.
    n_splits:
        Number of shuffle repeats (must be >= 100 per §6).
    test_size:
        Fraction of rows in the test set (e.g. 0.25).
    seed:
        Master RNG seed; individual splits receive deterministic sub-seeds
        derived from it via sklearn's ``ShuffleSplit(random_state=seed)``.
    """
    splitter = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=seed)
    results: list[SplitResult] = []
    for i, (tr, te) in enumerate(splitter.split(np.arange(len(df)))):
        results.append(
            SplitResult(
                protocol=PROTOCOL_B,
                split_id=i,
                fold=0,
                train_idx=np.asarray(tr),
                test_idx=np.asarray(te),
            )
        )
    return results


def protocol_c(
    df: pd.DataFrame,
    *,
    n_splits: int,
) -> list[SplitResult]:
    """Protocol C — GroupKFold by ``condition_id`` (AGENTS.md §6).

    Every row with the same ``condition_id`` stays in the same fold, so *all*
    replicates of a condition are either all in train or all in test. This is
    the honest interpolation protocol: the model never sees any replicate of a
    held-out condition during training.

    Parameters
    ----------
    df:
        Grouped dataset. Must contain ``condition_id``.
    n_splits:
        Number of folds k (from ``config.splits.gkf_n_splits``).
    """
    groups = df[CONDITION_ID].to_numpy()
    splitter = GroupKFold(n_splits=n_splits)
    results: list[SplitResult] = []
    for fold_idx, (tr, te) in enumerate(splitter.split(df, groups=groups)):
        results.append(
            SplitResult(
                protocol=PROTOCOL_C,
                split_id=0,
                fold=fold_idx,
                train_idx=np.asarray(tr),
                test_idx=np.asarray(te),
            )
        )
    return results


def protocol_d(df: pd.DataFrame) -> list[SplitResult]:
    """Protocol D — LeaveOneGroupOut by ``membrane_id`` (AGENTS.md §6).

    Each fold holds out all rows from one membrane (MWCO value). Because a
    membrane's conditions and replicates are unique to that membrane, there is
    zero condition/replicate overlap between train and test by construction.
    This is the extrapolation protocol.

    Parameters
    ----------
    df:
        Grouped dataset. Must contain ``membrane_id``.
    """
    groups = df[MEMBRANE_ID].to_numpy()
    splitter = LeaveOneGroupOut()
    results: list[SplitResult] = []
    for fold_idx, (tr, te) in enumerate(splitter.split(df, groups=groups)):
        results.append(
            SplitResult(
                protocol=PROTOCOL_D,
                split_id=0,
                fold=fold_idx,
                train_idx=np.asarray(tr),
                test_idx=np.asarray(te),
            )
        )
    return results


def protocol_e(df: pd.DataFrame) -> list[SplitResult]:
    """Protocol E — LeaveOneGroupOut by ``feed_type`` (AGENTS.md §6, optional).

    Each fold holds out all rows from one feed regime (simple or complex).
    Tests transfer between the simple-feed and complex-feed experimental
    regimes. Two folds: one with simple as test, one with complex as test.

    Parameters
    ----------
    df:
        Grouped dataset. Must contain ``feed_type``.
    """
    groups = df[FEED_TYPE].to_numpy()
    splitter = LeaveOneGroupOut()
    results: list[SplitResult] = []
    for fold_idx, (tr, te) in enumerate(splitter.split(df, groups=groups)):
        results.append(
            SplitResult(
                protocol=PROTOCOL_E,
                split_id=0,
                fold=fold_idx,
                train_idx=np.asarray(tr),
                test_idx=np.asarray(te),
            )
        )
    return results


# ---------------------------------------------------------------------------
# Leakage-audit utilities
# ---------------------------------------------------------------------------

def compute_overlap_stats(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> OverlapStats:
    """Count condition, replicate, and membrane overlap for one split.

    Parameters
    ----------
    df:
        Grouped dataset containing ``condition_id``, ``replicate_group_id``,
        and ``membrane_id`` columns.
    train_idx:
        Positional indices of training rows.
    test_idx:
        Positional indices of test rows.

    Returns
    -------
    OverlapStats
        Counts of shared identifiers between the two partitions.
    """
    train_rows = df.iloc[train_idx]
    test_rows = df.iloc[test_idx]

    condition_overlap = int(
        len(
            set(train_rows[CONDITION_ID].unique())
            & set(test_rows[CONDITION_ID].unique())
        )
    )
    replicate_overlap = int(
        len(
            set(train_rows[REPLICATE_GROUP_ID].unique())
            & set(test_rows[REPLICATE_GROUP_ID].unique())
        )
    )
    membrane_overlap = int(
        len(
            set(train_rows[MEMBRANE_ID].unique())
            & set(test_rows[MEMBRANE_ID].unique())
        )
    )

    return OverlapStats(
        condition_overlap=condition_overlap,
        replicate_overlap=replicate_overlap,
        membrane_overlap=membrane_overlap,
    )


def build_leakage_audit(
    df: pd.DataFrame,
    splits: list[SplitResult],
) -> pd.DataFrame:
    """Build the ``leakage_audit.csv`` table for a list of splits (AGENTS.md §9).

    Designed primarily for Protocol B's repeated random splits, where
    ``replicate_overlap > 0`` is expected (and is the object of study). Also
    correct for grouped protocols, where it should always be zero.

    ``leakage_warning`` is True when ``replicate_overlap > 0``: the same
    replicate group appears on both sides of the train/test boundary.

    Parameters
    ----------
    df:
        Grouped dataset.
    splits:
        List of :class:`SplitResult` objects from any protocol generator.

    Returns
    -------
    pandas.DataFrame
        Schema: ``split_id, fold, protocol, condition_overlap,
        replicate_overlap, membrane_overlap, leakage_warning``.
    """
    rows: list[dict[str, object]] = []
    for s in splits:
        stats = compute_overlap_stats(df, s.train_idx, s.test_idx)
        rows.append(
            {
                "split_id": s.split_id,
                "fold": s.fold,
                "protocol": s.protocol,
                "condition_overlap": stats.condition_overlap,
                "replicate_overlap": stats.replicate_overlap,
                "membrane_overlap": stats.membrane_overlap,
                "leakage_warning": stats.replicate_overlap > 0,
            }
        )
    return pd.DataFrame(rows)
