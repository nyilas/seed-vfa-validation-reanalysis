"""Dataset diagnostics: effective sample-to-feature ratio (AGENTS.md §10, Exp-D).

Purpose
-------
Compute the nominal vs effective sample-to-feature ratio (SFR) for each
feature set, quantifying how severely the dataset is underdetermined.

Three SFR levels
----------------
=========================================  ======================================
Level                                      N used
=========================================  ======================================
nominal_sfr                                distinct_N / n_features  (79 rows;
                                           treats every row as independent)
effective_sfr_condition                    n_conditions / n_features  (46 unique
                                           experimental conditions)
effective_sfr_domain                       n_domains / n_features  (4 MWCO
                                           membrane types)
=========================================  ======================================

At the domain level, any feature set with more than 4 columns has
effective_sfr_domain < 1 — the learning problem is severely underdetermined
with respect to membrane generalisation.

Output schema (:data:`SFR_TABLE_COLUMNS`)
-----------------------------------------
``feature_set, n_features, nominal_N, distinct_N, n_conditions, n_domains,
nominal_sfr, effective_sfr_condition, effective_sfr_domain``

``nominal_N``, ``distinct_N``, ``n_conditions``, and ``n_domains`` are the same
for every row (they are dataset-level quantities); repeating them makes the CSV
self-contained and removes the need to join a separate summary table.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .features import get_columns
from .groups import CONDITION_ID, MEMBRANE_ID


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

SFR_TABLE_COLUMNS: tuple[str, ...] = (
    "feature_set",
    "n_features",
    "nominal_N",
    "distinct_N",
    "n_conditions",
    "n_domains",
    "nominal_sfr",
    "effective_sfr_condition",
    "effective_sfr_domain",
)


# ---------------------------------------------------------------------------
# Dataset-level summary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSummary:
    """Dataset-level sample counts used to compute SFR values.

    Attributes
    ----------
    nominal_N:
        Row count before de-duplication (from raw CSV or config).
    distinct_N:
        Row count after de-duplication (``len(df)``).
    n_conditions:
        Number of distinct ``condition_id`` values (independent experimental
        conditions, ignoring within-condition replication).
    n_domains:
        Number of distinct ``membrane_id`` values (= number of MWCO levels;
        the coarsest grouping of samples).
    """

    nominal_N: int
    distinct_N: int
    n_conditions: int
    n_domains: int


def compute_dataset_summary(
    df: pd.DataFrame,
    *,
    nominal_n_raw: int,
) -> DatasetSummary:
    """Compute dataset-level counts from the grouped DataFrame.

    Parameters
    ----------
    df:
        Loaded ``seed_with_groups.csv`` (already de-duplicated).
    nominal_n_raw:
        Row count of the raw CSV before de-duplication (i.e.
        ``config.data_contract.raw_n_rows``).

    Returns
    -------
    DatasetSummary
    """
    return DatasetSummary(
        nominal_N=nominal_n_raw,
        distinct_N=len(df),
        n_conditions=int(df[CONDITION_ID].nunique()),
        n_domains=int(df[MEMBRANE_ID].nunique()),
    )


# ---------------------------------------------------------------------------
# SFR table
# ---------------------------------------------------------------------------


def compute_sfr_table(
    df: pd.DataFrame,
    feature_set_ids: tuple[str, ...],
    *,
    nominal_n_raw: int,
) -> pd.DataFrame:
    """Compute nominal and effective SFR for each feature set.

    Parameters
    ----------
    df:
        Loaded ``seed_with_groups.csv``.
    feature_set_ids:
        Ordered tuple of feature-set IDs to include (e.g.
        ``("FS1", "FS2", "FS3", "FS4", "FS5")``).
    nominal_n_raw:
        Raw row count before de-duplication (used as ``nominal_N`` in the
        output).

    Returns
    -------
    pandas.DataFrame
        One row per feature set, schema: :data:`SFR_TABLE_COLUMNS`.
    """
    summary = compute_dataset_summary(df, nominal_n_raw=nominal_n_raw)

    rows = []
    for fs_id in feature_set_ids:
        cols = get_columns(fs_id)
        n_feat = len(cols)
        rows.append(
            {
                "feature_set": fs_id,
                "n_features": n_feat,
                "nominal_N": summary.nominal_N,
                "distinct_N": summary.distinct_N,
                "n_conditions": summary.n_conditions,
                "n_domains": summary.n_domains,
                "nominal_sfr": summary.distinct_N / n_feat,
                "effective_sfr_condition": summary.n_conditions / n_feat,
                "effective_sfr_domain": summary.n_domains / n_feat,
            }
        )

    return pd.DataFrame(rows, columns=list(SFR_TABLE_COLUMNS))
