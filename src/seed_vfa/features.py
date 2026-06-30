"""Feature-set registry for the Seed/VFA reanalysis (AGENTS.md §8).

Purpose
-------
Define the seven named feature sets (FS1–FS7) and the FS2-no-pH variant used
in the identifiability demo, validate their column names against a DataFrame,
and provide a single retrieval function for use in experiment scripts.

This module performs *no* scaling, encoding, fitting, splitting, or metric
computation. It only answers the question "which columns belong to a given
feature set?", with loud failure if the answer is ambiguous or inconsistent.

Feature sets (AGENTS.md §8)
----------------------------
==========  ================  =====================================================
ID          Name              Columns
==========  ================  =====================================================
FS1         Full              all 14 raw measurement columns
FS2         Source-selected   ζ, pH, pressure, PWP, monovalent anion feed
FS3         Process-only      pH, pressure, temperature, feed_type
FS4         Membrane-only     MWCO, roughness, ζ, contact angle, MgSO4 rej,
                              NaCl rej, PWP
FS5         Parsimonious      pH, pressure, feed_type, MWCO
FS6         No-zeta           FS2 minus ζ  (Demo 2: aliased-pair drop)
FS7         No-membrane-props pH, temperature, pressure, feed_type, all 4 feed-ion
                              concentrations
FS2_NO_PH   Source-selected   FS2 minus pH  (Demo 2: symmetric pH↔zeta migration)
            no-pH
==========  ================  =====================================================

Notes on specific sets
----------------------
* FS3 and FS5 include the derived ``feed_type`` column (``"simple"`` / ``"complex"``).
  Models that use these sets must encode it (e.g. ``OrdinalEncoder`` inside a
  Pipeline).  Feature selection is the pipeline's responsibility; this module
  only names the columns.
* FS5 uses ``MWCO [Da]`` (the raw measurement column) rather than the derived
  ``membrane_id`` integer alias; they encode the same information but using the
  raw column keeps FS5 consistent with FS1 and FS4.
* FS6 and FS2_NO_PH exist to test whether importance migrates between the
  aliased pH–ζ pair when one member is removed (AGENTS.md §8, Demo 2).

Validation
----------
:func:`validate_columns` checks that every column in a feature set exists in a
given DataFrame, raising :class:`FeatureSetError` with a list of all missing
names. Silent column dropping is explicitly forbidden (rules §6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .data import TARGET


# ---------------------------------------------------------------------------
# Column name constants (single source of truth, no string literals scattered)
# ---------------------------------------------------------------------------

_MWCO = "MWCO [Da]"
_ROUGHNESS = "Average surface roughness[nm]"
_ZETA = "Zeta potential [mV]"
_CONTACT_ANGLE = "Static contact angle [°]"
_MGSO4_REJ = "MgSO4 rejection [%]"
_NACL_REJ = "NaCl rejection [%]"
_PH = "pH"
_TEMP = "Temperature [°C]"
_PRESSURE = "Pressure [bar]"
_PWP = "PWP [LMH/bar]"
_MONO_CAT_FEED = "Monovalent cation feed [mmol/L]"
_DIV_CAT_FEED = "Divalent cation feed [mmol/L]"
_MONO_AN_FEED = "Monovalent anion feed [mmol/L]"
_DIV_AN_FEED = "Divalent anion feed [mmol/L]"
_FEED_TYPE = "feed_type"   # derived categorical column from groups.py

# All 14 raw measurement columns in their canonical order (matches the CSV).
ALL_RAW_COLUMNS: tuple[str, ...] = (
    _MWCO,
    _ROUGHNESS,
    _ZETA,
    _CONTACT_ANGLE,
    _MGSO4_REJ,
    _NACL_REJ,
    _PH,
    _TEMP,
    _PRESSURE,
    _PWP,
    _MONO_CAT_FEED,
    _DIV_CAT_FEED,
    _MONO_AN_FEED,
    _DIV_AN_FEED,
)

# ---------------------------------------------------------------------------
# Feature-set identifiers
# ---------------------------------------------------------------------------

FeatureSetID = Literal[
    "FS1", "FS2", "FS3", "FS4", "FS5", "FS6", "FS7", "FS2_NO_PH"
]

ALL_FEATURE_SET_IDS: tuple[FeatureSetID, ...] = (
    "FS1", "FS2", "FS3", "FS4", "FS5", "FS6", "FS7", "FS2_NO_PH"
)


# ---------------------------------------------------------------------------
# Feature-set definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureSetDefinition:
    """Immutable record for one named feature set.

    Attributes
    ----------
    feature_set_id:
        Short identifier used throughout the pipeline (e.g. ``"FS2"``).
    name:
        Human-readable name for tables and figures.
    columns:
        Ordered tuple of column names.  Order is preserved in ``select_features``
        so downstream code can rely on column position if needed.
    description:
        One-line scientific rationale (traceability, rules §7.4).
    """

    feature_set_id: str
    name: str
    columns: tuple[str, ...]
    description: str


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# FS2 base definition — reused to derive FS6 and FS2_NO_PH without duplication.
_FS2_COLUMNS: tuple[str, ...] = (
    _ZETA, _PH, _PRESSURE, _PWP, _MONO_AN_FEED
)

FEATURE_SETS: dict[str, FeatureSetDefinition] = {
    "FS1": FeatureSetDefinition(
        feature_set_id="FS1",
        name="Full",
        columns=ALL_RAW_COLUMNS,
        description="All 14 raw measurement columns; maximal information, maximal collinearity.",
    ),
    "FS2": FeatureSetDefinition(
        feature_set_id="FS2",
        name="Source-selected",
        columns=_FS2_COLUMNS,
        description=(
            "Five features selected in the source paper: "
            "ζ, pH, pressure, PWP, monovalent anion feed. "
            "Core of the pH↔ζ identifiability demo (AGENTS.md §8)."
        ),
    ),
    "FS3": FeatureSetDefinition(
        feature_set_id="FS3",
        name="Process-only",
        columns=(_PH, _PRESSURE, _TEMP, _FEED_TYPE),
        description=(
            "Operating parameters only: pH, pressure, temperature, feed_type. "
            "No membrane-specific properties. Encodes feed_type as categorical."
        ),
    ),
    "FS4": FeatureSetDefinition(
        feature_set_id="FS4",
        name="Membrane-only",
        columns=(_MWCO, _ROUGHNESS, _ZETA, _CONTACT_ANGLE, _MGSO4_REJ, _NACL_REJ, _PWP),
        description=(
            "Seven membrane-characterisation features: MWCO, roughness, ζ, "
            "contact angle, MgSO4 rejection, NaCl rejection, PWP."
        ),
    ),
    "FS5": FeatureSetDefinition(
        feature_set_id="FS5",
        name="Parsimonious",
        columns=(_PH, _PRESSURE, _FEED_TYPE, _MWCO),
        description=(
            "Minimal interpretable set: pH, pressure, feed_type, MWCO. "
            "Uses raw MWCO [Da] rather than derived membrane_id (same information). "
            "Encodes feed_type as categorical."
        ),
    ),
    "FS6": FeatureSetDefinition(
        feature_set_id="FS6",
        name="No-zeta",
        columns=tuple(c for c in _FS2_COLUMNS if c != _ZETA),
        description=(
            "FS2 minus ζ. Drop-one-of-aliased-pair: tests whether pH importance "
            "increases when ζ is removed (AGENTS.md §8, Demo 2)."
        ),
    ),
    "FS7": FeatureSetDefinition(
        feature_set_id="FS7",
        name="No-membrane-props",
        columns=(
            _PH, _TEMP, _PRESSURE, _FEED_TYPE,
            _MONO_CAT_FEED, _DIV_CAT_FEED, _MONO_AN_FEED, _DIV_AN_FEED,
        ),
        description=(
            "Process and feed concentrations only; no membrane-specific properties. "
            "Encodes feed_type as categorical."
        ),
    ),
    "FS2_NO_PH": FeatureSetDefinition(
        feature_set_id="FS2_NO_PH",
        name="Source-selected no-pH",
        columns=tuple(c for c in _FS2_COLUMNS if c != _PH),
        description=(
            "FS2 minus pH. Symmetric counterpart of FS6: tests whether ζ importance "
            "increases when pH is removed (AGENTS.md §8, pH↔ζ migration test)."
        ),
    ),
}


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------

class FeatureSetError(KeyError):
    """Raised when a feature set ID is unknown or its columns are missing."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_definition(feature_set_id: str) -> FeatureSetDefinition:
    """Return the :class:`FeatureSetDefinition` for ``feature_set_id``.

    Parameters
    ----------
    feature_set_id:
        One of the IDs in :data:`ALL_FEATURE_SET_IDS`.

    Raises
    ------
    FeatureSetError
        If ``feature_set_id`` is not in the registry.
    """
    if feature_set_id not in FEATURE_SETS:
        raise FeatureSetError(
            f"Unknown feature set '{feature_set_id}'. "
            f"Valid IDs: {sorted(FEATURE_SETS)}."
        )
    return FEATURE_SETS[feature_set_id]


def get_columns(feature_set_id: str) -> tuple[str, ...]:
    """Return the ordered column tuple for ``feature_set_id``.

    Parameters
    ----------
    feature_set_id:
        One of the IDs in :data:`ALL_FEATURE_SET_IDS`.

    Raises
    ------
    FeatureSetError
        If ``feature_set_id`` is not in the registry.
    """
    return get_definition(feature_set_id).columns


def validate_columns(feature_set_id: str, df: pd.DataFrame) -> None:
    """Assert that every column in ``feature_set_id`` exists in ``df``.

    Fails loudly with all missing column names listed together rather than
    one-at-a-time, so a single validation pass surfaces every problem
    (rules §10).  Silent column dropping is explicitly forbidden.

    Parameters
    ----------
    feature_set_id:
        One of the IDs in :data:`ALL_FEATURE_SET_IDS`.
    df:
        DataFrame to validate against (typically ``seed_with_groups.csv``).

    Raises
    ------
    FeatureSetError
        If any column is absent from ``df``.
    """
    defn = get_definition(feature_set_id)
    available = set(df.columns)
    missing = [c for c in defn.columns if c not in available]
    if missing:
        raise FeatureSetError(
            f"Feature set '{feature_set_id}' ({defn.name}) references column(s) "
            f"not present in the DataFrame:\n  missing: {missing}\n"
            f"  available: {sorted(available)}"
        )


def select_features(feature_set_id: str, df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` restricted to the columns of ``feature_set_id``.

    Parameters
    ----------
    feature_set_id:
        One of the IDs in :data:`ALL_FEATURE_SET_IDS`.
    df:
        Grouped dataset (e.g. loaded from ``seed_with_groups.csv``). Must
        contain all columns declared for the feature set.

    Returns
    -------
    pandas.DataFrame
        A copy with exactly the feature columns, in their canonical order.
        The target column and all group-ID columns are excluded.

    Raises
    ------
    FeatureSetError
        If ``feature_set_id`` is unknown or any required column is absent.
    """
    validate_columns(feature_set_id, df)
    columns = get_columns(feature_set_id)

    # Guard: target and group-ID columns must never appear in a feature set.
    _NON_FEATURE_COLS = {
        TARGET, "row_id", "condition_id", "replicate_group_id",
        "domain_id", "is_duplicate_removed",
    }
    leaked = [c for c in columns if c in _NON_FEATURE_COLS]
    if leaked:
        raise FeatureSetError(
            f"Feature set '{feature_set_id}' contains non-feature column(s): "
            f"{leaked}. Target and group-ID columns must not appear in X."
        )

    return df[list(columns)].copy()
