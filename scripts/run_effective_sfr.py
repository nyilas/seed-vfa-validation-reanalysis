#!/usr/bin/env python3
"""Experiment Exp-D (F4) — Effective sample-to-feature ratio analysis.

Must be run *after* ``run_data_audit.py``.

Pipeline (AGENTS.md §10, Exp-D)
---------------------------------
1. Load config and grouped dataset.
2. Compute dataset-level summary: nominal N (pre-dedup), distinct N (post-dedup),
   n_conditions (46 unique experimental conditions), n_domains (4 membrane types).
3. Compute SFR table for all primary feature sets.
4. Save results/tables/effective_sfr.csv.
5. Print a summary table.

Scientific rationale (AGENTS.md §0 / §10)
-------------------------------------------
The source paper reports results on what appears to be 80 independent rows.
In reality the dataset contains:
- 1 exact duplicate (removed; 79 distinct rows remain).
- Multiple technical replicates sharing the same condition_id (same membrane,
  pH, pressure, temperature, feed-ion vector): only 46 distinct conditions.
- All rows belonging to exactly 4 membrane types (MWCO domains).

The effective sample-to-feature ratio (SFR) depends on which of these levels
we consider "independent":
    nominal_sfr          = distinct_N / n_features         (79 rows)
    effective_sfr_cond   = n_conditions / n_features       (46 conditions)
    effective_sfr_domain = n_domains / n_features          (4 membranes)

At the domain level, every feature set with more than 4 columns has
effective_sfr_domain < 1 — the membrane-generalisation learning problem is
severely underdetermined regardless of which features are used.

Definition of done (Exp-D): effective_sfr.csv populated; Fig. 6 generated.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from seed_vfa.config import load_config           # noqa: E402
from seed_vfa.data import save_dataframe           # noqa: E402
from seed_vfa.diagnostics import (                 # noqa: E402
    SFR_TABLE_COLUMNS,
    compute_dataset_summary,
    compute_sfr_table,
)

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Primary feature sets included in the SFR table (AGENTS.md §8).
# FS2_NO_PH is a demo-only variant; it is excluded to keep the table
# focused on the primary analytical sets.
SFR_FEATURE_SETS: tuple[str, ...] = (
    "FS1", "FS2", "FS3", "FS4", "FS5", "FS6", "FS7",
)


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_dataset_summary(summary) -> None:
    print()
    print("  Dataset summary")
    print("  -----------------------------------------------")
    print(f"  Nominal rows (pre-dedup)      : {summary.nominal_N}")
    print(f"  Distinct rows (post-dedup)    : {summary.distinct_N}")
    print(f"  Duplicate rows removed        : {summary.nominal_N - summary.distinct_N}")
    print(f"  Distinct conditions           : {summary.n_conditions}")
    print(f"  Membrane domains (MWCO)       : {summary.n_domains}")


def _print_sfr_table(sfr_df: pd.DataFrame) -> None:
    print()
    print(
        f"  {'FS':<10} {'p':>2}  "
        f"{'nom_sfr':>8}  {'eff_sfr(cond)':>14}  {'eff_sfr(dom)':>12}"
    )
    print("  " + "-" * 56)
    for _, row in sfr_df.iterrows():
        flag_cond = " *" if row["effective_sfr_condition"] < 5.0 else "  "
        flag_dom  = " **" if row["effective_sfr_domain"] < 1.0 else "   "
        print(
            f"  {row['feature_set']:<10} {int(row['n_features']):>2}  "
            f"{row['nominal_sfr']:8.2f}  "
            f"{row['effective_sfr_condition']:14.2f}{flag_cond}  "
            f"{row['effective_sfr_domain']:12.2f}{flag_dom}"
        )
    print()
    print("  * effective_sfr_condition < 5  (rule-of-thumb low for regression)")
    print("  ** effective_sfr_domain < 1    (severely underdetermined at domain level)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the effective SFR analysis. Returns exit code (0 = success)."""
    print("[1/4] Loading config and grouped dataset...")
    config = load_config()

    grouped_path = config.paths.seed_with_groups
    if not grouped_path.is_file():
        print(
            f"ERROR: {grouped_path} not found — "
            "run scripts/run_data_audit.py first.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(grouped_path)
    print(f"      {len(df)} rows loaded | seed = {config.seed}")

    print("[2/4] Computing dataset summary...")
    summary = compute_dataset_summary(
        df, nominal_n_raw=config.data_contract.raw_n_rows
    )
    _print_dataset_summary(summary)

    print(f"[3/4] Computing SFR table for {len(SFR_FEATURE_SETS)} feature sets...")
    sfr_df = compute_sfr_table(
        df,
        SFR_FEATURE_SETS,
        nominal_n_raw=config.data_contract.raw_n_rows,
    )

    tables_dir = config.paths.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_path = tables_dir / "effective_sfr.csv"
    save_dataframe(sfr_df, out_path)
    print(f"[4/4] Saved {len(sfr_df)} rows → {out_path.name}")

    _print_sfr_table(sfr_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
