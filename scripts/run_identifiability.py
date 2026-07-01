#!/usr/bin/env python3
"""Experiment Exp-C (F3) — Identifiability: pH / ζ importance migration.

Must be run *after* ``run_data_audit.py``.

Pipeline (AGENTS.md §10, Exp-C)
---------------------------------
1. Load config and grouped dataset.
2. Generate Protocol C splits (GroupKFold by condition_id, 5 folds).
   Honest interpolation prevents replicate leakage from inflating importance.
3. For each (feature set, method, fold):
   - Fit the appropriate model on the training fold only.
   - Extract fold-level importances.
4. Aggregate across folds: importance_mean, importance_std, rank_mean.
5. Compute pairwise Kendall τ across methods per feature set;
   write rank_kendall_tau_vs_other_methods into the table.
6. Save results/tables/importance_stability.csv.
7. Print a summary of the pH / ζ migration effect (Demo 2).

Scientific rationale (AGENTS.md §10, progetto §3 Demo 2)
----------------------------------------------------------
pH and ζ are physically coupled (ζ reflects surface charge, which depends on
pH) and statistically aliased in this dataset (r ≈ −0.61).  Four importance
methods should agree on which feature matters more; instead they give
inconsistent rankings of the aliased pair.  Removing one member
(FS6 = no-ζ; FS2_NO_PH = no-pH) causes the surviving member's importance to
migrate upward — demonstrating that the ranking is an artifact of which
aliased variable was retained, not of the underlying physics.

Feature sets used
-----------------
- FS2 : ζ, pH, pressure, PWP, monovalent anion feed  (both members of pair)
- FS6 : FS2 minus ζ                                   (only pH survives)
- FS2_NO_PH: FS2 minus pH                              (only ζ survives)

Methods (AGENTS.md §7)
-----------------------
- standardized_coef : |Ridge coefficient| in StandardScaler feature space
- permutation       : mean R² decrease under feature shuffling (Ridge)
- shap              : mean |SHAP value| over test samples (CatBoost)
- gpr_ard           : 1 / ARD length-scale, diagnostic probe only (GPR)

GPR is NOT presented as a contribution: it illustrates that even a
Bayesian nonparametric model with separate length-scales per feature is
confused by the aliasing (AGENTS.md §0).

Definition of done (Exp-C): importance_stability.csv complete; pH and/or ζ
importance increases after the other is removed; Kendall τ < 1.0 for FS2
on the aliased pair; rank of the surviving member rises in FS6 and FS2_NO_PH.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from seed_vfa.config import load_config           # noqa: E402
from seed_vfa.data import save_dataframe           # noqa: E402
from seed_vfa.importance import (                  # noqa: E402
    ALL_IMPORTANCE_METHODS,
    IMPORTANCE_STABILITY_COLUMNS,
    PH_COL,
    SHAP_AVAILABLE,
    ZETA_COL,
    add_kendall_tau,
    aggregate_fold_importances,
    compute_fold_importance,
)
from seed_vfa.splits import protocol_c             # noqa: E402

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

# Feature sets for Demo 2 (AGENTS.md §10 Exp-C).
IDENTIFIABILITY_FEATURE_SETS: tuple[str, ...] = ("FS2", "FS6", "FS2_NO_PH")

# Protocol used for honest importance estimation.
IDENTIFIABILITY_PROTOCOL: str = "C"


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_migration_summary(stability_df: pd.DataFrame) -> None:
    """Print the pH / ζ importance migration table (Demo 2 key result)."""
    print()
    print("  pH / ζ importance migration (normalized mean, rank)")
    print(f"  {'Method':<20}  {'FS2 pH':>8} {'FS2 ζ':>8}  "
          f"{'FS6 pH':>8}  {'FS2NP ζ':>8}")
    print("  " + "-" * 64)

    df = stability_df.copy()

    for method in ALL_IMPORTANCE_METHODS:
        m_rows = df[df["method"] == method]
        if m_rows.empty:
            continue

        def _get(fs: str, feat: str, col: str = "importance_mean") -> str:
            sub = m_rows[(m_rows["feature_set"] == fs) & (m_rows["feature"] == feat)]
            if sub.empty:
                return "  -   "
            val = sub[col].iloc[0]
            return f"{val:8.4f}"

        def _rank(fs: str, feat: str) -> str:
            sub = m_rows[(m_rows["feature_set"] == fs) & (m_rows["feature"] == feat)]
            if sub.empty:
                return "  -"
            val = sub["rank_mean"].iloc[0]
            return f"#{val:.1f}"

        fs2_ph = _get("FS2", PH_COL) + _rank("FS2", PH_COL)
        fs2_ze = _get("FS2", ZETA_COL) + _rank("FS2", ZETA_COL)
        fs6_ph = _get("FS6", PH_COL) + _rank("FS6", PH_COL)
        fs2np_ze = _get("FS2_NO_PH", ZETA_COL) + _rank("FS2_NO_PH", ZETA_COL)
        print(f"  {method:<20}  {fs2_ph}  {fs2_ze}  {fs6_ph}  {fs2np_ze}")

    print()
    print("  Kendall τ (mean pairwise, FS2 only)")
    for method in ALL_IMPORTANCE_METHODS:
        sub = stability_df[
            (stability_df["feature_set"] == "FS2") & (stability_df["method"] == method)
        ]
        if sub.empty:
            continue
        tau = sub["rank_kendall_tau_vs_other_methods"].iloc[0]
        if pd.isna(tau):
            print(f"    {method:<22}: n/a (only one method available)")
        else:
            print(f"    {method:<22}: τ = {tau:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the identifiability experiment. Returns exit code (0 = success)."""
    print("[1/6] Loading config and grouped dataset...")
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
    print(f"      {len(df)} rows | seed = {config.seed}")
    if not SHAP_AVAILABLE:
        print("      WARNING: shap not installed — 'shap' method will be skipped.")

    print(f"[2/6] Generating Protocol {IDENTIFIABILITY_PROTOCOL} splits...")
    splits = protocol_c(df, n_splits=config.splits.gkf_n_splits)
    print(f"      {len(splits)} folds (GroupKFold by condition_id)")

    print(
        f"[3/6] Computing importances: "
        f"{len(IDENTIFIABILITY_FEATURE_SETS)} feature sets × "
        f"{len(ALL_IMPORTANCE_METHODS)} methods × "
        f"{len(splits)} folds..."
    )

    tables_dir = config.paths.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Accumulate all aggregated rows.
    agg_frames: list[pd.DataFrame] = []

    total = len(IDENTIFIABILITY_FEATURE_SETS) * len(ALL_IMPORTANCE_METHODS)
    combo_idx = 0

    for fs_id in IDENTIFIABILITY_FEATURE_SETS:
        for method in ALL_IMPORTANCE_METHODS:
            combo_idx += 1

            if method == "shap" and not SHAP_AVAILABLE:
                print(f"  [{combo_idx:2d}/{total}] {fs_id} | {method:<20}  SKIP (shap not installed)")
                continue

            fold_results = []
            for sr in splits:
                fi = compute_fold_importance(method, df, sr, fs_id, config=config)
                if fi is not None:
                    fold_results.append(fi)

            if not fold_results:
                print(f"  [{combo_idx:2d}/{total}] {fs_id} | {method:<20}  SKIP (no results)")
                continue

            agg = aggregate_fold_importances(
                fold_results, protocol=IDENTIFIABILITY_PROTOCOL
            )
            agg_frames.append(agg)
            print(
                f"  [{combo_idx:2d}/{total}] {fs_id} | {method:<20}  "
                f"({len(fold_results)} folds)"
            )

    if not agg_frames:
        print("ERROR: no importance results computed.", file=sys.stderr)
        return 1

    print("[4/6] Computing pairwise Kendall τ across methods...")
    stability_df = pd.concat(agg_frames, ignore_index=True)
    stability_df = add_kendall_tau(stability_df)

    # Enforce exact schema column order.
    stability_df = stability_df[list(IMPORTANCE_STABILITY_COLUMNS)]

    print("[5/6] Saving importance_stability.csv...")
    out_path = tables_dir / "importance_stability.csv"
    save_dataframe(stability_df, out_path)
    print(f"      {len(stability_df)} rows → {out_path.name}")

    print("[6/6] Experiment complete.")
    _print_migration_summary(stability_df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
