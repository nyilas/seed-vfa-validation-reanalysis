#!/usr/bin/env python3
"""Table D: per-held-out-membrane breakdown of protocol-D (LOMO) results.

Reads  : results/predictions/all_predictions.csv
Writes : results/tables/table_D_per_membrane.csv
         results/figures/fig_extrapolation_by_membrane.png
         paper/figures/fig_extrapolation_by_membrane.pdf

Reporting rules (EXPECTED_RESULTS.md §9)
-----------------------------------------
Do NOT headline the fold-averaged R² from model_metrics.csv for protocol D.
The fold-mean R² for ridge/FS1 is ≈ −47 ± 80 — fragile because it averages
over four very unequal folds, one of which is catastrophic.  Instead report:

  1. This per-held-out-membrane table (primary headline result for LOMO).
  2. The pooled RMSE over all protocol-D test predictions (single summary number).

Expected results (EXPECTED_RESULTS.md §9, verified 2026-06-30)
-----------------------------------------------------------------
  ridge/FS1 held-out membrane 300  :  RMSE ≈ 9.64, R² ≈ −186, bias ≈ +2.99
  ridge/FS1 held-out membrane 400  :  RMSE ≈ 0.32, R² ≈ +0.78
  ridge/FS3 (all membranes)        :  RMSE < 0.65  (no explosion)
  catboost/FS3 (all membranes)     :  RMSE < 0.55
  random_forest/FS1 (all membranes):  RMSE < 0.70

Membrane 300 has the largest held-out test set (44 rows / 55 % of the 79-row
dataset), which is why leaving it out leaves the model severely underfitted.
The large positive bias (+2.99) means ridge/FS1 systematically under-predicts
the VFA retention for this membrane domain.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import from plotting first so matplotlib backend is set before pyplot is imported
from seed_vfa.plotting import apply_style, fig_lomo_per_membrane  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402 (backend already set)
import pandas as pd

from seed_vfa.config import load_config
from seed_vfa.data import save_dataframe
from seed_vfa.evaluation import TABLE_D_COLUMNS, compute_lomo_per_membrane

_DPI_PNG = 300


def _derive_references(tables_dir: Path) -> tuple[float, float]:
    """Return (trivial_rmse, noise_floor) from pre-computed CSVs."""
    nf = pd.read_csv(tables_dir / "noise_floor.csv")
    mm = pd.read_csv(tables_dir / "model_metrics.csv")
    noise_floor = float(nf["RMSE_floor"].iloc[0])
    trivial_rmse = float(
        mm[(mm["model"] == "dummy") & (mm["protocol"] == "B")]["RMSE_mean"].values[0]
    )
    return trivial_rmse, noise_floor


def _print_summary(table_d: pd.DataFrame, trivial_rmse: float, noise_floor: float) -> None:
    print()
    print("  Per-held-out-membrane RMSE (Protocol D)")
    print(f"  {'model':<14} {'FS':<5} {'membrane':>9} {'n':>4}  "
          f"{'RMSE':>7}  {'R2':>8}  {'bias':>7}")
    print("  " + "-" * 62)
    for _, row in table_d.sort_values(["model", "feature_set", "held_out_membrane"]).iterrows():
        flag = " ← COLLAPSE" if row["RMSE"] > trivial_rmse * 1.5 else ""
        r2_s = f"{row['R2']:8.3f}" if not (row['R2'] != row['R2']) else "     nan"
        print(
            f"  {row['model']:<14} {row['feature_set']:<5} {row['held_out_membrane']:>9}  "
            f"{int(row['n']):>4}  {row['RMSE']:>7.3f}  {r2_s}  "
            f"{row['bias']:>7.3f}{flag}"
        )

    print()
    # Pooled RMSE over all D predictions (single summary number)
    import numpy as np
    preds_path = PROJECT_ROOT / "results" / "predictions" / "all_predictions.csv"
    preds = pd.read_csv(preds_path)
    d = preds[preds["protocol"] == "D"]
    for model in ["ridge", "catboost", "random_forest"]:
        for fs in ["FS1", "FS2", "FS3", "FS5"]:
            sub = d[(d["model"] == model) & (d["feature_set"] == fs)]
            if sub.empty:
                continue
            pooled_rmse = float(
                (((sub["y_true"] - sub["y_pred"]) ** 2).mean()) ** 0.5
            )
            below = "← BELOW FLOOR" if pooled_rmse < noise_floor else ""
            if model == "ridge" and fs == "FS1":
                print(f"  Pooled RMSE D | {model}/{fs}: {pooled_rmse:.4f} {below}")
    print()
    print(f"  Reference lines:  trivial RMSE = {trivial_rmse:.4f} | "
          f"noise floor = {noise_floor:.4f}")


def main() -> int:
    """Run the per-membrane breakdown. Returns exit code (0 = success)."""
    apply_style()   # set rcParams once; must precede any plt.subplots() call
    print("[1/4] Loading config...")
    config = load_config()
    tables_dir = config.paths.results_dir / "tables"
    fig_dir_res = config.paths.results_dir / "figures"
    fig_dir_paper = PROJECT_ROOT / "paper" / "figures"

    preds_path = config.paths.results_dir / "predictions" / "all_predictions.csv"
    if not preds_path.is_file():
        print(f"ERROR: {preds_path} not found — run run_protocol_sweep.py first.",
              file=sys.stderr)
        return 1

    for p in [tables_dir, fig_dir_res, fig_dir_paper]:
        p.mkdir(parents=True, exist_ok=True)

    required = ["noise_floor.csv", "model_metrics.csv"]
    for fname in required:
        if not (tables_dir / fname).is_file():
            print(f"ERROR: {tables_dir / fname} not found — "
                  "run run_noise_floor.py and run_protocol_sweep.py first.",
                  file=sys.stderr)
            return 1

    print("[2/4] Computing per-membrane metrics...")
    preds = pd.read_csv(preds_path)
    table_d = compute_lomo_per_membrane(preds)
    # Enforce column order
    table_d = table_d[list(TABLE_D_COLUMNS)]

    out_path = tables_dir / "table_D_per_membrane.csv"
    save_dataframe(table_d, out_path)
    print(f"      {len(table_d)} rows → {out_path.name}")

    trivial_rmse, noise_floor = _derive_references(tables_dir)
    print(f"      trivial RMSE = {trivial_rmse:.4f} | noise floor = {noise_floor:.4f}")

    print("[3/4] Generating figure...")
    fig = fig_lomo_per_membrane(
        table_d, trivial_rmse=trivial_rmse, noise_floor=noise_floor
    )
    stem = "fig_extrapolation_by_membrane"
    fig.savefig(fig_dir_res / f"{stem}.png", dpi=_DPI_PNG, bbox_inches="tight")
    fig.savefig(fig_dir_paper / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"      PNG → {fig_dir_res / stem}.png")
    print(f"      PDF → {fig_dir_paper / stem}.pdf")

    print("[4/4] Done.")
    _print_summary(table_d, trivial_rmse, noise_floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
