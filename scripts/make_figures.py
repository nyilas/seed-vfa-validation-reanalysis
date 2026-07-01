#!/usr/bin/env python3
"""Generate all paper figures from pre-computed result CSVs (AGENTS.md §10).

Must be run *after* all experiment scripts:
    run_data_audit.py, run_noise_floor.py, run_protocol_sweep.py,
    run_identifiability.py, run_effective_sfr.py, run_table_d.py

Reads (from results/tables/ and results/predictions/)
------------------------------------------------------
- noise_floor.csv
- published_numbers_sidebar.csv
- model_metrics.csv
- importance_stability.csv
- effective_sfr.csv
- table_D_per_membrane.csv   (optional — skipped if absent)
- all_predictions.csv        (for unique-row dataset audit figure)

Writes
------
- results/figures/<stem>.pdf   (vector, primary deliverable)
- results/figures/<stem>.png   (300 dpi preview)
- paper/figures/<stem>.pdf     (copy for manuscript)

No experiments are re-run here: all figures are derived solely from the
pre-computed CSV files (reproducibility, AGENTS.md §1 invariant 5).

Reference values derived from CSVs (never hardcoded)
-----------------------------------------------------
- trivial_rmse  : RMSE_mean of DummyRegressor under Protocol B (100 splits).
- noise_floor   : RMSE_floor from noise_floor.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import from plotting first — this triggers matplotlib.use("Agg") at module load
# and must precede any import of matplotlib.pyplot.
from seed_vfa.plotting import (  # noqa: E402
    apply_style,
    save,
    fig_dataset_audit,
    fig_effective_sfr,
    fig_extrapolation,
    fig_importance,
    fig_lomo_per_membrane,
    fig_noise_floor,
    fig_protocol_sweep,
)

import matplotlib.pyplot as plt  # noqa: E402 (backend already set)
import pandas as pd               # noqa: E402

_DPI_PNG = 300


def _publish(fig: plt.Figure, stem: str, paper_fig_dir: Path) -> None:
    """Write PDF to paper/figures/, then call save() for results/figures/."""
    paper_fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(paper_fig_dir / f"{stem}.pdf", bbox_inches="tight")
    save(fig, stem)   # writes results/figures/{stem}.pdf + .png, closes figure


def _load_tables(tables_dir: Path) -> dict[str, pd.DataFrame]:
    """Load required result CSVs; raise FileNotFoundError with a clear message."""
    required = [
        "noise_floor",
        "published_numbers_sidebar",
        "model_metrics",
        "importance_stability",
        "effective_sfr",
    ]
    tables: dict[str, pd.DataFrame] = {}
    for name in required:
        path = tables_dir / f"{name}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Required table not found: {path}\n"
                "Run the corresponding experiment script first."
            )
        tables[name] = pd.read_csv(path)
    return tables


def _unique_rows(preds_df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique dataset observation (de-duplicated by row_id)."""
    return (
        preds_df
        .drop_duplicates("row_id")
        [["row_id", "condition_id", "replicate_group_id",
          "membrane_id", "feed_type", "y_true"]]
        .reset_index(drop=True)
    )


def main() -> int:
    """Generate Figures 1–6 (and the per-membrane LOMO figure). Returns 0."""
    apply_style()   # set rcParams once; must precede any plt.subplots() call

    from seed_vfa.config import load_config
    config = load_config()

    tables_dir    = config.paths.results_dir / "tables"
    preds_dir     = config.paths.results_dir / "predictions"
    paper_fig_dir = PROJECT_ROOT / "paper" / "figures"

    print("[1/3] Loading result CSVs...")
    try:
        tables = _load_tables(tables_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    preds_path = preds_dir / "all_predictions.csv"
    if not preds_path.is_file():
        print(f"ERROR: {preds_path} not found — run run_protocol_sweep.py first.",
              file=sys.stderr)
        return 1

    print("      Loading all_predictions.csv...")
    preds_df = pd.read_csv(preds_path)

    noise_floor_val = float(tables["noise_floor"]["RMSE_floor"].iloc[0])
    trivial_rmse = float(
        tables["model_metrics"][
            (tables["model_metrics"]["model"] == "dummy") &
            (tables["model_metrics"]["protocol"] == "B")
        ]["RMSE_mean"].values[0]
    )
    unique_rows = _unique_rows(preds_df)

    print(f"      Trivial RMSE (dummy/B): {trivial_rmse:.4f}")
    print(f"      Noise floor:            {noise_floor_val:.4f}")
    print(f"      Unique observations:    {len(unique_rows)}")

    print("[2/3] Generating figures 1–6...")

    # ── Fig. 1 — Noise floor ────────────────────────────────────────────────
    print("  Fig. 1 — noise floor...")
    _publish(
        fig_noise_floor(
            tables["noise_floor"],
            tables["published_numbers_sidebar"],
            trivial_rmse=trivial_rmse,
        ),
        "fig1_noise_floor", paper_fig_dir,
    )

    # ── Fig. 2 — Dataset audit ──────────────────────────────────────────────
    print("  Fig. 2 — dataset audit...")
    _publish(fig_dataset_audit(unique_rows), "fig2_dataset_audit", paper_fig_dir)

    # ── Fig. 3 — Protocol sweep RMSE ────────────────────────────────────────
    print("  Fig. 3 — protocol sweep RMSE...")
    _publish(
        fig_protocol_sweep(
            tables["model_metrics"],
            feature_set="FS2",
            trivial_rmse=trivial_rmse,
            noise_floor=noise_floor_val,
        ),
        "fig3_protocol_sweep", paper_fig_dir,
    )

    # ── Fig. 4 — Extrapolation contrast ─────────────────────────────────────
    print("  Fig. 4 — extrapolation contrast...")
    _publish(
        fig_extrapolation(
            tables["model_metrics"],
            trivial_rmse=trivial_rmse,
            noise_floor=noise_floor_val,
        ),
        "fig4_extrapolation", paper_fig_dir,
    )

    # ── Fig. 5 — Importance migration ───────────────────────────────────────
    print("  Fig. 5 — importance migration...")
    _publish(
        fig_importance(tables["importance_stability"]),
        "fig5_importance", paper_fig_dir,
    )

    # ── Fig. 6 — Effective SFR ──────────────────────────────────────────────
    print("  Fig. 6 — effective SFR...")
    _publish(
        fig_effective_sfr(tables["effective_sfr"]),
        "fig6_effective_sfr", paper_fig_dir,
    )

    # ── Fig. extra — per-membrane LOMO breakdown ────────────────────────────
    table_d_path = tables_dir / "table_D_per_membrane.csv"
    if table_d_path.is_file():
        print("  Fig. extra — per-membrane LOMO...")
        table_d = pd.read_csv(table_d_path)
        _publish(
            fig_lomo_per_membrane(
                table_d,
                trivial_rmse=trivial_rmse,
                noise_floor=noise_floor_val,
            ),
            "fig_extrapolation_by_membrane", paper_fig_dir,
        )
    else:
        print("  Fig. extra — skipped (run scripts/run_table_d.py first)")

    print("[3/3] All figures saved.")
    print()
    from seed_vfa.plotting import _FIG_DIR
    print(f"  PDF + PNG: {_FIG_DIR}/")
    print(f"  Paper PDF: {paper_fig_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
