#!/usr/bin/env python3
"""Experiment Exp-B (F2) — Protocol sweep over {A,B,C,D} × {models} × {FS1,FS2,FS3,FS5}.

Must be run *after* ``run_data_audit.py`` and ``run_noise_floor.py``.

Pipeline (AGENTS.md §10, Exp-B)
---------------------------------
1. Load config, noise floor from ``noise_floor.csv``, and grouped dataset.
2. Generate all protocol splits for A (single 75/25), B (repeated ShuffleSplit),
   C (GroupKFold by condition_id), and D (LOGO by membrane_id).
3. Build the leakage audit from all splits (especially B's repeated random splits).
4. Run the sweep: for every (protocol, model, feature_set) combination,
   clone and fit the pipeline inside each training fold only, predict the test
   fold, collect per-row predictions and fold-level metrics.
5. Save fold-by-fold predictions → results/predictions/all_predictions.csv.
6. Save aggregated model metrics → results/tables/model_metrics.csv.
7. Print a summary showing the leakage gap and extrapolation gap.

AGENTS.md §1 invariants enforced here
---------------------------------------
- (1) De-duplication already done; script reads ``seed_with_groups.csv``.
- (2) Grouped protocols (C, D) never split replicate_group_id — enforced by the
  split generators in ``splits.py``.
- (3) Each pipeline is cloned and fit anew inside each training fold; no scaler
  or selector is fitted on the full dataset.
- (5) All randomness is seeded from ``config.seed``; re-running reproduces
  identical output bit-for-bit.
- (6) Fold-by-fold predictions are accumulated before aggregation; nothing is
  discarded.

Definition of done (Exp-B): model_metrics.csv populated for {A,B,C,D};
leakage gap = RMSE_mean(C) − RMSE_mean(B) clearly positive; extrapolation gap
= RMSE_mean(D) − RMSE_mean(C) positive; RF-vs-Ridge extrapolation contrast
visible in the Ridge+FS2 rows.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np           # noqa: E402
import pandas as pd          # noqa: E402
from sklearn.base import clone  # noqa: E402

from seed_vfa.config import load_config      # noqa: E402
from seed_vfa.data import TARGET, save_dataframe  # noqa: E402
from seed_vfa.evaluation import (            # noqa: E402
    aggregate_metrics,
    evaluate_fold,
    metrics_to_record,
)
from seed_vfa.features import get_columns, select_features  # noqa: E402
from seed_vfa.models import ALL_MODEL_IDS, build_pipeline   # noqa: E402
from seed_vfa.splits import (               # noqa: E402
    SplitResult,
    build_leakage_audit,
    protocol_a,
    protocol_b,
    protocol_c,
    protocol_d,
)

# ---------------------------------------------------------------------------
# Sweep scope (Exp-B, AGENTS.md §10)
# ---------------------------------------------------------------------------

# Protocols A–D (E is optional and not part of Exp-B).
SWEEP_PROTOCOLS: tuple[str, ...] = ("A", "B", "C", "D")

# All four diagnostic models defined in AGENTS.md §7.
SWEEP_MODELS: tuple[str, ...] = ALL_MODEL_IDS  # dummy, ridge, catboost, random_forest

# Feature sets selected for Exp-B (AGENTS.md §10).
SWEEP_FEATURE_SETS: tuple[str, ...] = ("FS1", "FS2", "FS3", "FS5")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_rmse_floor(tables_dir: Path) -> float:
    """Read the pre-computed RMSE floor from ``noise_floor.csv``."""
    floor_csv = tables_dir / "noise_floor.csv"
    if not floor_csv.is_file():
        raise FileNotFoundError(
            f"{floor_csv} not found — run scripts/run_noise_floor.py first."
        )
    return float(pd.read_csv(floor_csv).loc[0, "RMSE_floor"])


def _generate_protocol_splits(
    df: pd.DataFrame, config
) -> dict[str, list[SplitResult]]:
    """Generate all protocol splits from config parameters (AGENTS.md §6)."""
    return {
        "A": protocol_a(df, test_size=config.splits.test_size, seed=config.seed),
        "B": protocol_b(
            df,
            n_splits=config.splits.n_shuffle_splits,
            test_size=config.splits.test_size,
            seed=config.seed,
        ),
        "C": protocol_c(df, n_splits=config.splits.gkf_n_splits),
        "D": protocol_d(df),
    }


def _run_combination(
    df: pd.DataFrame,
    splits: list[SplitResult],
    model_id: str,
    fs_id: str,
    *,
    config,
    rmse_floor: float,
) -> tuple[list[pd.DataFrame], dict[str, object]]:
    """Fit, predict, and evaluate one (model, feature_set) combination over all folds.

    Returns
    -------
    fold_predictions:
        One predictions DataFrame per fold (§9 schema).
    metrics_record:
        Flat dict for the model_metrics.csv row.
    """
    feature_cols = get_columns(fs_id)
    pipeline_template = build_pipeline(
        model_id, feature_cols, seed=config.seed, models_config=config.models
    )

    fold_preds: list[pd.DataFrame] = []
    fold_metrics = []

    for sr in splits:
        pipe = clone(pipeline_template)
        X_train = select_features(fs_id, df.iloc[sr.train_idx])
        y_train = df.iloc[sr.train_idx][TARGET].to_numpy(dtype=float)
        X_test = select_features(fs_id, df.iloc[sr.test_idx])
        pipe.fit(X_train, y_train)
        y_pred: np.ndarray = pipe.predict(X_test)
        preds_df, fm = evaluate_fold(
            df, sr, y_pred, model=model_id, feature_set=fs_id
        )
        fold_preds.append(preds_df)
        fold_metrics.append(fm)

    agg = aggregate_metrics(
        fold_metrics,
        protocol=splits[0].protocol,
        model=model_id,
        feature_set=fs_id,
        rmse_floor=rmse_floor,
    )
    return fold_preds, metrics_to_record(agg)


def _print_summary(metrics_df: pd.DataFrame, rmse_floor: float) -> None:
    """Print a compact Demo-1 summary table for Ridge + FS2."""
    ref = metrics_df[
        (metrics_df["model"] == "ridge") & (metrics_df["feature_set"] == "FS2")
    ].set_index("protocol")

    print()
    print("  Ridge + FS2  (Demo 1 reference)")
    print(f"  {'Proto':<6} {'RMSE_mean':>9} {'±RMSE_std':>9} {'R2_mean':>8}  flag")
    print("  " + "-" * 48)
    for proto in ["A", "B", "C", "D"]:
        if proto not in ref.index:
            continue
        row = ref.loc[proto]
        flag = " ← BELOW FLOOR" if row["below_floor"] else ""
        print(
            f"  {proto:<6} {row['RMSE_mean']:9.4f} {row['RMSE_std']:9.4f}"
            f" {row['R2_mean']:8.4f}{flag}"
        )

    print(f"  {'noise floor':.<6} {rmse_floor:9.4f}")
    print()

    b_rmse = ref.loc["B", "RMSE_mean"] if "B" in ref.index else None
    c_rmse = ref.loc["C", "RMSE_mean"] if "C" in ref.index else None
    d_rmse = ref.loc["D", "RMSE_mean"] if "D" in ref.index else None

    if b_rmse is not None and c_rmse is not None:
        print(f"  Leakage gap     RMSE(C) − RMSE(B)  = {c_rmse - b_rmse:+.4f}")
    if c_rmse is not None and d_rmse is not None:
        print(f"  Extrapolation   RMSE(D) − RMSE(C)  = {d_rmse - c_rmse:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the protocol sweep. Returns a process exit code (0 = success)."""
    print("[1/6] Loading config, noise floor, and grouped dataset...")
    config = load_config()

    tables_dir = config.paths.results_dir / "tables"
    preds_dir = config.paths.results_dir / "predictions"
    tables_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)

    grouped_path = config.paths.seed_with_groups
    if not grouped_path.is_file():
        print(
            f"ERROR: {grouped_path} not found — "
            "run scripts/run_data_audit.py first.",
            file=sys.stderr,
        )
        return 1

    try:
        rmse_floor = _load_rmse_floor(tables_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    df = pd.read_csv(grouped_path)
    print(
        f"      {len(df)} rows | "
        f"RMSE_floor = {rmse_floor:.6f} | "
        f"seed = {config.seed}"
    )

    print("[2/6] Generating protocol splits...")
    protocol_splits = _generate_protocol_splits(df, config)
    for pid in SWEEP_PROTOCOLS:
        n = len(protocol_splits[pid])
        print(f"      {pid}: {n} split{'s' if n > 1 else ''}")

    print("[3/6] Building leakage audit...")
    all_splits_flat = [s for pid in SWEEP_PROTOCOLS for s in protocol_splits[pid]]
    leakage_df = build_leakage_audit(df, all_splits_flat)
    leakage_csv = tables_dir / "leakage_audit.csv"
    save_dataframe(leakage_df, leakage_csv)
    n_warnings = int(leakage_df["leakage_warning"].sum())
    print(
        f"      {len(leakage_df)} audit rows | "
        f"{n_warnings} leakage warnings → {leakage_csv.name}"
    )

    # Build the ordered list of (protocol, model, feature_set) combinations.
    combos: list[tuple[str, str, str]] = [
        (proto_id, model_id, fs_id)
        for proto_id in SWEEP_PROTOCOLS
        for model_id in SWEEP_MODELS
        for fs_id in SWEEP_FEATURE_SETS
    ]
    total_folds = sum(
        len(protocol_splits[pid]) for pid, _, _ in combos
    )
    print(
        f"[4/6] Running protocol sweep "
        f"({len(combos)} combinations, {total_folds} total fold evaluations)..."
    )

    all_preds: list[pd.DataFrame] = []
    all_metrics: list[dict[str, object]] = []

    for i, (proto_id, model_id, fs_id) in enumerate(combos, start=1):
        splits = protocol_splits[proto_id]
        n_folds = len(splits)
        fold_preds, metric_record = _run_combination(
            df, splits, model_id, fs_id,
            config=config, rmse_floor=rmse_floor,
        )
        all_preds.extend(fold_preds)
        all_metrics.append(metric_record)
        print(
            f"  [{i:2d}/{len(combos)}] "
            f"{proto_id} | {model_id:<14s} | {fs_id}  "
            f"({n_folds} fold{'s' if n_folds > 1 else ''})"
        )

    print("[5/6] Saving outputs...")

    predictions_df = pd.concat(all_preds, ignore_index=True)
    predictions_csv = preds_dir / "all_predictions.csv"
    save_dataframe(predictions_df, predictions_csv)
    print(f"      {len(predictions_df):,} prediction rows → {predictions_csv.name}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_csv = tables_dir / "model_metrics.csv"
    save_dataframe(metrics_df, metrics_csv)
    print(f"      {len(metrics_df)} metric rows → {metrics_csv.name}")

    print("[6/6] Sweep complete.")
    _print_summary(metrics_df, rmse_floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
