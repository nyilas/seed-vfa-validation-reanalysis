#!/usr/bin/env python3
"""Experiment Exp-A (F1), noise-floor half: compute floor, CI, and sidebar.

Must be run *after* ``run_data_audit.py`` (which writes
``data/processed/seed_with_groups.csv``).

Pipeline (AGENTS.md §10, Exp-A)
-------------------------------
1. Load ``config.yaml`` and ``seed_with_groups.csv``.
2. Compute the replicate-derived RMSE noise floor (AGENTS.md §5 formula).
3. Compute a percentile bootstrap CI over replicate groups.
4. Save ``results/tables/noise_floor.csv``.
5. Compute the published-numbers sidebar (§5 leakage-check sanity test).
6. Save ``results/tables/published_numbers_sidebar.csv``.
7. Print a scannable summary.

Definition of done (Exp-A): floor computed with CI; sidebar reproduces the
≈0.21 < 0.23 flag for the source paper's best augmented result.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd  # noqa: E402

from seed_vfa import noise_floor as nf  # noqa: E402
from seed_vfa.config import load_config  # noqa: E402
from seed_vfa.data import TARGET, save_dataframe  # noqa: E402
from seed_vfa.groups import REPLICATE_GROUP_ID  # noqa: E402

# Published-numbers sidebar (AGENTS.md §5).
# Source paper: Galiano et al. (or equivalent cited source in the manuscript).
# These numbers are treated as fixed input data, not tunable parameters.
# std_test ≈ 0.835 is back-calculated from RMSE = std * sqrt(1 − R²) for the
# baseline model: 0.283 = std * sqrt(1 − 0.885) → std ≈ 0.835.
_SIDEBAR_ENTRIES: list[dict[str, object]] = [
    {
        "source": "source_paper_baseline",
        "R2": 0.885,
        "RMSE_reported": 0.283,
        "std_test": 0.835,
        "note": "Random 75/25 split, baseline model (source paper)",
    },
    {
        "source": "source_paper_best_augmented",
        "R2": 0.937,
        "RMSE_reported": None,   # not explicitly given; implied only
        "std_test": 0.835,
        "note": "Random 75/25 split, best augmented model (source paper)",
    },
]


def _compute_sidebar(rmse_floor: float) -> pd.DataFrame:
    """Build the published-numbers sidebar DataFrame (AGENTS.md §5)."""
    rows: list[dict[str, object]] = []
    for entry in _SIDEBAR_ENTRIES:
        r2 = float(entry["R2"])  # type: ignore[arg-type]
        std = float(entry["std_test"])  # type: ignore[arg-type]
        rmse_implied = nf.rmse_from_r2_and_std(r2, std)
        rows.append(
            {
                "source": entry["source"],
                "R2": r2,
                "std_test": std,
                "RMSE_implied": round(rmse_implied, 6),
                "RMSE_floor": round(rmse_floor, 6),
                "below_floor": nf.flag_below_floor(rmse_implied, rmse_floor),
                "note": entry["note"],
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    """Run the noise-floor computation. Returns a process exit code (0 = success)."""
    print("[1/5] Loading configuration and grouped dataset...")
    config = load_config()
    nf_cfg = config.noise_floor

    grouped_path = config.paths.seed_with_groups
    if not grouped_path.is_file():
        print(
            f"ERROR: {grouped_path} not found. "
            "Run scripts/run_data_audit.py first.",
            file=sys.stderr,
        )
        return 1

    df = pd.read_csv(grouped_path)
    print(f"      loaded {len(df)} rows from {grouped_path.name}")

    print(
        f"[2/5] Computing noise floor "
        f"(bootstrap n={nf_cfg.n_bootstrap}, seed={config.seed})..."
    )
    result = nf.compute_floor(
        df,
        target_col=TARGET,
        group_col=REPLICATE_GROUP_ID,
        n_bootstrap=nf_cfg.n_bootstrap,
        ci_level=nf_cfg.ci_level,
        seed=config.seed,
    )
    ci = result.ci
    ci_pct = int(round(ci.ci_level * 100))
    print(
        f"      RMSE_floor = {result.rmse_floor:.6f}  "
        f"{ci_pct}% CI [{ci.lower:.6f}, {ci.upper:.6f}]"
    )
    print(
        f"      n_groups_used={result.n_groups_used}, "
        f"n_obs_used={result.n_obs_used}"
    )

    print("[3/5] Saving noise_floor.csv...")
    tables_dir = config.paths.results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    floor_csv = tables_dir / "noise_floor.csv"
    save_dataframe(pd.DataFrame([nf.to_record(result)]), floor_csv)
    print(f"      wrote: {floor_csv}")

    print("[4/5] Computing published-numbers sidebar (AGENTS.md §5)...")
    sidebar = _compute_sidebar(result.rmse_floor)
    for _, row in sidebar.iterrows():
        flag = "BELOW FLOOR" if row["below_floor"] else "above floor"
        print(
            f"      {row['source']}: R²={row['R2']}, "
            f"RMSE_implied={row['RMSE_implied']:.4f} → {flag}"
        )

    sidebar_csv = tables_dir / "published_numbers_sidebar.csv"
    save_dataframe(sidebar, sidebar_csv)
    print(f"      wrote: {sidebar_csv}")

    print("[5/5] Done.")
    print()
    print("Noise floor summary")
    print("-------------------")
    print(f"  RMSE_floor:       {result.rmse_floor:.6f}")
    print(f"  {ci_pct}% CI:          [{ci.lower:.6f}, {ci.upper:.6f}]")
    print(f"  n_groups_used:    {result.n_groups_used}")
    print(f"  n_obs_used:       {result.n_obs_used}")
    print(f"  best-augmented R²=0.937 RMSE_implied below floor: "
          f"{sidebar.loc[sidebar['source']=='source_paper_best_augmented', 'below_floor'].item()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
