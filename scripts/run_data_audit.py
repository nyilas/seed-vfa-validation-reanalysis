#!/usr/bin/env python3
"""Experiment Exp-A (F1), data-audit half: de-duplicate and build groups.

Pipeline (AGENTS.md §10, Exp-A)
-------------------------------
1. Load ``config.yaml``.
2. Load ``data/raw/Seed_Dataset.csv`` (validate raw schema and row count).
3. Remove the single known exact-duplicate row (80 -> 79) and record it.
4. Write ``data/processed/seed_clean.csv``.
5. Build the §4 derived grouping columns.
6. Validate the §4 data contract (79 / 46 / 4 / 2); fail loudly on violation.
7. Write ``data/processed/seed_with_groups.csv``.

This script is fully deterministic and does not fit, scale, split, or model
anything (AGENTS.md §1, invariant 3). It exits non-zero if any contract
invariant is violated.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the src/ layout importable when run as a plain script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from seed_vfa import data, groups  # noqa: E402
from seed_vfa.config import load_config  # noqa: E402


def main() -> int:
    """Run the data audit. Returns a process exit code (0 on success)."""
    print("[1/6] Loading configuration (config.yaml)...")
    config = load_config()
    contract = config.data_contract

    print(f"[2/6] Loading raw dataset: {config.paths.raw_dataset}")
    raw = data.load_raw(
        config.paths.raw_dataset, expected_n_rows=contract.raw_n_rows
    )
    print(f"      raw rows={len(raw)}, columns={raw.shape[1]}")

    print("[3/6] Removing exact-duplicate row(s) before any split...")
    dedup = data.deduplicate(raw, expected_n_removed=contract.n_exact_duplicates)
    print(
        f"      removed {dedup.n_removed} row(s) at raw index "
        f"{list(dedup.removed_raw_index)}; {len(raw)} -> {len(dedup.clean)} rows"
    )

    clean_path = data.save_dataframe(dedup.clean, config.paths.seed_clean)
    print(f"      wrote de-duplicated data: {clean_path}")

    print("[4/6] Building derived grouping columns (AGENTS.md §4)...")
    grouped = groups.build_groups(dedup)
    print(
        f"      rows={len(grouped)}, "
        f"conditions={grouped[groups.CONDITION_ID].nunique()}, "
        f"membranes={grouped[groups.MEMBRANE_ID].nunique()}, "
        f"feed_types={grouped[groups.FEED_TYPE].nunique()}"
    )

    print("[5/6] Validating data contract (79 / 46 / 4 / 2)...")
    groups.validate_contract(grouped, contract)
    print("      contract OK.")

    print("[6/6] Writing grouped dataset...")
    groups_path = data.save_dataframe(grouped, config.paths.seed_with_groups)
    print(f"      wrote: {groups_path}")

    # Compact, scannable summary (rules §8.5).
    membrane_counts = (
        grouped[groups.MEMBRANE_ID].value_counts().sort_index().to_dict()
    )
    feed_counts = grouped[groups.FEED_TYPE].value_counts().to_dict()
    print("\nData audit summary")
    print("------------------")
    print(f"  rows (deduplicated):  {len(grouped)}")
    print(f"  conditions:           {grouped[groups.CONDITION_ID].nunique()}")
    print(f"  replicate groups:     {grouped[groups.REPLICATE_GROUP_ID].nunique()}")
    print(f"  membranes (by MWCO):  {membrane_counts}")
    print(f"  feed types:           {feed_counts}")
    print("  data audit PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
