"""Central configuration loading for the Seed/VFA reanalysis.

Purpose
-------
Parse ``config.yaml`` (AGENTS.md §3) into typed, immutable structures so that
paths, the global seed, and the de-duplicated data contract are read from a
single source of truth rather than hardcoded in scripts (rules §7.1, §9).

Inputs / outputs
----------------
``load_config`` takes an optional path to the YAML file and returns a frozen
``Config``. All filesystem paths are resolved relative to the project root
(the directory that contains ``config.yaml``) so scripts can be launched from
anywhere.

Invariants
----------
The ``DataContract`` mirrors AGENTS.md §4 exactly; it carries the expected row,
condition, membrane and feed-type counts that ``groups.validate_contract``
enforces. This module only *loads* the contract — it does not validate data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# The project root is two parents above this file: src/seed_vfa/config.py.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class Paths:
    """Resolved (absolute) filesystem paths used by the pipeline."""

    raw_dataset: Path
    processed_dir: Path
    seed_clean: Path
    seed_with_groups: Path
    results_dir: Path


@dataclass(frozen=True)
class DataContract:
    """Expected invariants of the de-duplicated dataset (AGENTS.md §4).

    These are correctness conditions, not tunable parameters. They are enforced
    by ``groups.validate_contract`` and independently re-checked by the test
    suite (AGENTS.md §11).
    """

    raw_n_rows: int
    n_exact_duplicates: int
    n_rows: int
    n_conditions: int
    n_membranes: int
    n_feed_types: int
    membrane_mwco: tuple[int, ...]
    feed_types: tuple[str, ...]


@dataclass(frozen=True)
class NoiseFloorConfig:
    """Bootstrap settings for the noise-floor computation (AGENTS.md §5)."""

    n_bootstrap: int
    ci_level: float
    floor_min: float
    floor_max: float


@dataclass(frozen=True)
class SplitsConfig:
    """Parameters for the five validation protocols (AGENTS.md §6)."""

    test_size: float    # fraction held out for protocols A and B
    n_shuffle_splits: int   # number of repeated random splits for protocol B
    gkf_n_splits: int   # k for GroupKFold in protocol C


@dataclass(frozen=True)
class ModelsConfig:
    """Hyperparameters for the diagnostic model set (AGENTS.md §7).

    These are fixed reference values, not tuned results. The models are
    diagnostic tools whose scientific role is to demonstrate protocol effects,
    not to maximise benchmark performance.
    """

    ridge_alpha: float
    catboost_iterations: int
    catboost_depth: int
    catboost_learning_rate: float
    random_forest_n_estimators: int


@dataclass(frozen=True)
class Config:
    """Top-level configuration object loaded from ``config.yaml``."""

    seed: int
    paths: Paths
    data_contract: DataContract
    noise_floor: NoiseFloorConfig
    splits: SplitsConfig
    models: ModelsConfig


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    """Return ``mapping[key]`` or raise a clear error naming the missing key."""
    if key not in mapping:
        raise KeyError(f"Missing required config key '{key}' in {context}.")
    return mapping[key]


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate ``config.yaml`` into a frozen :class:`Config`.

    Parameters
    ----------
    path:
        Path to the YAML configuration file. Defaults to the project-root
        ``config.yaml``.

    Returns
    -------
    Config
        Fully populated configuration with all paths resolved to absolute paths
        relative to the directory containing the config file.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    KeyError
        If a required key is absent (fail loudly rather than defaulting).
    """
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        raw: dict[str, Any] = yaml.safe_load(handle) or {}

    root = config_path.parent

    paths_raw = _require(raw, "paths", "config")
    paths = Paths(
        raw_dataset=(root / _require(paths_raw, "raw_dataset", "paths")).resolve(),
        processed_dir=(root / _require(paths_raw, "processed_dir", "paths")).resolve(),
        seed_clean=(root / _require(paths_raw, "seed_clean", "paths")).resolve(),
        seed_with_groups=(
            root / _require(paths_raw, "seed_with_groups", "paths")
        ).resolve(),
        results_dir=(root / _require(paths_raw, "results_dir", "paths")).resolve(),
    )

    contract_raw = _require(raw, "data_contract", "config")
    data_contract = DataContract(
        raw_n_rows=int(_require(contract_raw, "raw_n_rows", "data_contract")),
        n_exact_duplicates=int(
            _require(contract_raw, "n_exact_duplicates", "data_contract")
        ),
        n_rows=int(_require(contract_raw, "n_rows", "data_contract")),
        n_conditions=int(_require(contract_raw, "n_conditions", "data_contract")),
        n_membranes=int(_require(contract_raw, "n_membranes", "data_contract")),
        n_feed_types=int(_require(contract_raw, "n_feed_types", "data_contract")),
        membrane_mwco=tuple(
            int(v) for v in _require(contract_raw, "membrane_mwco", "data_contract")
        ),
        feed_types=tuple(
            str(v) for v in _require(contract_raw, "feed_types", "data_contract")
        ),
    )

    nf_raw = _require(raw, "noise_floor", "config")
    noise_floor_cfg = NoiseFloorConfig(
        n_bootstrap=int(_require(nf_raw, "n_bootstrap", "noise_floor")),
        ci_level=float(_require(nf_raw, "ci_level", "noise_floor")),
        floor_min=float(_require(nf_raw, "floor_min", "noise_floor")),
        floor_max=float(_require(nf_raw, "floor_max", "noise_floor")),
    )

    sp_raw = _require(raw, "splits", "config")
    splits_cfg = SplitsConfig(
        test_size=float(_require(sp_raw, "test_size", "splits")),
        n_shuffle_splits=int(_require(sp_raw, "n_shuffle_splits", "splits")),
        gkf_n_splits=int(_require(sp_raw, "gkf_n_splits", "splits")),
    )

    m_raw = _require(raw, "models", "config")
    models_cfg = ModelsConfig(
        ridge_alpha=float(_require(m_raw, "ridge_alpha", "models")),
        catboost_iterations=int(_require(m_raw, "catboost_iterations", "models")),
        catboost_depth=int(_require(m_raw, "catboost_depth", "models")),
        catboost_learning_rate=float(_require(m_raw, "catboost_learning_rate", "models")),
        random_forest_n_estimators=int(
            _require(m_raw, "random_forest_n_estimators", "models")
        ),
    )

    return Config(
        seed=int(_require(raw, "seed", "config")),
        paths=paths,
        data_contract=data_contract,
        noise_floor=noise_floor_cfg,
        splits=splits_cfg,
        models=models_cfg,
    )
