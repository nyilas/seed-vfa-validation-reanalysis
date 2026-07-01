# Seed/VFA Validation-Aware Reanalysis

Reanalysis of the Seed *et al.* volatile fatty acid (VFA) retention dataset
(80 rows, 14 membrane and process features, target: *Cret* total VFAs).
The study investigates how cross-validation protocol choice and feature-set
design affect reported model performance and feature importance — both
measured against a replicate-derived noise floor.

---

## Goal

Small structured experimental datasets have two properties that affect how
machine-learning results should be interpreted.

First, performance estimates depend on the validation protocol.
A random split treats all rows as independent, which inflates estimates when
rows share an experimental condition (replicates).
A grouped or leave-one-out split reveals the harder generalisation problem.
The protocol choice can shift R² by 0.2–0.5 on this dataset.

Second, feature importance is not uniquely identified when two features are
statistically aliased.
pH and ζ (zeta potential) are physically coupled (ζ reflects surface charge,
which depends on pH) and correlated at r ≈ −0.61 in this dataset.
Removing one raises the other's reported importance, regardless of which
importance method is used.
The reported ranking is therefore a property of which aliased variable was
retained, not of the physics.

Both effects are quantified relative to a noise floor estimated from
within-replicate variability: the irreducible RMSE below which no model can
improve further without overfitting to measurement noise.

---

## Validation protocols

| ID | Protocol | scikit-learn object | Independent unit | What it measures |
|----|----------|---------------------|------------------|------------------|
| A  | Random 75/25 | `train_test_split` | Row | Replicates the source-paper setting |
| B  | Repeated random | `ShuffleSplit(100)` | Row | Variance of the random estimate |
| C  | Grouped by condition | `GroupKFold(5)` | Condition (46 distinct) | Honest interpolation |
| D  | Leave-one-membrane-out | `LeaveOneGroupOut` | Membrane domain (4) | Extrapolation to unseen membrane |

**Protocol A/B** (random splits) may place replicate rows of the same
experimental condition into both train and test.
The model can then interpolate between nearly identical points, which
inflates R².
This is the setting used in the source paper.

**Protocol C** (grouped by condition) holds out all rows of each condition
together.
No replicate information leaks from test to train.
This estimates interpolation performance under honest data separation.

**Protocol D** (leave-one-membrane-out) holds out all rows from one membrane
type at a time.
The model must generalise to a membrane it has never seen.
This estimates extrapolation performance.
Ridge regression collapses on FS1 under Protocol D (R² = −47, RMSE = 3.0),
while process-only feature sets (FS3, FS5) degrade modestly.

---

## Noise floor

The noise floor is the RMSE estimable from within-replicate variability.
It is the lower bound below which no model can improve without fitting noise.

Formally:

```
For each replicate group g with n_g ≥ 2 observations:
    s²_g = sample variance of the target within group g
RMSE_floor = sqrt( Σ_g (n_g − 1) s²_g  /  Σ_g (n_g − 1) )
```

This reanalysis estimates RMSE_floor ≈ 0.229 g/L (95% CI: 0.170–0.283,
bootstrap over groups).

The noise floor serves as a reference line on every performance figure.
A model RMSE below the floor indicates either (a) the evaluation protocol
leaks replicate information from test to train, or (b) the dataset has
changed since the floor was estimated.

---

## Repository layout

```
seed-vfa-validation-reanalysis/
├── config.yaml                  # paths, seed, data contract, hyperparameters
├── data/
│   ├── raw/Seed_Dataset.csv     # read-only; original input
│   └── processed/
│       ├── seed_clean.csv       # de-duplicated (80 → 79 rows)
│       └── seed_with_groups.csv # + derived group columns
├── src/seed_vfa/
│   ├── config.py      data.py      groups.py
│   ├── splits.py      features.py  models.py
│   ├── evaluation.py  diagnostics.py
│   ├── importance.py  noise_floor.py  plotting.py
├── scripts/
│   ├── run_data_audit.py        # Exp-A: de-duplicate, assign groups
│   ├── run_noise_floor.py       # Exp-A: estimate RMSE floor with CI
│   ├── run_protocol_sweep.py    # Exp-B: Demo 1 — protocol × model × FS
│   ├── run_identifiability.py   # Exp-C: Demo 2 — pH/ζ importance migration
│   ├── run_effective_sfr.py     # Exp-D: nominal vs effective SFR
│   └── make_figures.py          # Figures 1–6
├── results/
│   ├── tables/                  # CSV outputs from each experiment
│   ├── predictions/             # fold-by-fold predictions
│   └── figures/                 # PNG figures for inspection
├── paper/figures/               # PDF figures for the manuscript
└── tests/                       # pytest suite (263 tests)
```

---

## Reproducing the pipeline

### 1. Create the environment

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -e ".[dev]"
```

This installs the `seed_vfa` package in editable mode together with
`numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `catboost`,
`shap`, `pyyaml`, and the dev tools `pytest`, `ruff`, `black`.
Python ≥ 3.11 is required.

### 3. Run the test suite

```bash
pytest
```

All 263 tests must pass before running any experiment.
The test suite enforces the hard invariants from `AGENTS.md`:
no global scaler fit, no replicate leakage across grouped folds,
deterministic reproducibility, and correct output schemas.

### 4. Run the experiments in order

Each script writes its outputs to `results/` and must be run before
any script that depends on its outputs.

```bash
# Exp-A — data audit and noise floor
python scripts/run_data_audit.py
python scripts/run_noise_floor.py

# Exp-B — protocol sweep (Demo 1: performance vs protocol)
python scripts/run_protocol_sweep.py

# Exp-C — identifiability (Demo 2: pH/ζ importance migration)
python scripts/run_identifiability.py

# Exp-D — effective sample-to-feature ratio
python scripts/run_effective_sfr.py

# Generate all figures
python scripts/make_figures.py
```

The full pipeline takes approximately 5–15 minutes on a standard laptop,
dominated by the 100-split protocol B sweep and the GPR ARD fits in Exp-C.

---

## Scripts

### `run_data_audit.py`
Loads `data/raw/Seed_Dataset.csv`, removes the single known exact-duplicate
row (80 → 79 rows), assigns derived group columns (`condition_id`,
`replicate_group_id`, `membrane_id`, `feed_type`), and validates the data
contract from `config.yaml`.
Writes `data/processed/seed_clean.csv` and `data/processed/seed_with_groups.csv`.

### `run_noise_floor.py`
Estimates the replicate noise floor from within-condition variability using
weighted pooling of within-group variances.
Computes a 95% bootstrap confidence interval over replicate groups.
Writes `results/tables/noise_floor.csv` and `results/tables/published_numbers_sidebar.csv`.

### `run_protocol_sweep.py`
Runs protocols A, B, C, D over 4 models (Dummy, Ridge, CatBoost, Random Forest)
and 4 feature sets (FS1, FS2, FS3, FS5) — 64 combinations, up to 1760 fold
evaluations total.
Each pipeline is cloned and fitted inside the training fold only.
Writes `results/predictions/all_predictions.csv`,
`results/tables/model_metrics.csv`, and `results/tables/leakage_audit.csv`.

### `run_identifiability.py`
For feature sets FS2 (pH + ζ), FS6 (FS2 − ζ), and FS2_NO_PH (FS2 − pH),
estimates feature importance using four methods:
standardised Ridge coefficients, permutation importance, SHAP values
(CatBoost), and GPR ARD length-scales (diagnostic probe only).
Computes pairwise Kendall τ across methods to quantify ranking agreement.
Writes `results/tables/importance_stability.csv`.

### `run_effective_sfr.py`
Computes the nominal and effective sample-to-feature ratio (SFR) for each
primary feature set at three aggregation levels: distinct rows (79),
experimental conditions (46), and membrane domains (4).
Writes `results/tables/effective_sfr.csv`.

### `make_figures.py`
Reads all result CSVs and generates Figures 1–6.
Does not recompute any experiment.
Writes PNG files to `results/figures/` (300 dpi, for inspection) and PDF
files to `paper/figures/` (vector, for the manuscript).

---

## Outputs

### Tables (`results/tables/`)

| File | Contents |
|------|----------|
| `noise_floor.csv` | RMSE floor, 95% CI, bootstrap parameters |
| `published_numbers_sidebar.csv` | Source-paper RMSE values vs noise floor |
| `model_metrics.csv` | R², RMSE, MAE per (protocol, model, feature set) |
| `leakage_audit.csv` | Per-split condition/replicate/membrane overlap |
| `importance_stability.csv` | Feature importance and Kendall τ per method |
| `effective_sfr.csv` | Nominal and effective SFR per feature set |

### Predictions (`results/predictions/`)

`all_predictions.csv` — one row per (row_id, protocol, split, fold, model,
feature set), with columns:
`row_id, condition_id, replicate_group_id, membrane_id, feed_type, protocol,
split_id, fold, model, feature_set, y_true, y_pred, residual`.

### Figures

| Figure | File stem | Contents |
|--------|-----------|----------|
| 1 | `fig1_noise_floor` | Noise floor CI; source-paper RMSE comparison |
| 2 | `fig2_dataset_audit` | Target distribution; strip plot by membrane |
| 3 | `fig3_protocol_sweep` | RMSE by protocol for all models (FS2) |
| 4 | `fig4_extrapolation` | Protocol C vs D RMSE across models and feature sets |
| 5 | `fig5_importance` | pH/ζ importance migration across FS2/FS6/FS2_NO_PH |
| 6 | `fig6_effective_sfr` | Nominal vs effective SFR per feature set |

PNG copies are in `results/figures/`; PDF copies are in `paper/figures/`.
Every performance figure (Figs. 1, 3, 4) includes a trivial-model RMSE
reference line and the replicate noise floor reference line.

---

## Configuration

All paths, the global RNG seed, data-contract invariants, and model
hyperparameters are in `config.yaml`.
Scripts read settings from this file; nothing is hardcoded.
To reproduce with a different seed, change `seed: 42` and re-run.

---

## Out of scope (v1)

The following are explicitly excluded from this version:

- **Active learning** — no acquisition functions, no iterative query strategies.
- **Synthetic data augmentation** — no SMOTE, GANs, or physics-based generation.
- **GPR as a predictive model** — GPR appears only as a diagnostic probe in
  Exp-C to illustrate that ARD length-scales are affected by aliasing.
- **Hyperparameter optimisation** — models use fixed reference hyperparameters
  from `config.yaml`; nested cross-validation is not run in v1.
- **Protocol E** (leave-one-feed-regime-out) — defined in `splits.py` but not
  included in the sweep.
- **Statistical significance testing** of performance differences across protocols.

---

## Licence

See [LICENSE](LICENSE).
