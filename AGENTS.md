# AGENTS.md — Seed/VFA validation-aware reanalysis

Executable spec for the IDE development agent. The scientific rationale lives in
`progetto_paper_seed_vfa-v0_3.md`; **this file governs implementation**. If a scientific
decision is unclear, defer to that document. If an implementation detail is unclear, ask
before guessing — do not silently introduce assumptions that could create leakage.

> Scope lock: implement only what serves **Demo 1** (performance vs protocol), **Demo 2**
> (identifiability), and the **noise floor**. Do **not** implement active learning, synthetic
> augmentation, or GPR uncertainty/active-learning loops in v1. They are explicitly out of scope.

---

## 0. Mission

Reanalyze `Seed_Dataset.csv` (80 rows, 14 features, target `Cret total VFAs`) to show that, in a
small structured experimental dataset, (a) predictive performance is an artifact of the
cross-validation protocol, and (b) feature importance is an artifact of design identifiability —
both measured against a replicate-derived noise floor. Produce reproducible tables, figures, and a
documented dataset release.

---

## 1. Hard invariants (NON-NEGOTIABLE)

These are correctness conditions, not style preferences. Every pipeline must satisfy all of them;
the test suite (§10) enforces them. A violation is a bug even if results "look fine".

1. **De-duplicate first.** Remove the exact duplicate row (the known CSV duplicate, ~rows 80–81)
   before any split. 80 → 79 rows. Record which row was removed.
2. **Never split replicates across train/test.** All rows sharing a `replicate_group_id` must fall
   in the same fold for every grouped protocol. Random protocols are the *only* place replicates
   may be separated, and that separation is the object of study, not an accident.
3. **No information from test into train, ever.** Feature scaling, feature selection (incl. any
   Pearson/RFE replication of the source paper), imputation, and hyperparameter tuning happen
   **inside the training fold only**, via `sklearn.pipeline.Pipeline` + nested CV. No global
   `fit` on the full dataset before splitting.
4. **Test sets contain only real data.** No synthetic rows anywhere in v1 (synthetic is out of
   scope), but the invariant stands as a guard.
5. **Determinism.** Every stochastic step takes an explicit seed from `config.yaml`. Re-running a
   script reproduces identical metrics bit-for-bit (within float tolerance 1e-9 for deterministic
   models; fixed seed for RF/CatBoost).
6. **Save predictions fold-by-fold.** Aggregate metrics are derived from per-row predictions, never
   computed and discarded. Schema in §9.
7. **Report RMSE and MAE alongside R².** R² alone is misleading near the noise floor; the noise
   floor is defined in RMSE units.
8. **Two reference lines on every performance figure:** trivial-model RMSE (≈0.72) and replicate
   noise floor (≈0.23, recomputed exactly in F1).

### Figure conventions (enforced)

All figures go through `src/seed_vfa/plotting.py`, which is the **single source of truth** for
style. Scripts call `apply_style()` once at startup and must not set `plt.rcParams` or
`plt.style.use` locally.

* **Format**: PDF (vector, primary deliverable) + PNG (300 dpi preview).  Both written to
  `results/figures/`; PDF also copied to `paper/figures/`.
* **Venue bundle**: `tueplots.bundles.tmlr2023` — TMLR text width 6.5 in (469.755 pt).
  Exposed as `WIDTH_FULL = 6.5` and `WIDTH_HALF = 3.25` in `plotting.py`.
* **LaTeX fonts**: pdflatex/pgf path; falls back to mathtext with an explicit warning.
* **Palette**: Okabe–Ito with a fixed semantic mapping (`MODEL_COLORS`, `PROTOCOL_COLORS`,
  `MEMBRANE_COLORS`) so each entity has the same colour across all figures.
* **Titles**: none on axes (`set_title`/`suptitle` forbidden). Descriptions go in the LaTeX
  caption.
* **Reference lines**: every performance figure uses `REF_TRIVIAL` and `REF_NOISE_FLOOR` kwargs
  from `plotting.py` — consistent style, never overridden per figure.

---

## 2. Environment

```
python>=3.11
numpy, pandas, scipy, scikit-learn, statsmodels
matplotlib            # final figures (no seaborn for final figs)
tueplots              # venue figure sizes + bundles (tmlr2023)
catboost              # reference model only
shap                  # Demo 2, optional method
joblib, tqdm, pyyaml
pytest, ruff, black
```

System LaTeX toolchain (required for embedded fonts in PDFs):
```
texlive-latex-base  texlive-latex-extra  texlive-fonts-recommended
cm-super  dvipng  ghostscript
```
Without pdflatex, `apply_style()` falls back to mathtext (Computer Modern) and
emits a `logging.warning` — figures render but fonts will not match the manuscript.

GPR uses `sklearn.gaussian_process` (RBF/Matern + WhiteKernel + ARD) **only** as a probe in Demo 2
(its ARD length-scales are shown to be fooled by aliasing). It is not a contribution.

---

## 3. Repository layout

```
seed-vfa-validation-reanalysis/
├── AGENTS.md
├── progetto_paper_seed_vfa-v0_3.md
├── pyproject.toml / requirements.txt
├── config.yaml
├── data/
│   ├── raw/Seed_Dataset.csv            # read-only
│   └── processed/
│       ├── seed_clean.csv              # dedup'd
│       └── seed_with_groups.csv        # + derived columns (§4)
├── src/seed_vfa/
│   ├── config.py        data.py        groups.py
│   ├── splits.py        features.py    models.py
│   ├── evaluation.py    diagnostics.py importance.py
│   ├── noise_floor.py   plotting.py
├── scripts/
│   ├── run_data_audit.py            # F1
│   ├── run_noise_floor.py           # F1
│   ├── run_protocol_sweep.py        # F2 (Demo 1)
│   ├── run_identifiability.py       # F3 (Demo 2)
│   ├── run_effective_sfr.py         # F4
│   └── make_figures.py
├── results/{tables,figures,predictions,splits,logs}/
├── paper/{manuscript.tex,references.bib,figures,tables}/
└── tests/{test_groups.py,test_splits.py,test_no_leakage.py,test_metrics.py,test_noise_floor.py}
```

---

## 4. Data contract: derived columns

Built once by `groups.py`, written to `seed_with_groups.csv`. Construction rules:

| Column | Type | Rule |
|---|---|---|
| `row_id` | int | stable index after dedup |
| `membrane_id` | cat | one of {225, 250, 300, 400} by MWCO (counts 8/14/44/14) |
| `feed_type` | cat | `simple` if all feed-ion concentrations == 0 (first 32 rows), else `complex` |
| `condition_id` | cat | unique key = (membrane_id, pH, temperature, pressure, feed ion vector). 46 distinct |
| `replicate_group_id` | cat | rows sharing identical (condition_id) collapse to one group (≈ pairs) |
| `domain_id` | cat | alias of `membrane_id` (domain = membrane) |
| `is_duplicate_removed` | bool | True on the dropped exact-duplicate's twin record |

Validation: assert 79 rows, 46 `condition_id`, 4 `membrane_id`, 2 `feed_type`. Fail loudly otherwise.

---

## 5. Noise floor (`noise_floor.py`) — headline result

Definition: the irreducible RMSE estimated from within-replicate variability.

```
For each replicate_group g with n_g >= 2 observations of target y:
    within-group residual variance s2_g = var(y_g, ddof=1)
RMSE_floor = sqrt( weighted_mean_g( s2_g ) )   # weight by (n_g - 1)
```

Outputs: `results/tables/noise_floor.csv` with `RMSE_floor`, `n_groups_used`, `n_obs_used`, CI via
bootstrap over groups. Expected ≈ 0.23 — **verify, do not hardcode**.

Leakage check (the proof device): for any reported (R², test-target-std) pair, compute
`RMSE = std_test * sqrt(1 - R2)`. Flag `leakage_suspected = (RMSE < RMSE_floor)`. Apply this to:
(i) our own random-split results, (ii) the source paper's published aggregates as a sanity sidebar
(baseline R²=0.885/RMSE=0.283 → std_test≈0.835; best augmented R²=0.937 → RMSE≈0.21 < floor).
Report the sidebar factually as a property of the protocol, not as a refutation.

---

## 6. Validation protocols (`splits.py`)

| ID | Protocol | sklearn object | Unit | Question |
|---|---|---|---|---|
| A | Random 75/25 | `train_test_split` (repeated, many seeds) | row | replicate of source paper's setting |
| B | Repeated random | `ShuffleSplit(n_splits>=100)` | row | variance of the random estimate + leakage audit |
| C | Grouped-by-condition | `GroupKFold(groups=condition_id)` | condition | honest interpolation |
| D | Leave-one-membrane-out | `LeaveOneGroupOut(groups=membrane_id)` | domain | extrapolation to unseen membrane |
| E (opt) | Leave-one-feed-regime-out | `LeaveOneGroupOut(groups=feed_type)` | regime | transfer simple↔complex |

For B, also log per-split: condition overlap, replicate overlap, membrane overlap → `leakage_audit.csv`.

---

## 7. Models (`models.py`) — diagnostic, not a race

All wrapped in a `Pipeline` with `StandardScaler` (fit inside fold). Tuning via nested CV where applicable.

- `DummyRegressor(strategy="mean")` — trivial baseline (defines the 0.72 line).
- `Ridge` full feature set — collapses under extrapolation (the dramatic case).
- `Ridge` parsimonious — degrades gracefully.
- Process-only linear — pH, pressure, temperature, feed_type.
- `CatBoost` — reference, ties to source paper; tuned modestly inside CV; NOT the protagonist.
- `RandomForest` — non-linear control that does not extrapolate beyond seen range.

---

## 8. Feature sets (`features.py`)

| ID | Name | Features |
|---|---|---|
| FS1 | Full | all 14 |
| FS2 | Source-selected | ζ, pH, pressure, PWP, monovalent anion feed |
| FS3 | Process-only | pH, pressure, temperature, feed_type |
| FS4 | Membrane-only | MWCO, roughness, ζ, contact angle, MgSO4 rej, NaCl rej, PWP |
| FS5 | Parsimonious | pH, pressure, feed_type, (MWCO or membrane_id) |
| FS6 | No-zeta | FS2 minus ζ |
| FS7 | No-membrane-props | process + feed only |

FS2/FS6 are the core of Demo 2 (drop-one-of-aliased-pair). A symmetric `No-pH` variant of FS2 is
also required for the pH↔zeta migration test.

---

## 9. Output schemas

`results/predictions/all_predictions.csv`:
`row_id, condition_id, replicate_group_id, membrane_id, feed_type, protocol, split_id, fold, model,
feature_set, y_true, y_pred, residual`

`results/tables/model_metrics.csv`:
`protocol, model, feature_set, R2_mean, R2_std, RMSE_mean, RMSE_std, MAE_mean, MAE_std,
n_train, n_test, n_conditions_train, n_conditions_test, n_domains_train, n_domains_test, below_floor`

`results/tables/leakage_audit.csv`:
`split_id, protocol, condition_overlap, replicate_overlap, membrane_overlap, leakage_warning`

`results/tables/importance_stability.csv`:
`feature_set, method, protocol, feature, importance_mean, importance_std, rank_mean, rank_kendall_tau_vs_other_methods`

---

## 10. Experiments

### Exp-A — Audit + noise floor (F1)
`run_data_audit.py` then `run_noise_floor.py`. Asserts in §4 must pass. Produces `noise_floor.csv`
and the published-numbers sidebar. **Definition of done:** floor computed with CI; sidebar reproduces
the ≈0.21<0.23 flag; Fig. 1 and Fig. 2 generated.

### Exp-B — Protocol sweep (Demo 1, F2)
`run_protocol_sweep.py` over {A,B,C,D} × {models} × {FS1,FS2,FS3,FS5}. Compute leakage gap =
RMSE(C)−RMSE(B_mean); extrapolation gap = RMSE(D)−RMSE(C). **Done:** model_metrics.csv populated,
Fig. 3 and Fig. 4 generated, RF-vs-Ridge extrapolation contrast visible.

### Exp-C — Identifiability (Demo 2, F3)
`run_identifiability.py`: for FS2, FS6 (no-zeta), FS2-no-pH, compute importance via 4 methods
(standardized coef, permutation, SHAP on CatBoost, GPR-ARD length-scales). Show pH↔zeta importance
migration; compute importance-stability (Kendall τ across methods/folds, restricted to aliased pair).
**Done:** importance_stability.csv, Fig. 5.

### Exp-D — Effective SFR (F4)
`run_effective_sfr.py`: nominal N vs distinct vs conditions vs domains; nominal vs effective SFR per
feature set. **Done:** Fig. 6 + SFR table.

---

## 11. Acceptance tests (`pytest`, must stay green)

- `test_groups.py`: 79 rows / 46 conditions / 4 membranes / 2 feed regimes; duplicate removed.
- `test_no_leakage.py::test_replicates_never_split` — for C/D, no `replicate_group_id` spans folds.
- `test_no_leakage.py::test_no_global_fit` — scaler/selector are inside the Pipeline, not pre-fit.
- `test_no_leakage.py::test_no_synthetic_in_test` — test indices map to real rows only.
- `test_noise_floor.py::test_floor_positive_and_reasonable` — 0.1 < RMSE_floor < 0.4.
- `test_noise_floor.py::test_below_floor_flag` — flag fires iff RMSE < floor.
- `test_metrics.py::test_reproducible` — two runs, same seed, identical metrics.

---

## 12. Definition of done (whole project v1)

All tests green; `model_metrics.csv` complete for {A,B,C,D}; Figures 1–6 in `paper/figures/`;
`noise_floor.csv` with the published-numbers sidebar; `seed_with_groups.csv` released with a data
card (license check noted). No active-learning / synthetic / GPR-uncertainty code present. README
documents how to reproduce every figure from raw data with one command per figure.

## Validation gate
After every run, validate outputs against `EXPECTED_RESULTS.md`.
RED violations (structural) block progress and must be fixed before continuing.
AMBER (predictive out of range) must be investigated and explained, not ignored.
Run `python scripts/check_expected.py` and ensure it exits 0 before marking a phase done.