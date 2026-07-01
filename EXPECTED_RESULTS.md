# EXPECTED_RESULTS.md — verified reference rubric

Companion to `AGENTS.md`. Use this to decide, for every output file, whether a result
**confirms the hypothesis** or **smells like a bug**. All numbers below were verified on the real
`Seed_Dataset.csv` and on a clean reference run (the one reviewed on 2026-06-30). They are the
ground truth the agent must check its own outputs against.

**Tolerance philosophy.** Two classes of expectation:
- **Structural (must match tightly):** row/condition/membrane counts, group-overlap flags, noise
  floor, correlations. A mismatch here is almost always a bug.
- **Predictive (match by sign and order of magnitude only):** model R²/RMSE drift with seed, fold,
  and regularization. Do not assert exact equality; assert the *direction* and *rough range*.

A result that violates a structural expectation is **RED** (stop, fix). A predictive result outside
its range is **AMBER** (investigate before trusting). Matching = **GREEN**.

---

## 1. Ground truth — data structure (RED if violated)

Exact column names in the raw CSV (header), in order:
```
MWCO [Da], Average surface roughness[nm], Zeta potential [mV], Static contact angle [°],
MgSO4 rejection [%], NaCl rejection [%], pH, Temperature [°C], Pressure [bar], PWP [LMH/bar],
Monovalent cation feed [mmol/L], Divalent cation feed [mmol/L], Monovalent anion feed [mmol/L],
Divalent anion feed [mmol/L], Cret total VFAs
```
Target = `Cret total VFAs` (NOT `RVFA`; they are numerically equivalent but the column is named this).

| Quantity | Expected | Bug signal if different |
|---|---|---|
| Raw rows | 80 | wrong file |
| Exact duplicate rows | 1 (→ 79 distinct after dedup) | dedup logic wrong |
| Distinct rows after dedup | **79** | |
| `membrane_id` counts (MWCO) | 225:8, 250:13, 300:44, 400:14 | (note: 250 is **13** after dedup, not 14 — the duplicate was a 250 row) |
| `feed_type` | simple (all 4 feed-ion cols == 0): **32**, complex: **47** | feed_type rule wrong |
| **`condition_id` distinct** | **46** | If you get **78**, you wrongly included `PWP` in the key. **PWP MUST be excluded** from `condition_id` (it varies within a replicate pair). Cols = MWCO, pH, Temperature, Pressure, 4 feed-ion cols. |
| `replicate_group_id` distinct | 46 (33 groups of size 2, 13 singletons) | |
| Target mean / std / range | 2.65 / 0.725 / [1.35, 4.44] | |

---

## 2. Noise floor (`noise_floor.csv`) — RED if outside band

| Field | Expected | Bug signal |
|---|---|---|
| `RMSE_floor` | **0.229** (band 0.20–0.26) | <0.15 or >0.30 → replicate grouping or floor formula wrong |
| 95% CI | ≈ [0.17, 0.28] | |
| `n_groups_used` | 33 | should equal #groups with size ≥ 2 |
| `n_obs_used` | 66 | |

Formula: weighted (by `n_g − 1`) mean of within-replicate-group target variance, square-rooted.
**Never hardcode 0.229** — recompute; this row just says what the recompute should yield.

---

## 3. Source-paper sidebar (`published_numbers_sidebar.csv`)

| Source | R² | RMSE_implied | below_floor | Expected |
|---|---|---|---|---|
| baseline | 0.885 | ≈ 0.283 | **False** | above floor — honest-ish |
| best_augmented | 0.937 | ≈ 0.210 | **True** | **below floor → leakage red flag** |

This is the smoking gun: the augmented model implied RMSE (0.210) sits **below** the replicate noise
floor (0.229). Present as a property of the protocol, not a refutation. **Caveat to keep in the
text:** `std_test = 0.835` is inferred from the baseline aggregates and reused for the augmented row;
the flag holds unless the augmented test set had substantially larger variance. Suggestive, not
definitive, until per-point predictions are available.

---

## 4. Model metrics (`model_metrics.csv`) — AMBER ranges

Protocols: A = single random 75/25 (R²_std = 0); B = repeated random (ShuffleSplit); C =
GroupKFold by condition (honest interpolation); D = LeaveOneMembraneOut (extrapolation).

Reference run values (sign + order of magnitude; expect drift ±0.05–0.10 in R²):

| model / FS | A (rand) | B (rep rand) | C (grouped) | D (LOMO) | Reading |
|---|---|---|---|---|---|
| dummy / any | RMSE 0.89* | RMSE **0.728** | RMSE 0.725 | RMSE 0.734 | trivial baseline ≈ **0.72–0.73**; A's 0.89 is a single-split artifact, do **not** use it as "the baseline" |
| ridge / FS1 | 0.77 | 0.77 | 0.75 | **≪ 0, collapse** | full linear explodes under LOMO |
| ridge / FS3 (process) | 0.57 | 0.48 | 0.47 | **0.45** | nearly flat across protocols — degrades with grace |
| catboost / FS1 | 0.86 | 0.83 | 0.80 | 0.33 | drops but does not explode |
| catboost / FS3 | 0.77 | 0.70 | 0.67 | **0.55** | best LOMO model |
| random_forest / FS1 | 0.79 | 0.80 | 0.80 | 0.48 | trees don't extrapolate → survive |

`below_floor` for **all our models** must be **False**. Any `True` → leakage in *our* pipeline
(check scaling/selection are inside the fold).

Structural columns to verify per protocol:
- C: `n_conditions_train + n_conditions_test > 46` is fine? **No** — conditions are split, so per fold
  train+test conditions ≈ 46 with **no overlap**; membranes shared (overlap > 0).
- D: `n_domains_test = 1`, `n_domains_train = 3`, membrane overlap = 0.

---

## 5. Headline phenomena (GREEN = confirms thesis)

1. **Extrapolation collapse (Demo 1 core).** `ridge/FS1` under D must be strongly negative.
   Reference: fold-mean R² ≈ −47 (±80), **pooled RMSE ≈ 7.2**. Magnitude varies wildly by fold —
   that variance is the point. RF and CatBoost under D stay positive (~0.3–0.5).
2. **The collapse is LOCALIZED to held-out membrane 300 (`table_D_per_membrane.csv`).** The fold
   average hides this; the per-membrane breakdown is the headline LOMO artifact. Expected pattern
   for `ridge/FS1`:

   | held-out membrane | RMSE | R² | bias | reading |
   |---|---|---|---|---|
   | 225 | ~1.0 | ~−1.3 | +0.9 | mild |
   | 250 | ~1.0 | ~−0.9 | +1.0 | mild |
   | **300** | **~9.6** | **~−186** | **+3.0** | **catastrophe** |
   | 400 | ~0.33 | **~+0.78** | ~0 | fine (interpolation in disguise) |

   Mechanism: membrane 300 is the majority class (44 rows) and the only one spanning the full target
   range (up to 4.44); holding it out forces true extrapolation, and the full model with aliased
   membrane descriptors sticks out wildly (large positive bias = systematic under-prediction). Holding
   out 400 stays inside the training hull → even `ridge/FS1` does fine (R² ~0.78). This is the
   "most OOD tests are actually interpolation" point made concrete. For `ridge/FS3`, `catboost/FS3`,
   `random_forest/FS1`, **every** membrane stays bounded (RMSE < ~0.8, no explosion).
   **Bug signal:** if 300 is *not* the worst held-out membrane for `ridge/FS1`, or 400 is *not* the
   easiest, or any parsimonious/tree model explodes on some membrane → investigate the split/features.
3. **Parsimony wins under extrapolation.** Under D, `ridge/FS3` (≈0.45) and `catboost/FS3` (≈0.55)
   beat `ridge/FS1` (collapse) by a large margin.
4. **Replicate-leakage gap is SMALL (important, counter-intuitive).** B (random) vs C (grouped)
   differ by only ~0.02–0.03 R² (e.g. catboost FS1 0.83 vs 0.80). **Do not expect a large random→
   grouped gap.** The dramatic gap is C→D (interpolation→extrapolation). If you see a *large*
   random→grouped gap, that is suspicious, not a triumph — check whether random test sets are
   unluckily hard or whether grouping is malfunctioning.
5. **Demo 2 — pH↔zeta migration (`importance_stability.csv`).** With both present (FS2): pH rank ≈ 1
   (importance ~0.32–0.43), zeta rank ≈ 2–3 (~0.18–0.22). Remove zeta (FS6): pH jumps to ~0.55–0.78.
   Remove pH (FS2_NO_PH): zeta jumps to ~0.43–0.87. Cross-method agreement Kendall τ ≈ 0.22–0.67
   (methods disagree). **Migration present across all 4 methods (coef, permutation, SHAP, GPR-ARD) =
   GREEN.** If pH importance is unchanged when zeta is removed, the aliasing test isn't firing — bug.

---

## 6. Leakage audit (`leakage_audit.csv`) — RED if violated

| protocol | replicate_overlap | membrane_overlap | leakage_warning |
|---|---|---|---|
| A | > 0 (≈ 15) | 4 | True |
| B | > 0 (≈ 12–15) | ≈ 4 | True |
| C | **0** | > 0 (≈ 3.8) | **False** |
| D | **0** | **0** | **False** |

`C` or `D` showing `replicate_overlap > 0` → grouping broken (replicates split across folds): **RED**.
`D` showing `membrane_overlap > 0` → LOMO not isolating the held-out membrane: **RED**.

---

## 7. Effective SFR (`effective_sfr.csv`)

Nominal SFR overstates information; effective (per independent condition) and per-domain SFR are much
smaller. Reference: FS1 nominal 5.6 / cond 3.3 / domain 0.29; FS3 nominal 19.8 / cond 11.5 / domain
1.0. Domain-SFR ≤ 1 for every feature set (only 4 membranes) — that is the message.

---

## 8. Confirm-vs-bug quick heuristics (run mentally on every new result)

| Symptom | Likely cause | Action |
|---|---|---|
| `condition_id` count = 78 | PWP included in key | exclude PWP (§1) — RED |
| noise floor far from 0.23 | replicate grouping or formula | recheck `noise_floor.py` — RED |
| C/D `replicate_overlap > 0` | replicates split across folds | fix grouping — RED |
| D `membrane_overlap > 0` | LOMO not isolating membrane | fix split — RED |
| `ridge/FS1` under D **not** strongly negative | membranes not truly held out, or scaler/selector leaking, or λ too large | inspect pipeline — AMBER→RED |
| held-out 300 **not** the worst for `ridge/FS1`, or 400 not the easiest | split/feature wiring changed the extrapolation geometry | inspect `table_D_per_membrane.csv` — AMBER |
| a parsimonious/tree model explodes on some held-out membrane | feature leakage of membrane identity, or bad fold | inspect features/split — RED |
| B (random) ≈ D (LOMO) | membrane identity leaking via a near-constant feature, or test always same membrane | inspect features/splits — RED |
| C (grouped) ≫ B (random) | grouped should not beat random much | check random test-set difficulty — AMBER |
| any of our models `below_floor = True` | leakage in our own pipeline | move scaling/selection inside fold — RED |
| Demo 2: pH importance unchanged when zeta dropped | aliasing test not exercised / wrong FS | check FS2/FS6/FS2_NO_PH wiring — AMBER |

---

## 9. Reporting fixes already known (bake into figures/tables)

- **LOMO metric.** Do not headline fold-averaged R² for D (−47±80 is fragile; averaging R² over 4
  tiny folds is dubious). Headline a **pooled** metric over all test predictions (pooled RMSE ≈ 7.2)
  and add a **per-held-out-membrane table** (which membrane breaks the model — expected: the extremes
  225/400). Agent must expose `held_out_membrane` per fold to enable this.
- **Protocol A.** Single split, redundant with B. Keep only as "reproduces the source-paper setting";
  never use A's dummy RMSE (0.89) as the trivial baseline — use B's **0.728**.

---

## 10. Update protocol

When a result legitimately changes (new data, fixed bug, new model), update the relevant row here and
note it in the changelog below. This file is the single source of truth for "what a correct run looks
like"; if reality and this file disagree, one of them is wrong — resolve it, don't ignore it.

### Changelog
- 2026-06-30 — initial version from verified clean run. Noise floor 0.229; extrapolation collapse,
  parsimony-wins, small replicate-leakage gap, and pH↔zeta migration all confirmed on real data.
- 2026-06-30 — added §5.2 per-held-out-membrane expectations (`table_D_per_membrane.csv`): the
  `ridge/FS1` collapse is localized to membrane 300 (RMSE ~9.6, R² ~−186, bias +3); membrane 400 is
  interpolation in disguise (R² ~0.78). Added matching §8 heuristics.
