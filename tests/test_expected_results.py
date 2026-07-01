from pathlib import Path

import pandas as pd
import pytest

def test_noise_floor_band():
    nf = pd.read_csv("results/tables/noise_floor.csv").iloc[0]
    assert 0.20 <= nf.RMSE_floor <= 0.26

def test_no_replicate_leakage_in_grouped():
    la = pd.read_csv("results/tables/leakage_audit.csv")
    assert la[la.protocol.isin(["C","D"])].replicate_overlap.max() == 0

def test_condition_count():
    g = pd.read_csv("data/processed/seed_with_groups.csv")
    assert g.condition_id.nunique() == 46   # 78 => PWP wrongly in key


# ---------------------------------------------------------------------------
# Table D (per-held-out-membrane) tests
# ---------------------------------------------------------------------------

TABLE_D_PATH = "results/tables/table_D_per_membrane.csv"
TABLE_D_REQUIRED_COLS = {"model", "feature_set", "held_out_membrane", "n", "y_min", "y_max",
                         "RMSE", "MAE", "R2", "bias"}


@pytest.fixture(scope="module")
def table_d():
    pytest.importorskip("pandas")
    p = Path(TABLE_D_PATH)
    if not p.is_file():
        pytest.skip(f"{TABLE_D_PATH} not yet generated; run scripts/run_table_d.py first")
    return pd.read_csv(p)


@pytest.fixture(scope="module")
def ridge_fs1(table_d):
    return table_d[(table_d.model == "ridge") & (table_d.feature_set == "FS1")]


def _row(df, membrane):
    r = df[df.held_out_membrane == membrane]
    assert len(r) == 1, f"Expected exactly 1 row for membrane {membrane}"
    return r.iloc[0]


def test_table_d_schema(table_d):
    assert TABLE_D_REQUIRED_COLS.issubset(set(table_d.columns))
    assert set(table_d["held_out_membrane"].unique()).issubset({225, 250, 300, 400})


def test_ridge_fs1_membrane_300_collapse(ridge_fs1):
    row = _row(ridge_fs1, 300)
    assert row["RMSE"] > 5.0, f"ridge/FS1/300 RMSE={row['RMSE']:.3f} should be >> 1 (expected ~9.64)"
    assert row["R2"] < -10, f"ridge/FS1/300 R2={row['R2']:.2f} should be << 0 (expected ~-186)"


def test_ridge_fs1_membrane_400_ok(ridge_fs1):
    row = _row(ridge_fs1, 400)
    assert row["R2"] > 0.3, f"ridge/FS1/400 R2={row['R2']:.3f} should be > 0.3 (expected ~0.78)"


def test_300_is_worst_membrane_ridge_fs1(ridge_fs1):
    worst_mem = ridge_fs1.loc[ridge_fs1["RMSE"].idxmax(), "held_out_membrane"]
    assert worst_mem == 300, f"Worst held-out membrane for ridge/FS1 is {worst_mem}, expected 300"


def test_bounded_models_on_all_membranes(table_d):
    for model, fs in [("ridge", "FS3"), ("catboost", "FS3"), ("random_forest", "FS1")]:
        sub = table_d[(table_d.model == model) & (table_d.feature_set == fs)]
        if sub.empty:
            pytest.skip(f"No rows for {model}/{fs}")
        max_rmse = sub["RMSE"].max()
        assert max_rmse < 0.8, (
            f"{model}/{fs} max RMSE across membranes = {max_rmse:.3f}, expected < 0.8"
        )


def test_bias_300_ridge_fs1_positive(ridge_fs1):
    row = _row(ridge_fs1, 300)
    assert row["bias"] > 1.5, (
        f"ridge/FS1/300 bias={row['bias']:.3f} should be > 1.5 (model under-predicts; expected ~2.99)"
    )