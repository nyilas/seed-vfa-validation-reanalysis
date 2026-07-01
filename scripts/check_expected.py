import sys, os, shutil, pandas as pd, numpy as np
R = "results/tables"
red, amber = [], []

# --- STRUCTURAL (RED) ---
nf = pd.read_csv(f"{R}/noise_floor.csv").iloc[0]
if not (0.20 <= nf.RMSE_floor <= 0.26):
    red.append(f"noise floor {nf.RMSE_floor:.3f} outside [0.20,0.26]")

la = pd.read_csv(f"{R}/leakage_audit.csv")
for p in ("C", "D"):
    if la.loc[la.protocol == p, "replicate_overlap"].max() > 0:
        red.append(f"protocol {p}: replicates split across folds")
if la.loc[la.protocol == "D", "membrane_overlap"].max() > 0:
    red.append("protocol D: held-out membrane not isolated")

mm = pd.read_csv(f"{R}/model_metrics.csv")
if mm.below_floor.any():
    red.append("a model is below the noise floor -> leakage in our pipeline")

# --- PREDICTIVE (AMBER) ---
def g(model, fs, proto, col="R2_mean"):
    s = mm[(mm.model==model)&(mm.feature_set==fs)&(mm.protocol==proto)][col]
    return s.iloc[0] if len(s) else np.nan

if g("ridge","FS1","D") > -1:      amber.append("ridge/FS1 under D not collapsing")
if g("ridge","FS3","D") < 0.2:     amber.append("process-only FS3 under D too low")
if abs(g("catboost","FS1","B") - g("catboost","FS1","C")) > 0.15:
    amber.append("random->grouped gap unexpectedly large (should be small)")

_td_path = f"{R}/table_D_per_membrane.csv"
if os.path.isfile(_td_path):
    td = pd.read_csv(_td_path)

    def _td(model, fs, membrane, col):
        s = td[(td.model == model) & (td.feature_set == fs) & (td.held_out_membrane == membrane)][col]
        return s.iloc[0] if len(s) else np.nan

    # Catastrophic collapse: ridge/FS1 on membrane 300
    if _td("ridge","FS1",300,"RMSE") <= 5.0:
        amber.append("ridge/FS1/membrane_300 RMSE should be >>1 (expected ~9.64)")
    if _td("ridge","FS1",300,"R2") >= -10:
        amber.append("ridge/FS1/membrane_300 R2 should be << 0 (expected ~-186)")
    if _td("ridge","FS1",300,"bias") <= 1.5:
        amber.append("ridge/FS1/membrane_300 bias should be large positive (expected ~2.99)")

    # Membrane 400 should be tractable
    if _td("ridge","FS1",400,"R2") < 0.3:
        amber.append("ridge/FS1/membrane_400 R2 < 0.3 (expected ~0.78)")

    # Membrane 300 must be the worst held-out membrane for ridge/FS1
    ridge_fs1 = td[(td.model == "ridge") & (td.feature_set == "FS1")]
    if len(ridge_fs1) >= 2 and ridge_fs1["RMSE"].idxmax() != \
            ridge_fs1[ridge_fs1.held_out_membrane == 300].index[0]:
        amber.append("membrane_300 is not the worst held-out membrane for ridge/FS1")

    # Bounded models should not explode on any membrane
    for _model, _fs in [("ridge", "FS3"), ("catboost", "FS3"), ("random_forest", "FS1")]:
        sub = td[(td.model == _model) & (td.feature_set == _fs)]
        if not sub.empty and sub["RMSE"].max() >= 0.8:
            amber.append(f"{_model}/{_fs} max RMSE >= 0.8 across membranes (expected < 0.8)")

# --- FIGURE STYLE / ACCEPTANCE (RED) ---
# §5: every PNG in results/figures/ must have a PDF sibling (vector output)
import glob, re, subprocess
_fig_dir = "results/figures"
if os.path.isdir(_fig_dir):
    for png in glob.glob(f"{_fig_dir}/*.png"):
        pdf = png[:-4] + ".pdf"
        if not os.path.isfile(pdf):
            red.append(f"vector PDF missing for {os.path.basename(png)}")

# §5: no script sets rcParams locally (grep catches plt.rcParams and plt.style.use)
for _script in glob.glob("scripts/*.py"):
    with open(_script) as _f:
        _src = _f.read()
    if re.search(r'plt\.rcParams\b', _src) or re.search(r'plt\.style\.use\b', _src):
        red.append(f"{os.path.basename(_script)}: local rcParams/style.use forbidden")

# §5: optional pdffonts check (requires poppler-utils; skip if tool absent)
if os.path.isdir(_fig_dir) and shutil.which("pdffonts"):
    for pdf in glob.glob(f"{_fig_dir}/*.pdf"):
        try:
            out = subprocess.check_output(
                ["pdffonts", pdf], stderr=subprocess.DEVNULL, text=True
            )
            # pdffonts columns: name  type  encoding  emb  sub  uni  object ID
            # "no" in the emb column means font is NOT embedded
            for line in out.splitlines()[2:]:   # skip header lines
                parts = line.split()
                if len(parts) >= 4 and parts[3].lower() == "no":
                    amber.append(f"unembedded font in {os.path.basename(pdf)}: {parts[0]}")
        except subprocess.CalledProcessError:
            pass

for m in red:   print("RED  :", m)
for m in amber: print("AMBER:", m)
print("GREEN" if not red and not amber else "")
sys.exit(1 if red else 0)