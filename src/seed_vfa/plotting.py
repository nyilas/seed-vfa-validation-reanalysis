"""Figure generation for the Seed/VFA reanalysis paper (AGENTS.md §1.8, §10).

Style entry point
-----------------
Call :func:`apply_style` **once** at process startup before creating any figure::

    from seed_vfa.plotting import apply_style
    apply_style()

:func:`apply_style` sets all rcParams via ``tueplots.bundles.tmlr2023``
(constrained layout, LaTeX pdflatex + pgf, venue font sizes).  It also
enables LaTeX via the pdflatex/pgf path and falls back to mathtext (Computer
Modern) with an explicit :mod:`logging` warning if pdflatex is absent.

No figure function or calling script may call ``plt.rcParams.update`` or
``plt.style.use`` independently — all style decisions live here.

Width constants
---------------
TMLR text width = 6.5 in (469.755 pt).
Source: ``tueplots.figsizes.tmlr2023``, ``base_width_in = 6.5``.
See: https://www.jmlr.org/tmlr/author-guide.html

:data:`WIDTH_FULL` = 6.5 in — single full-width figure.
:data:`WIDTH_HALF` = 3.25 in — half-column paired layout.

Palette
-------
Okabe–Ito colorblind-safe palette with a fixed semantic mapping so each
entity has the same color across all figures.  See :data:`MODEL_COLORS`,
:data:`PROTOCOL_COLORS`, :data:`MEMBRANE_COLORS`, :data:`REF_NOISE_FLOOR`,
:data:`REF_TRIVIAL`.

Reference-line invariant
------------------------
Every performance figure must include two horizontal reference lines
(AGENTS.md §1, invariant 8):

* Trivial-model RMSE — style :data:`REF_TRIVIAL`
* Replicate noise floor — style :data:`REF_NOISE_FLOOR`

Titles
------
No in-figure titles (``ax.set_title`` / ``fig.suptitle``).  Descriptions
belong in the LaTeX caption, not on the axes.

Figure inventory
----------------
=======  =============================  ======================================
Fig.     Function                       Contents
=======  =============================  ======================================
1        :func:`fig_noise_floor`        Floor CI + source-paper comparison
2        :func:`fig_dataset_audit`      Target distribution + domain structure
3        :func:`fig_protocol_sweep`     RMSE by protocol × model (FS2)
4        :func:`fig_extrapolation`      Ridge vs RF contrast: protocol C vs D
5        :func:`fig_importance`         pH / ζ importance migration Demo 2
6        :func:`fig_effective_sfr`      Nominal vs effective SFR per FS
extra    :func:`fig_lomo_per_membrane`  Per-membrane RMSE breakdown (LOMO)
=======  =============================  ======================================
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths (derived from this module's location)
# ---------------------------------------------------------------------------

_PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
_FIG_DIR: Path = _PROJECT_ROOT / "results" / "figures"

# ---------------------------------------------------------------------------
# Venue width constants
# TMLR text width confirmed in tueplots.figsizes.tmlr2023 (base_width_in=6.5).
# ---------------------------------------------------------------------------

WIDTH_FULL: float = 6.5   # inches — TMLR text width (469.755 pt)
WIDTH_HALF: float = 3.25  # inches — half-column / paired layout

# Pre-computed tueplots figsize recommendations (nrows × ncols @ golden ratio)
_FS_1X1: tuple[float, float] = (6.500000, 4.017221)
_FS_1X2: tuple[float, float] = (6.500000, 2.008610)   # 2 half-width panels
_FS_2X2: tuple[float, float] = (6.500000, 4.017221)   # 2×2 grid


# ---------------------------------------------------------------------------
# Style entry point
# ---------------------------------------------------------------------------


def apply_style(width: str = "full") -> None:
    """Apply TMLR venue style globally via tueplots.

    Must be called once before creating any figure.  Sets constrained layout,
    venue font sizes, and enables LaTeX via pdflatex/pgf.  Falls back to
    mathtext (Computer Modern) with a loud warning if pdflatex is absent.

    Parameters
    ----------
    width:
        ``"full"`` (default) — default figsize = ``WIDTH_FULL × golden-ratio``.
        ``"half"`` — default figsize = ``WIDTH_HALF × golden-ratio``.
        Individual figure functions override figsize for their own layout.
    """
    from tueplots import bundles, figsizes

    params: dict[str, object] = dict(bundles.tmlr2023())

    if width == "half":
        params["figure.figsize"] = figsizes.tmlr2023(nrows=1, ncols=2)["figure.figsize"]
    else:
        params["figure.figsize"] = figsizes.tmlr2023(nrows=1, ncols=1)["figure.figsize"]

    plt.rcParams.update(params)

    # pgf path for embedded fonts; tmlr2023 already sets text.usetex=True
    if shutil.which("pdflatex"):
        plt.rcParams.update({
            "pgf.texsystem": "pdflatex",
            "pgf.rcfonts": False,
        })
    else:
        logging.warning(
            "pdflatex not found in PATH. Falling back to mathtext (Computer Modern). "
            "Figure fonts will NOT match the manuscript — install texlive to fix."
        )
        plt.rcParams.update({
            "text.usetex": False,
            "font.family": "serif",
            "mathtext.fontset": "cm",
        })


# ---------------------------------------------------------------------------
# Per-axes finalizer
# ---------------------------------------------------------------------------


def finalize(ax: plt.Axes) -> None:
    """Remove top/right spines; add light y-grid; set_axisbelow.

    Call once per Axes after all data has been added.
    """
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#e6e6e6", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure save helper
# ---------------------------------------------------------------------------


def save(fig: plt.Figure, name: str, *, dpi: int = 300) -> None:
    """Save *fig* to ``results/figures/<name>.pdf`` (primary) and ``.png``.

    PDF is the primary deliverable (vector, embeds fonts when
    :func:`apply_style` has been called).  PNG is a 300 dpi preview.
    Closes *fig* after writing.

    Parameters
    ----------
    fig:
        Figure to save.
    name:
        Base filename without extension (e.g. ``"fig3_protocol_sweep"``).
    dpi:
        Resolution for the PNG preview (default 300).
    """
    _FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(_FIG_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Okabe–Ito palette — fixed semantic mapping across all figures
# ---------------------------------------------------------------------------

# Semantic model roles.
# "ridge_full"    = Ridge with dense/full feature sets (FS1, FS2, FS4, FS6).
# "ridge_process" = Ridge with process-only feature sets (FS3, FS5, FS7).
MODEL_COLORS: dict[str, str] = {
    "ridge_full":    "#D55E00",   # vermillion  — the model that collapses under LOMO
    "ridge_process": "#0072B2",   # blue        — degrades gracefully
    "catboost":      "#009E73",   # green
    "random_forest": "#E69F00",   # orange
    "dummy":         "#999999",   # grey
}

MODEL_LABELS: dict[str, str] = {
    "ridge_full":    "Ridge (full)",
    "ridge_process": "Ridge (process-only)",
    "catboost":      "CatBoost",
    "random_forest": "Random Forest",
    "dummy":         "Dummy",
}

PROTOCOL_COLORS: dict[str, str] = {
    "A": "#56B4E9",   # sky blue
    "B": "#0072B2",   # blue
    "C": "#009E73",   # green
    "D": "#D55E00",   # vermillion (extrapolation — parallels ridge_full failure)
}

PROTOCOL_LABELS: dict[str, str] = {
    "A": "A: random 75/25",
    "B": "B: repeated random",
    "C": "C: grouped (cond.)",
    "D": "D: LOMO (memb.)",
}

# Membrane (MWCO) colours — Okabe-Ito, distinct from model colours
MEMBRANE_COLORS: dict[int, str] = {
    225: "#CC79A7",   # reddish purple
    250: "#56B4E9",   # sky blue
    300: "#D55E00",   # vermillion — the catastrophic held-out membrane
    400: "#009E73",   # green
}

METHOD_COLORS: dict[str, str] = {
    "standardized_coef": "#0072B2",
    "permutation":       "#E69F00",
    "shap":              "#009E73",
    "gpr_ard":           "#CC79A7",
}

METHOD_LABELS: dict[str, str] = {
    "standardized_coef": "Std.\\ |coef|",
    "permutation":       "Permutation",
    "shap":              "SHAP",
    "gpr_ard":           "GPR ARD",
}

# Reference-line kwargs — same on every performance figure
REF_NOISE_FLOOR: dict[str, object] = {"ls": ":", "lw": 1.2, "color": "#222222"}
REF_TRIVIAL: dict[str, object] = {"ls": "--", "lw": 1.1, "color": "#555555"}

# pH / ζ colours (importance migration figures)
_PH_COLOR: str = "#E69F00"    # orange
_ZETA_COLOR: str = "#56B4E9"  # sky blue

_PH_FEATURE: str = "pH"
_ZETA_FEATURE: str = "Zeta potential [mV]"

_FS_MIGRATION_LABELS: dict[str, str] = {
    "FS2":      "FS2\n(pH + \\zeta)",
    "FS6":      "FS6\n(\\zeta removed)",
    "FS2_NO_PH": "FS2\\_NO\\_PH\n(pH removed)",
}

# Feature sets that use the full / dense feature collection
_RIDGE_FULL_FS: frozenset[str] = frozenset({"FS1", "FS2", "FS4", "FS6"})


def _model_role(model: str, feature_set: str) -> str:
    """Map ``(model, feature_set)`` → semantic role key for :data:`MODEL_COLORS`.

    For ``model="ridge"``, returns ``"ridge_full"`` when *feature_set* is in the
    dense set (FS1/FS2/FS4/FS6) and ``"ridge_process"`` otherwise.  All other
    models map to themselves.
    """
    if model == "ridge":
        return "ridge_full" if feature_set in _RIDGE_FULL_FS else "ridge_process"
    return model


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _add_reference_lines(
    ax: plt.Axes,
    trivial_rmse: float,
    noise_floor: float,
    *,
    add_labels: bool = True,
) -> None:
    """Draw trivial-RMSE and noise-floor reference lines (AGENTS.md §1.8)."""
    ax.axhline(
        trivial_rmse,
        label=(f"Trivial RMSE = {trivial_rmse:.3f}" if add_labels else None),
        zorder=0,
        **REF_TRIVIAL,  # type: ignore[arg-type]
    )
    ax.axhline(
        noise_floor,
        label=(f"Noise floor = {noise_floor:.3f}" if add_labels else None),
        zorder=0,
        **REF_NOISE_FLOOR,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------------
# Fig. 1 — Noise floor
# ---------------------------------------------------------------------------


def fig_noise_floor(
    noise_floor_df: pd.DataFrame,
    sidebar_df: pd.DataFrame,
    *,
    trivial_rmse: float,
    figsize: tuple[float, float] = _FS_1X1,
) -> plt.Figure:
    """Fig. 1: replicate noise floor estimate and source-paper context.

    Left panel  — noise floor with 95 % bootstrap CI vs trivial RMSE.
    Right panel — source-paper implied RMSE values vs noise floor.
    """
    floor = float(noise_floor_df["RMSE_floor"].iloc[0])
    ci_lo = float(noise_floor_df["ci_lower_95"].iloc[0])
    ci_hi = float(noise_floor_df["ci_upper_95"].iloc[0])
    ymax = max(trivial_rmse, ci_hi) * 1.3

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)

    # ── Left: floor estimate ──────────────────────────────────────────────
    ax_l.errorbar(
        [0.5], [floor],
        yerr=[[floor - ci_lo], [ci_hi - floor]],
        fmt="o", color=REF_NOISE_FLOOR["color"], markersize=9,
        capsize=7, linewidth=2.0, zorder=3,
        label=f"Floor = {floor:.3f}",
    )
    ax_l.axhline(
        trivial_rmse, label=f"Trivial RMSE = {trivial_rmse:.3f}", zorder=0,
        **REF_TRIVIAL,  # type: ignore[arg-type]
    )
    ax_l.axhline(floor, zorder=0, alpha=0.35, **REF_NOISE_FLOOR)  # type: ignore[arg-type]
    ax_l.annotate(
        f"95\\% CI\n[{ci_lo:.3f},\\,{ci_hi:.3f}]",
        xy=(0.5, ci_hi), xytext=(0.72, ci_hi + 0.03),
        fontsize=7, color=str(REF_NOISE_FLOOR["color"]),
        arrowprops=dict(arrowstyle="-", color=REF_NOISE_FLOOR["color"]),
    )
    ax_l.set_xlim(0, 1)
    ax_l.set_xticks([0.5])
    ax_l.set_xticklabels(["Noise floor\nestimate"])
    ax_l.set_ylabel("RMSE (g/L)")
    ax_l.set_ylim(0, ymax)
    ax_l.legend(frameon=False, loc="upper right")
    finalize(ax_l)

    # ── Right: published comparison ───────────────────────────────────────
    pub_labels: list[str] = []
    pub_values: list[float] = []
    pub_below: list[bool] = []
    for _, row in sidebar_df.iterrows():
        src = str(row["source"])
        if "baseline" in src:
            lbl = "Source baseline\n($R^2$=0.885)"
        else:
            lbl = "Source best\n(augmented, $R^2$=0.937)"
        pub_labels.append(lbl)
        pub_values.append(float(row["RMSE_implied"]))
        pub_below.append(bool(row["below_floor"]))

    xs = np.arange(len(pub_labels))
    bar_colors = ["#D55E00" if bf else "#0072B2" for bf in pub_below]
    bars = ax_r.bar(
        xs, pub_values, color=bar_colors, width=0.45, edgecolor="white", zorder=2
    )
    for bar, bf in zip(bars, pub_below):
        if bf:
            ax_r.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                "below floor", ha="center", va="bottom",
                color="#D55E00", fontstyle="italic",
            )
    ax_r.axhline(
        floor, label=f"Noise floor = {floor:.3f}", zorder=0,
        **REF_NOISE_FLOOR,  # type: ignore[arg-type]
    )
    ax_r.axhline(
        trivial_rmse, label=f"Trivial RMSE = {trivial_rmse:.3f}", zorder=0,
        **REF_TRIVIAL,  # type: ignore[arg-type]
    )
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels(pub_labels)
    ax_r.set_ylabel("Implied RMSE (g/L)")
    ax_r.set_ylim(0, ymax)
    ax_r.legend(frameon=False, loc="upper right")
    finalize(ax_r)

    return fig


# ---------------------------------------------------------------------------
# Fig. 2 — Dataset audit
# ---------------------------------------------------------------------------

_FEED_COLORS: dict[str, str] = {
    "simple":  "#0072B2",   # blue
    "complex": "#E69F00",   # orange
}


def fig_dataset_audit(
    unique_rows: pd.DataFrame,
    *,
    figsize: tuple[float, float] = _FS_1X1,
) -> plt.Figure:
    """Fig. 2: dataset structure from de-duplicated observations.

    Left  — histogram of target VFA retention by feed type.
    Right — strip plot of target by membrane domain (MWCO), jittered.

    Parameters
    ----------
    unique_rows:
        One row per unique ``row_id``.
        Requires columns: ``y_true``, ``feed_type``, ``membrane_id``.
    """
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=figsize)
    rng = np.random.RandomState(0)

    # ── Left: target distribution by feed type ────────────────────────────
    y_all = unique_rows["y_true"].values
    bins = np.linspace(float(y_all.min()) - 0.05, float(y_all.max()) + 0.05, 17)

    for ft in ["simple", "complex"]:
        grp = unique_rows[unique_rows["feed_type"] == ft]
        if grp.empty:
            continue
        ax_l.hist(
            grp["y_true"], bins=bins, alpha=0.65,
            color=_FEED_COLORS[ft], edgecolor="white", linewidth=0.5,
            label=f"{ft} (n={len(grp)})",
        )

    ax_l.set_xlabel("Cret total VFAs (g/L)")
    ax_l.set_ylabel("Count")
    ax_l.legend(frameon=False)
    finalize(ax_l)

    # ── Right: strip plot by membrane MWCO ───────────────────────────────
    membranes: list[int] = sorted(int(m) for m in unique_rows["membrane_id"].unique())

    for xi, mwco in enumerate(membranes):
        grp = unique_rows[unique_rows["membrane_id"] == mwco]
        jitter = rng.uniform(-0.18, 0.18, size=len(grp))
        ax_r.scatter(
            xi + jitter, grp["y_true"].values,
            color=MEMBRANE_COLORS.get(mwco, "#888888"),
            alpha=0.75, s=24, edgecolors="none", zorder=3,
            label=f"MWCO {mwco}\\,Da  (n={len(grp)})",
        )

    ax_r.set_xticks(range(len(membranes)))
    ax_r.set_xticklabels([f"{m}\\,Da" for m in membranes])
    ax_r.set_xlabel("Membrane (MWCO)")
    ax_r.set_ylabel("Cret total VFAs (g/L)")
    ax_r.legend(frameon=False)
    finalize(ax_r)

    return fig


# ---------------------------------------------------------------------------
# Fig. 3 — Protocol sweep RMSE
# ---------------------------------------------------------------------------


def fig_protocol_sweep(
    metrics_df: pd.DataFrame,
    *,
    feature_set: str = "FS2",
    trivial_rmse: float,
    noise_floor: float,
    figsize: tuple[float, float] = _FS_1X1,
) -> plt.Figure:
    """Fig. 3: RMSE by protocol for each model (feature_set fixed).

    Connected dot plot: each model is one line across protocols A→B→C→D.
    Error bars show RMSE_std.  Includes both mandatory reference lines.
    """
    sub = metrics_df[metrics_df["feature_set"] == feature_set].copy()
    protocols = ["A", "B", "C", "D"]
    models = ["dummy", "ridge", "catboost", "random_forest"]

    fig, ax = plt.subplots(figsize=figsize)
    _add_reference_lines(ax, trivial_rmse, noise_floor)

    x_base = np.arange(len(protocols), dtype=float)
    n_m = len(models)
    role = _model_role("ridge", feature_set)

    for i, model in enumerate(models):
        if model == "ridge":
            color = MODEL_COLORS[role]
            label = MODEL_LABELS[role]
        else:
            color = MODEL_COLORS[model]
            label = MODEL_LABELS[model]

        m_sub = sub[sub["model"] == model].set_index("protocol")
        ys = np.array([
            float(m_sub.loc[p, "RMSE_mean"]) if p in m_sub.index else np.nan
            for p in protocols
        ])
        errs = np.array([
            float(m_sub.loc[p, "RMSE_std"]) if p in m_sub.index else 0.0
            for p in protocols
        ])
        offset = (i - (n_m - 1) / 2) * 0.08
        xs = x_base + offset
        mask = ~np.isnan(ys)

        ax.plot(
            xs[mask], ys[mask],
            color=color, marker="o", markersize=6.5,
            linewidth=1.6, label=label, zorder=3,
        )
        ax.errorbar(
            xs[mask], ys[mask], yerr=errs[mask],
            fmt="none", color=color, capsize=3.5, linewidth=1.0, zorder=3,
        )

    ax.set_xticks(x_base)
    ax.set_xticklabels([PROTOCOL_LABELS[p] for p in protocols])
    ax.set_ylabel("RMSE (g/L)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, frameon=False, ncol=2)
    finalize(ax)
    return fig


# ---------------------------------------------------------------------------
# Fig. 4 — Extrapolation contrast (C vs D)
# ---------------------------------------------------------------------------


def fig_extrapolation(
    metrics_df: pd.DataFrame,
    *,
    trivial_rmse: float,
    noise_floor: float,
    y_cap: float = 1.1,
    figsize: tuple[float, float] = _FS_1X1,
) -> plt.Figure:
    """Fig. 4: RMSE under honest interpolation (C) vs extrapolation (D).

    Left  — Protocol C: GroupKFold(condition_id).
    Right — Protocol D: LOMO by membrane_id.

    Bars that exceed *y_cap* are drawn at the cap with actual value annotated.
    Both mandatory reference lines appear on each panel.
    Bar colour reflects the semantic model role (ridge_full vs ridge_process).
    """
    feature_sets = ["FS1", "FS2", "FS3", "FS5"]
    models_shown = ["ridge", "catboost", "random_forest"]

    x = np.arange(len(feature_sets))
    bar_w = 0.24
    n_m = len(models_shown)

    fig, (ax_c, ax_d) = plt.subplots(1, 2, figsize=figsize, sharey=False)

    # Collect all semantic roles that actually appear, for the shared legend
    _seen_roles: set[str] = set()

    for panel_idx, (proto, ax) in enumerate([("C", ax_c), ("D", ax_d)]):
        _add_reference_lines(ax, trivial_rmse, noise_floor, add_labels=(panel_idx == 0))
        sub = metrics_df[metrics_df["protocol"] == proto]

        for j, model in enumerate(models_shown):
            m_sub = sub[sub["model"] == model].set_index("feature_set")
            offset = (j - (n_m - 1) / 2) * bar_w

            for fi, fs in enumerate(feature_sets):
                if fs not in m_sub.index:
                    continue
                role = _model_role(model, fs)
                _seen_roles.add(role)
                color = MODEL_COLORS[role]
                y_raw = float(m_sub.loc[fs, "RMSE_mean"])
                err = float(m_sub.loc[fs, "RMSE_std"])
                y_plot = min(y_raw, y_cap)
                xpos = x[fi] + offset

                ax.bar(
                    xpos, y_plot, width=bar_w * 0.90,
                    color=color, alpha=0.85, edgecolor="white", zorder=2,
                )
                if err > 0:
                    err_top = min(y_raw + err, y_cap) - y_plot
                    err_bot = min(y_plot, err)
                    ax.errorbar(
                        xpos, y_plot + err_top / 2,
                        yerr=[[err_bot], [err_top]],
                        fmt="none", color=color,
                        capsize=2, linewidth=0.9, zorder=3,
                    )
                if y_raw > y_cap:
                    ax.text(
                        xpos, y_cap + 0.015, f"{y_raw:.2f}↑",
                        ha="center", va="bottom",
                        color=color, fontweight="bold",
                    )

        ax.set_xticks(x)
        ax.set_xticklabels(feature_sets)
        ax.set_xlabel("Feature set")
        ax.set_ylabel("RMSE (g/L)")
        ax.set_ylim(0, y_cap + 0.20)
        finalize(ax)

    # Shared legend on left panel: model roles + reference lines
    role_order = ["ridge_full", "ridge_process", "catboost", "random_forest"]
    role_handles = [
        mpatches.Patch(color=MODEL_COLORS[r], label=MODEL_LABELS[r])
        for r in role_order if r in _seen_roles
    ]
    ref_handles, ref_labels = ax_c.get_legend_handles_labels()
    ax_c.legend(
        handles=role_handles + ref_handles,
        labels=[MODEL_LABELS[r] for r in role_order if r in _seen_roles] + ref_labels,
        frameon=False, loc="upper right",
    )

    return fig


# ---------------------------------------------------------------------------
# Fig. 5 — Importance migration
# ---------------------------------------------------------------------------


def fig_importance(
    importance_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = (WIDTH_FULL, 5.0),
) -> plt.Figure:
    """Fig. 5: pH / ζ importance migration across FS2, FS6, FS2_NO_PH.

    2×2 subplot grid — one panel per importance method.
    Within each panel: bars for pH (orange) and ζ (sky blue) in each feature
    set.  The surviving feature's bar rises when its alias is removed.
    """
    methods = ["standardized_coef", "permutation", "shap", "gpr_ard"]
    fs_order = ["FS2", "FS6", "FS2_NO_PH"]
    bar_w = 0.32

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for idx, method in enumerate(methods):
        ax = axes.flat[idx]
        m_df = importance_df[importance_df["method"] == method]

        for xi, fs_id in enumerate(fs_order):
            fs_df = m_df[m_df["feature_set"] == fs_id]
            ph_row = fs_df[fs_df["feature"] == _PH_FEATURE]
            ze_row = fs_df[fs_df["feature"] == _ZETA_FEATURE]

            has_ph = not ph_row.empty
            has_ze = not ze_row.empty

            if has_ph and has_ze:
                ax.bar(
                    xi - bar_w / 2,
                    float(ph_row["importance_mean"].iloc[0]),
                    width=bar_w, color=_PH_COLOR, alpha=0.85,
                    edgecolor="white", zorder=2,
                )
                ax.bar(
                    xi + bar_w / 2,
                    float(ze_row["importance_mean"].iloc[0]),
                    width=bar_w, color=_ZETA_COLOR, alpha=0.85,
                    edgecolor="white", zorder=2,
                )
            elif has_ph:
                ax.bar(
                    xi, float(ph_row["importance_mean"].iloc[0]),
                    width=bar_w, color=_PH_COLOR, alpha=0.85,
                    edgecolor="white", zorder=2,
                )
            elif has_ze:
                ax.bar(
                    xi, float(ze_row["importance_mean"].iloc[0]),
                    width=bar_w, color=_ZETA_COLOR, alpha=0.85,
                    edgecolor="white", zorder=2,
                )

        ax.set_xticks(range(len(fs_order)))
        ax.set_xticklabels(
            [_FS_MIGRATION_LABELS[fs] for fs in fs_order],
        )
        ax.set_ylabel("Normalised importance")
        ax.set_ylim(0, 1.05)
        # Method label inside panel (not a title) so the panel is identified
        ax.text(
            0.97, 0.97, METHOD_LABELS[method],
            transform=ax.transAxes, ha="right", va="top",
        )
        finalize(ax)

    # Shared legend in top-left panel
    axes.flat[0].legend(
        handles=[
            mpatches.Patch(color=_PH_COLOR, label="pH"),
            mpatches.Patch(color=_ZETA_COLOR, label="$\\zeta$ (zeta potential)"),
        ],
        frameon=False,
    )

    return fig


# ---------------------------------------------------------------------------
# Fig. 6 — Effective SFR
# ---------------------------------------------------------------------------


def fig_effective_sfr(
    sfr_df: pd.DataFrame,
    *,
    figsize: tuple[float, float] = _FS_1X1,
) -> plt.Figure:
    """Fig. 6: nominal vs effective sample-to-feature ratio.

    Grouped bar chart (3 bars per feature set): nominal SFR (79 rows),
    effective SFR at condition level (46 conditions), effective SFR at domain
    level (4 membranes).  Reference lines at SFR = 1 and SFR = 5.
    """
    levels: list[tuple[str, str, str]] = [
        ("nominal_sfr",             "#aec7e8", "Nominal  (N=79 rows)"),
        ("effective_sfr_condition", "#ffbb78", "Effective: conditions (n=46)"),
        ("effective_sfr_domain",    "#98df8a", "Effective: domains (n=4)"),
    ]
    feature_sets: list[str] = sfr_df["feature_set"].tolist()
    x = np.arange(len(feature_sets))
    bar_w = 0.26
    n_lev = len(levels)

    fig, ax = plt.subplots(figsize=figsize)

    for j, (col, color, label) in enumerate(levels):
        offset = (j - (n_lev - 1) / 2) * bar_w
        ax.bar(
            x + offset, sfr_df[col].values,
            width=bar_w * 0.92, color=color,
            edgecolor="white", label=label, zorder=2,
        )

    ax.axhline(
        1.0, zorder=1,
        label="SFR = 1  (underdetermination boundary)",
        **REF_NOISE_FLOOR,  # type: ignore[arg-type]
    )
    ax.axhline(
        5.0, zorder=1,
        label="SFR = 5  (rule-of-thumb minimum)",
        **REF_TRIVIAL,  # type: ignore[arg-type]
    )

    ax.set_xticks(x)
    ax.set_xticklabels(feature_sets)
    ax.set_xlabel("Feature set")
    ax.set_ylabel("Sample-to-feature ratio (SFR)")
    ax.legend(frameon=False, loc="upper right")
    finalize(ax)
    return fig


# ---------------------------------------------------------------------------
# Fig. extra — per-held-out-membrane LOMO breakdown
# ---------------------------------------------------------------------------


def fig_lomo_per_membrane(
    table_d: pd.DataFrame,
    *,
    trivial_rmse: float,
    noise_floor: float,
    feature_sets: tuple[str, ...] = ("FS1", "FS2", "FS3", "FS5"),
    figsize: tuple[float, float] = (WIDTH_FULL, 5.5),
) -> plt.Figure:
    """Per-held-out-membrane RMSE for protocol D (log-scale y axis).

    2×2 subplots — one per feature set.  x-axis: models (Ridge, CatBoost, RF).
    Bars: one per held-out membrane, coloured by MWCO via :data:`MEMBRANE_COLORS`.
    Log-scale y axis accommodates the ridge/FS1/membrane-300 collapse (RMSE ≈ 9.6).
    Both mandatory reference lines on each panel.

    Parameters
    ----------
    table_d:
        Output of :func:`~seed_vfa.evaluation.compute_lomo_per_membrane`.
    trivial_rmse:
        Trivial-model RMSE reference.
    noise_floor:
        Replicate noise floor RMSE reference.
    feature_sets:
        Feature sets to include (default: FS1/FS2/FS3/FS5).
    figsize:
        Figure dimensions.  Width should equal :data:`WIDTH_FULL`.
    """
    models_shown = ["ridge", "catboost", "random_forest"]
    model_x_labels = ["Ridge", "CatBoost", "RF"]
    membranes: list[int] = sorted(int(m) for m in table_d["held_out_membrane"].unique())

    x = np.arange(len(models_shown))
    n_mem = len(membranes)
    bar_w = 0.18
    offsets = np.linspace(-(n_mem - 1) / 2, (n_mem - 1) / 2, n_mem) * bar_w

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    for idx, fs in enumerate(feature_sets):
        ax = axes.flat[idx]
        sub = table_d[table_d["feature_set"] == fs]

        ax.axhline(
            trivial_rmse,
            label=f"Trivial RMSE = {trivial_rmse:.3f}",
            zorder=0,
            **REF_TRIVIAL,  # type: ignore[arg-type]
        )
        ax.axhline(
            noise_floor,
            label=f"Noise floor = {noise_floor:.3f}",
            zorder=0,
            **REF_NOISE_FLOOR,  # type: ignore[arg-type]
        )

        for j, _model in enumerate(models_shown):
            m_sub = sub[sub["model"] == _model].set_index("held_out_membrane")
            for k, mem in enumerate(membranes):
                if mem not in m_sub.index:
                    continue
                rmse_val = float(m_sub.loc[mem, "RMSE"])
                xpos = float(x[j]) + offsets[k]
                ax.bar(
                    xpos, rmse_val, width=bar_w * 0.88,
                    color=MEMBRANE_COLORS.get(mem, "#888888"),
                    alpha=0.88, edgecolor="white", zorder=2,
                )

        ax.set_yscale("log")
        ax.set_ylim(0.08, 20.0)
        ax.set_xticks(x)
        ax.set_xticklabels(model_x_labels)
        ax.set_xlabel("Model")
        ax.set_ylabel("RMSE (g/L, log scale)")
        # Panel label (not a title)
        ax.text(0.97, 0.97, fs, transform=ax.transAxes, ha="right", va="top")
        finalize(ax)

    # Figure-level shared legend
    mem_handles = [
        mpatches.Patch(color=MEMBRANE_COLORS[m], label=f"Membrane {m}\\,Da")
        for m in membranes
    ]
    ref_handles = [
        plt.Line2D(
            [0], [0], label=f"Trivial RMSE = {trivial_rmse:.3f}",
            **REF_TRIVIAL,  # type: ignore[arg-type]
        ),
        plt.Line2D(
            [0], [0], label=f"Noise floor = {noise_floor:.3f}",
            **REF_NOISE_FLOOR,  # type: ignore[arg-type]
        ),
    ]
    fig.legend(
        handles=mem_handles + ref_handles,
        frameon=False, loc="lower center", ncol=3,
        bbox_to_anchor=(0.5, -0.02),
    )

    return fig
