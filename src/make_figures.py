#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figures.py — figures for the README, from the versioned results only
=========================================================================
Nothing is recomputed here: every figure reads ``results/*.json``, the
out-of-fold / hold-out prediction files and the cleaned bases. Re-running the
script after new model runs refreshes the figures.

    python -m src.make_figures            # figures 1–5 (protocol results)
    python -m src.make_figures --shap     # + figures 6–8 (needs results/shap/)

Figures (``results/figures/``):

1. ``fig1_blocks_map.png``        — the 10 K-means blocks (leave-one-block-out
   folds) over the Level A sample, and the out-of-time RMSE_ln difference
   TabPFN − XGB+lag per block on the map
2. ``fig2_per_block_rmse.png``    — RMSE_ln per block, three models, both legs
3. ``fig3_pred_vs_obs_2026.png``  — predicted vs observed ln(R$/m²) on the
   47,810 transactions of 2026, XGB+lag and TabPFN trained on the full base
4. ``fig4_months_ahead_bias.png`` — mean residual (bias) by month ahead, 2026
5. ``fig5_moran.png``             — Moran's I of the residuals, both legs
6. ``fig6_shap_importance.png``   — mean |SHAP| per feature (ln units)
7. ``fig7_shap_beeswarm.png``     — SHAP values per feature, coloured by the
   feature value
8. ``fig8_shap_waterfall.png``    — one explained property, from the base
   value to the prediction

Colour roles are fixed per model across every figure (SAR blue, XGB+lag
orange, TabPFN aqua, OLS grey); no figure encodes identity by colour alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import pandas as pd                      # noqa: E402

RESULTS = Path("results")
FIG_DIR = RESULTS / "figures"
LEVEL_A = Path("data/itbi_sp_2025_level_a.csv")

# fixed colour roles (validated categorical palette, light surface)
C = {"ols": "#8a8984", "sar_gm": "#2a78d6", "xgb_lag": "#eb6834", "tabpfn_t0": "#1baf7a",
     "text": "#0b0b0b", "text2": "#52514e", "grid": "#e6e5e1", "surface": "#fcfcfb",
     "neg": "#2a78d6", "pos": "#eb6834", "mid": "#c3c2b7"}
NAME = {"ols": "OLS hedonic", "sar_gm": "SAR-GM (k-NN-8, ρ)", "xgb_lag": "XGBoost + spatial lag",
        "tabpfn_t0": "TabPFN-3.5 zero-shot (plain table)"}
MARK = {"ols": "s", "sar_gm": "o", "xgb_lag": "^", "tabpfn_t0": "D"}
FEATURE_LABEL = {"area_construida_m2": "built area (m²)", "area_terreno_m2": "lot area (m²)",
                 "idade": "age (years)", "padrao_nivel": "finish grade (IPTU)",
                 "dist_estacao_m": "distance to station (m)", "mes_idx": "month index",
                 "lat": "latitude", "lon": "longitude", "tipo_imovel": "property type"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": C["grid"], "axes.linewidth": 0.8, "axes.labelcolor": C["text2"],
    "xtick.color": C["text2"], "ytick.color": C["text2"], "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.color": C["grid"], "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
    "figure.facecolor": C["surface"], "axes.facecolor": C["surface"], "savefig.dpi": 200,
    "text.color": C["text"], "axes.titlecolor": C["text"], "axes.spines.top": False,
    "axes.spines.right": False,
})


def _j(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def per_block(res: dict, key: str, metric: str = "rmse_ln") -> pd.Series:
    rows = res["per_seed"]["42"][key]
    return pd.Series({int(r["block"]): r[metric] for r in rows}).sort_index()


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, bbox_inches="tight", facecolor=C["surface"])
    plt.close(fig)
    print(f"  {FIG_DIR / name}")


# --------------------------------------------------------------------------
# 1. blocks map
# --------------------------------------------------------------------------
def fig1_blocks_map() -> None:
    df = pd.read_csv(LEVEL_A, usecols=["lon", "lat", "bloco"])
    cent = df.groupby("bloco")[["lon", "lat"]].median()
    n = df["bloco"].value_counts().sort_index()
    diff = (per_block(_j("oot_tabpfn_t0_full.json"), "per_block")
            - per_block(_j("oot_xgb_lag_full.json"), "per_block"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharex=True, sharey=True)
    for ax in axes:
        ax.set_aspect(1 / np.cos(np.deg2rad(23.55)))
        ax.grid(False)
        ax.set_xlabel("longitude"); ax.set_ylabel("latitude")

    ax = axes[0]
    ax.scatter(df["lon"], df["lat"], s=1.2, c=C["mid"], alpha=0.35, linewidths=0, rasterized=True)
    for b, row in cent.iterrows():
        sub = df[df["bloco"] == b]
        try:
            from scipy.spatial import ConvexHull
            pts = sub[["lon", "lat"]].to_numpy()
            hull = ConvexHull(pts)
            poly = pts[np.append(hull.vertices, hull.vertices[0])]
            ax.plot(poly[:, 0], poly[:, 1], color=C["text2"], lw=0.7, alpha=0.7)
        except Exception:                       # pragma: no cover - scipy missing
            pass
        ax.annotate(str(b), (row["lon"], row["lat"]), ha="center", va="center",
                    fontsize=9, fontweight="bold", color=C["text"],
                    bbox=dict(boxstyle="circle,pad=0.25", fc="white", ec=C["text2"], lw=0.6))
    ax.text(0.98, 0.02, "\n".join(f"block {b}: n = {n[b]:,}" for b in n.index),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color=C["text2"],
            family="DejaVu Sans Mono")
    ax.set_title("(a) The 10 K-means blocks = leave-one-block-out folds\n"
                 "2025 Level A sample, n = 24,000", loc="left")

    ax = axes[1]
    ax.scatter(df["lon"], df["lat"], s=1.2, c=C["mid"], alpha=0.2, linewidths=0, rasterized=True)
    vmax = float(np.abs(diff).max())
    for b, row in cent.iterrows():
        d = diff[b]
        col = C["neg"] if d < 0 else C["pos"]
        ax.scatter(row["lon"], row["lat"], s=120 + 6000 * abs(d), c=col, alpha=0.85,
                   edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(str(b), (row["lon"], row["lat"]), ha="center", va="center", fontsize=8,
                    fontweight="bold", color="white", zorder=4)
        ax.annotate(f"{d:+.3f}", (row["lon"], row["lat"]), xytext=(0, -16),
                    textcoords="offset points", ha="center", fontsize=7.5, color=C["text"])
    ax.set_title("(b) 2026 out-of-time: RMSE_ln difference per block, TabPFN − XGBoost+lag\n"
                 "both fitted on the full 2025 base (82k); negative = TabPFN better",
                 loc="left")
    ax.scatter([], [], c=C["neg"], s=60, label="TabPFN better (negative)")
    ax.scatter([], [], c=C["pos"], s=60, label="XGBoost+lag better (positive)")
    ax.legend(loc="lower left")
    fig.text(0.01, -0.02, f"largest |difference| = {vmax:.3f} ln units; marker area ∝ |difference|",
             fontsize=7.5, color=C["text2"])
    save(fig, "fig1_blocks_map.png")


# --------------------------------------------------------------------------
# 2. per-block RMSE, both legs
# --------------------------------------------------------------------------
def fig2_per_block_rmse() -> None:
    legs = [("Leg 1 — spatial CV on Level A (24k), out-of-fold",
             {"sar_gm": ("cv_sar_gm_2025_level_a.json", "per_fold"),
              "xgb_lag": ("cv_xgb_lag_2025_level_a.json", "per_fold"),
              "tabpfn_t0": ("cv_tabpfn_t0_2025_level_a.json", "per_fold")}),
            ("Leg 2 — fitted on 2025 (82k), evaluated on 2026 (47,810)",
             {"sar_gm": ("oot_sar_gm_full.json", "per_block"),
              "xgb_lag": ("oot_xgb_lag_full.json", "per_block"),
              "tabpfn_t0": ("oot_tabpfn_t0_full.json", "per_block")})]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharey=True)
    for ax, (title, spec) in zip(axes, legs):
        series = {m: per_block(_j(f), k) for m, (f, k) in spec.items()}
        blocks = series["tabpfn_t0"].index
        y = np.arange(len(blocks))
        for m, s in series.items():
            ax.scatter(s.values, y, marker=MARK[m], s=42, c=C[m], label=NAME[m], zorder=3,
                       edgecolors="white", linewidths=0.8)
        for i in y:
            vals = [s.iloc[i] for s in series.values()]
            ax.plot([min(vals), max(vals)], [i, i], color=C["grid"], lw=1.2, zorder=1)
        ax.set_yticks(y); ax.set_yticklabels([f"block {b}" for b in blocks])
        ax.set_xlabel("RMSE of ln(R$/m²), lower is better")
        wins = int((series["tabpfn_t0"] < series["xgb_lag"]).sum())
        ax.set_title(f"{title}\nTabPFN below XGBoost+lag in {wins}/{len(blocks)} blocks", loc="left")
        ax.grid(axis="y", visible=False)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04))
    save(fig, "fig2_per_block_rmse.png")


# --------------------------------------------------------------------------
# 3. predicted vs observed, 2026
# --------------------------------------------------------------------------
def fig3_pred_vs_obs() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharex=True, sharey=True)
    for ax, m in zip(axes, ["xgb_lag", "tabpfn_t0"]):
        p = pd.read_csv(RESULTS / f"oot_pred_{m}_full_seed42.csv")
        res = _j(f"oot_{m}_full.json")["per_seed"]["42"]
        hb = ax.hexbin(p["y_ln"], p["yhat_ln"], gridsize=55, bins="log", cmap="Blues",
                       mincnt=1, linewidths=0.2, edgecolors="none")
        lo, hi = 6.3, 10.4
        ax.plot([lo, hi], [lo, hi], color=C["text2"], lw=1, ls="--")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("observed ln(R$/m²), 2026 declared price")
        ax.set_title(NAME[m] + " — trained on 2025 (82k)", loc="left")
        ax.text(0.03, 0.97, f"RMSE_ln {res['pooled']['rmse_ln']:.3f}   MAPE {res['pooled']['mape_pct']:.1f}%\n"
                            f"R²_ln {res['pooled']['r2_ln']:.3f}   bias {res['bias_ln']:+.3f}   (seed 42)",
                transform=ax.transAxes, va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="none", alpha=0.9))
        ax.grid(False)
    axes[0].set_ylabel("predicted ln(R$/m²)")
    cb = fig.colorbar(hb, ax=axes, shrink=0.8, pad=0.02)
    cb.set_label("transactions per cell (log scale)")
    cb.outline.set_visible(False)
    save(fig, "fig3_pred_vs_obs_2026.png")


# --------------------------------------------------------------------------
# 4. bias per month ahead
# --------------------------------------------------------------------------
def fig4_months_ahead_bias() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for m in ["ols", "sar_gm", "xgb_lag", "tabpfn_t0"]:
        rows = [r for r in _j(f"oot_{m}_full.json")["per_seed"]["42"]["per_month_ahead"]
                if r["n"] >= 500]
        x = [r["months_ahead"] for r in rows]; y = [r["bias_ln"] for r in rows]
        ax.plot(x, y, color=C[m], lw=2, marker=MARK[m], ms=5, label=NAME[m])
        ax.annotate(NAME[m].split(" (")[0], (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=C["text"])
    ax.axhline(0, color=C["text2"], lw=0.8)
    ax.set_xlabel("months after the last training month (December 2025)")
    ax.set_ylabel("mean residual, observed − predicted (ln)")
    ax.set_title("2026: every model under-predicts a rising market; the non-linear ones more\n"
                 "(all fitted on the full 2025 base; months with ≥ 500 transactions)", loc="left")
    ax.set_xlim(-0.3, 8.6)
    ax.set_xticks(range(0, 7))
    ax.set_ylim(bottom=-0.005)
    ax.legend(loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.16))
    save(fig, "fig4_months_ahead_bias.png")


# --------------------------------------------------------------------------
# 5. Moran's I of the residuals
# --------------------------------------------------------------------------
def fig5_moran() -> None:
    legs = [("Leg 1 — CV on Level A (24k),\nout-of-fold residuals",
             {"ols": ("cv_ols_2025_level_a.json", "moran_oof"),
              "sar_gm": ("cv_sar_gm_2025_level_a.json", "moran_oof"),
              "xgb_lag": ("cv_xgb_lag_2025_level_a.json", "moran_oof"),
              "tabpfn_t0": ("cv_tabpfn_t0_2025_level_a.json", "moran_oof")}),
            ("Leg 2 — 2026 residuals,\nmodels fitted on 2025 (82k)",
             {"ols": ("oot_ols_full.json", "moran_test"),
              "sar_gm": ("oot_sar_gm_full.json", "moran_test"),
              "xgb_lag": ("oot_xgb_lag_full.json", "moran_test"),
              "tabpfn_t0": ("oot_tabpfn_t0_full.json", "moran_test")})]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharex=True)
    for ax, (title, spec) in zip(axes, legs):
        models = list(spec)
        for i, m in enumerate(models):
            I = _j(spec[m][0])["per_seed"]["42"][spec[m][1]]["I"]
            ax.barh(i, I, color=C[m], height=0.62, edgecolor="white")
            ax.annotate(f"{I:.3f}", (I, i), xytext=(5, 0), textcoords="offset points",
                        va="center", fontsize=8, color=C["text"])
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels([NAME[m].replace(" (plain table)", "") for m in models])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
    axes[0].set_xlim(0, 0.47)
    fig.supxlabel("Moran's I of the residuals, k-NN-8 weights (0 = no spatial structure left; lower is better)",
                  fontsize=9, color=C["text2"])
    fig.suptitle("Residual spatial autocorrelation — the sharpest refutation criterion",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig5_moran.png")


# --------------------------------------------------------------------------
# 6–8. SHAP
# --------------------------------------------------------------------------
def shap_frames(label: str):
    df = pd.read_csv(RESULTS / "shap" / f"shap_values_{label}.csv", dtype={"sql": str})
    sub = pd.read_csv("data/shap_subsample.csv", dtype={"sql": str}).set_index("sql")
    feats = [c[len("shap_"):] for c in df.columns if c.startswith("shap_")]
    S = df[[f"shap_{f}" for f in feats]].to_numpy()
    return df, sub, feats, S


def fig6_shap_importance(label: str) -> None:
    df, sub, feats, S = shap_frames(label)
    order = np.argsort(np.abs(S).mean(axis=0))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    types = ["apartment", "house", "commercial"]
    tcol = {"apartment": C["sar_gm"], "house": C["xgb_lag"], "commercial": C["tabpfn_t0"]}
    h = 0.26
    for k, t in enumerate(types):
        mask = (df["tipo_imovel"] == t).to_numpy()
        vals = np.abs(S[mask]).mean(axis=0)[order]
        ax.barh(np.arange(len(feats)) + (k - 1) * h, vals, height=h, color=tcol[t],
                label=f"{t} (n={mask.sum()})", edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([FEATURE_LABEL.get(f, f) for f in np.array(feats)[order]])
    ax.set_xlabel("mean |SHAP| in ln(R$/m²)  —  exp(x) is the typical multiplicative effect on the unit value")
    ax.set_title(f"What moves a valuation: permutation SHAP on {len(df)} 2026 properties\n"
                 "(TabPFN-3.5 fitted on the 2025 Level A base, no refit)", loc="left")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, "fig6_shap_importance.png")


def fig7_shap_beeswarm(label: str) -> None:
    df, sub, feats, S = shap_frames(label)
    order = np.argsort(np.abs(S).mean(axis=0))[::-1]
    X = sub.loc[df["sql"]]
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    rng = np.random.default_rng(42)
    cmap = matplotlib.colormaps["Blues"]
    for row, j in enumerate(order):
        f = feats[j]
        v = S[:, j]
        if f == "tipo_imovel":
            x = X["tipo_imovel"].map({"apartment": 0.0, "house": 0.5, "commercial": 1.0}).to_numpy()
        else:
            x = pd.to_numeric(X[f], errors="coerce").to_numpy(dtype=float)
            x = np.nan_to_num(x, nan=0.0)
            lo, hi = np.nanpercentile(x, [5, 95])
            x = np.clip((x - lo) / (hi - lo + 1e-9), 0, 1)
        jitter = rng.uniform(-0.28, 0.28, len(v))
        ax.scatter(v, row + jitter, c=cmap(0.25 + 0.7 * x), s=22, edgecolors="white",
                   linewidths=0.4, zorder=3)
    ax.axvline(0, color=C["text2"], lw=0.8)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([FEATURE_LABEL.get(feats[j], feats[j]) for j in order])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("SHAP value, ln(R$/m²): negative pulls the unit value down, positive up")
    ax.set_title("Direction of the effects, one point per explained property\n"
                 "(property type: light = apartment, mid = house, dark = commercial)", loc="left")
    sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "b", [cmap(0.25), cmap(0.95)]), norm=matplotlib.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.02, ticks=[0, 1])
    cb.ax.set_yticklabels(["low", "high"]); cb.set_label("feature value (5th–95th pct)")
    cb.outline.set_visible(False)
    save(fig, "fig7_shap_beeswarm.png")


def fig8_shap_waterfall(label: str, sql: str | None = None) -> None:
    df, sub, feats, S = shap_frames(label)
    if sql is None:                              # the median-error apartment
        apt = df[df["tipo_imovel"] == "apartment"].copy()
        apt["err"] = (apt["yhat_ln"] - apt["ln_vu"]).abs()
        sql = apt.sort_values("err").iloc[len(apt) // 2]["sql"]
    r = df[df["sql"] == sql].iloc[0]
    x = sub.loc[sql]
    contrib = pd.Series({f: r[f"shap_{f}"] for f in feats})
    contrib = contrib.reindex(contrib.abs().sort_values(ascending=False).index)
    labels = []
    for f in contrib.index:
        v = x[f]
        if f in ("lat", "lon"):
            vs = f"{v:.4f}"
        elif isinstance(v, (int, float, np.floating)):
            vs = f"{v:,.0f}"
        else:
            vs = str(v)
        labels.append(f"{FEATURE_LABEL.get(f, f)} = {vs}")

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    base, run = r["base_value_ln"], r["base_value_ln"]
    ys = np.arange(len(contrib))
    cum = [base]
    for i, (f, v) in enumerate(contrib.items()):
        col = C["pos"] if v > 0 else C["neg"]
        ax.barh(i, v, left=run, color=col, height=0.6, edgecolor="white")
        ax.annotate(f"{v:+.3f} (×{np.exp(v):.2f})", (max(run, run + v), i), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8,
                    color=C["text"])
        run += v
        cum.append(run)
    lo, hi = min(cum), max(cum)
    ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.35 * (hi - lo))
    ax.axvline(base, color=C["text2"], lw=0.8, ls="--",
               label=f"base value (mean over the background) = {base:.3f}")
    ax.axvline(r["yhat_ln"], color=C["tabpfn_t0"], lw=1.2,
               label=f"prediction = {r['yhat_ln']:.3f}")
    ax.set_yticks(ys); ax.set_yticklabels(labels); ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ln(R$/m²)")
    obs = float(r["ln_vu"])
    ax.set_title(f"One valuation explained — {r['tipo_imovel']}, SQL {sql}, block {r['bloco']}\n"
                 f"base R$ {np.exp(base):,.0f}/m² → predicted R$ {np.exp(r['yhat_ln']):,.0f}/m²  "
                 f"(declared: R$ {np.exp(obs):,.0f}/m²)", loc="left")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.12))
    save(fig, "fig8_shap_waterfall.png")


def main() -> None:
    ap = argparse.ArgumentParser(description="README figures from the versioned results")
    ap.add_argument("--shap", action="store_true", help="also the SHAP figures")
    ap.add_argument("--shap-label", default="tabpfn_t0_level_a")
    ap.add_argument("--only", nargs="*", type=int, default=None, help="figure numbers")
    args = ap.parse_args()
    steps = {1: fig1_blocks_map, 2: fig2_per_block_rmse, 3: fig3_pred_vs_obs,
             4: fig4_months_ahead_bias, 5: fig5_moran}
    if args.shap:
        steps.update({6: lambda: fig6_shap_importance(args.shap_label),
                      7: lambda: fig7_shap_beeswarm(args.shap_label),
                      8: lambda: fig8_shap_waterfall(args.shap_label)})
    for k, fn in steps.items():
        if args.only is None or k in args.only:
            fn()


if __name__ == "__main__":
    main()
