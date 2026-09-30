#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Location, location, location — a Streamlit view of the TabPFN-3.5 vs spatial
models study on São Paulo ITBI transactions.

Everything shown here is read from the versioned files under ``data/`` and
``results/``: no model is re-run, so what you see is exactly what the
protocol produced. The one exception is the *Appraise* tab, which calls
TabPFN-3.5 through the Prior Labs API and only switches on when a
``TABPFN_TOKEN`` is available (environment variable or Streamlit secret).

Run from the repository root::

    streamlit run app/app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA, RESULTS = ROOT / "data", ROOT / "results"

# --------------------------------------------------------------------------
# Palette (validated categorical order; one entity = one colour everywhere)
# --------------------------------------------------------------------------
MODELS = {  # key -> (label, short label, colour)
    "tabpfn_t0": ("TabPFN-3.5, plain table (zero-shot)", "TabPFN-3.5", "#2a78d6"),
    "xgb_lag":   ("XGBoost + rotated coordinates + k-NN-8 lag", "XGBoost+lag", "#eb6834"),
    "sar_gm":    ("SAR lag, GM_Lag", "SAR", "#1baf7a"),
    "ols":       ("OLS hedonic", "OLS", "#eda100"),
}
ORDER = list(MODELS)                       # bar order = validated adjacency
DIVERGING = [[0.0, "#0d366b"], [0.5, "#f0efec"], [1.0, "#b12e2e"]]   # blue <- 0 -> red
SEQUENTIAL = [[0.0, "#cde2fb"], [1.0, "#0d366b"]]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
TYPE_LABEL = {"apartment": "apartment", "house": "house", "commercial": "commercial"}

DTYPES = {"sql": str, "cep": str, "setor": str, "setor_quadra": str,
          "padrao_iptu": str, "padrao_tipo": str}
T0_NUMERIC = ["area_construida_m2", "area_terreno_m2", "idade", "padrao_nivel",
              "dist_estacao_m", "mes_idx", "lat", "lon"]
T0_CATEGORICAL = ["tipo_imovel"]
PROPERTY_TYPES = ["apartment", "commercial", "house"]   # category order of the training table

st.set_page_config(page_title="Location, location, location — TabPFN-3.5 on São Paulo",
                   page_icon="📍", layout="wide")


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading the 2026 transactions and the cached predictions…")
def load_test() -> pd.DataFrame:
    """47,810 transactions of 2026 + one prediction column per model × training base."""
    df = pd.read_csv(DATA / "itbi_sp_2026_test.csv", dtype=DTYPES, parse_dates=["data_transacao"])
    for train in ("level_a", "full"):
        for m in MODELS:
            p = RESULTS / f"oot_pred_{m}_{train}_seed42.csv"
            if p.exists():
                pred = pd.read_csv(p, usecols=["sql", "yhat_ln"], dtype={"sql": str})
                assert (pred["sql"].values == df["sql"].values).all(), p.name
                df[f"yhat_{m}_{train}"] = pred["yhat_ln"].values
                df[f"res_{m}_{train}"] = df["ln_vu"] - df[f"yhat_{m}_{train}"]
    df["vu"] = np.exp(df["ln_vu"])
    return df


@st.cache_data(show_spinner=False)
def load_train(base: str = "level_a") -> pd.DataFrame:
    """2025 training rows: 'level_a' (24k), 'full' (82k) or 'level_a_fin' (financed rows of Level A)."""
    if base == "full":
        return pd.read_csv(DATA / "itbi_sp_2025_train.csv.gz", dtype=DTYPES, parse_dates=["data_transacao"])
    df = pd.read_csv(DATA / "itbi_sp_2025_level_a.csv", dtype=DTYPES, parse_dates=["data_transacao"])
    if base == "level_a_fin":
        df = df[df["financiado"].astype(bool)].reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    """Headline tables straight from the versioned JSON files."""
    out = {"cv": [], "oot_level_a": [], "oot_full": []}
    cv_rows = [("ols", "cv_ols_2025_level_a.json"), ("sar_gm", "cv_sar_gm_2025_level_a.json"),
               ("xgb_lag", "cv_xgb_lag_2025_level_a.json"), ("tabpfn_t0", "cv_tabpfn_t0_2025_level_a.json"),
               ("tabpfn_think", "cv_tabpfn_t0_think_medium_2025_level_a.json")]
    for key, f in cv_rows:
        p = RESULTS / f
        if not p.exists():
            continue
        d = json.load(open(p))
        s = d["per_seed"]["42"]
        row = {"model": key, "rmse_ln": s["pooled"]["rmse_ln"], "mape_pct": s["pooled"]["mape_pct"],
               "r2_ln": s["pooled"]["r2_ln"], "moran_I": s["moran_oof"]["I"]}
        if "seed_summary" in d:
            row["rmse_ci95"] = d["seed_summary"]["rmse_ln"]["ci95"]
        out["cv"].append(row)
    for train in ("level_a", "full"):
        for key in ORDER:
            p = RESULTS / f"oot_{key}_{train}.json"
            if not p.exists():
                continue
            s = json.load(open(p))["per_seed"]["42"]
            out[f"oot_{train}"].append({
                "model": key, "rmse_ln": s["pooled"]["rmse_ln"], "mape_pct": s["pooled"]["mape_pct"],
                "r2_ln": s["pooled"]["r2_ln"], "bias_ln": s["bias_ln"], "moran_I": s["moran_test"]["I"],
                "per_block": pd.DataFrame(s["per_block"]), "per_month": pd.DataFrame(s["per_month_ahead"])})
    out["paired_cv"] = pd.read_csv(RESULTS / "paired_comparisons_2025_level_a.csv")
    out["paired_oot_level_a"] = pd.read_csv(RESULTS / "oot_paired_comparisons_level_a.csv")
    out["paired_oot_full"] = pd.read_csv(RESULTS / "oot_paired_comparisons_full.csv")
    return out


@st.cache_data(show_spinner=False)
def load_shap() -> tuple[pd.DataFrame, dict]:
    p = RESULTS / "shap" / "shap_values_tabpfn_t0_level_a.csv"
    if not p.exists():
        return pd.DataFrame(), {}
    return (pd.read_csv(p, dtype={"sql": str}, parse_dates=["data_transacao"]),
            json.load(open(RESULTS / "shap" / "shap_summary_tabpfn_t0_level_a.json")))


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def geo_figure(traces: list, center: tuple[float, float], zoom: float, height: int, basemap: bool,
               span: float | None = None) -> go.Figure:
    """Map traces on a Carto basemap, or on plain lat/lon axes when offline (``basemap=False``)."""
    if basemap:
        fig = go.Figure([go.Scattermap(**t) for t in traces])
        fig.update_layout(map=dict(style="carto-positron", center=dict(lat=center[0], lon=center[1]), zoom=zoom),
                          height=height, margin=dict(l=0, r=0, t=0, b=0),
                          legend=dict(orientation="h", y=0.02, x=0.02))
        return fig
    fig = go.Figure([go.Scatter(x=t.pop("lon"), y=t.pop("lat"), **t) for t in traces])
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#f9f9f7", legend=dict(orientation="h", y=1.08, x=0))
    if span:
        fig.update_xaxes(range=[center[1] - span, center[1] + span])
        fig.update_yaxes(range=[center[0] - span * 0.92, center[0] + span * 0.92])
    fig.update_xaxes(title="longitude", gridcolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(title="latitude", gridcolor=GRID, tickfont=dict(color=MUTED),
                     scaleanchor="x", scaleratio=1 / np.cos(np.radians(-23.55)))
    return fig


def brl(x: float) -> str:
    return f"R$ {x:,.0f}"


def base_layout(fig: go.Figure, height: int = 380, **kw) -> go.Figure:
    fig.update_layout(template="plotly_white", height=height, margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=INK),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=-0.15, x=0), **kw)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, tickfont=dict(color=MUTED))
    return fig


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
test = load_test()
metrics = load_metrics()

with st.sidebar:
    st.title("📍 Location, location, location")
    st.caption("TabPFN-3.5 with the plain table vs models that see space explicitly — "
               "São Paulo ITBI transactions, 2025 → 2026.")
    train_base = st.radio("Training base for the 2026 predictions",
                          ["level_a", "full"], index=0,
                          format_func=lambda k: {"level_a": "Level A — 24,000 rows (all four models)",
                                                 "full": "Full 2025 base — 82,187 rows"}[k])
    types = st.multiselect("Property type", ["apartment", "house", "commercial"],
                           default=["apartment", "house", "commercial"])
    fin = st.radio("Financing", ["all", "financed only", "cash only"], horizontal=True)
    basemap = st.checkbox("Street basemap on the maps (needs internet)", value=True)
    st.divider()
    st.markdown("**Repository** · [abenvenho/tabpfn-itbi-sp](https://github.com/abenvenho/tabpfn-itbi-sp)  \n"
                "Prior Labs TabPFN-3.5 Hackathon entry · Apache-2.0")
    st.caption("This is a research demo, not an appraisal report. Prices are the values buyers "
               "declared on the ITBI form; estimates are model outputs with no legal standing.")

mask = test["tipo_imovel"].isin(types)
if fin == "financed only":
    mask &= test["financiado"].astype(bool)
elif fin == "cash only":
    mask &= ~test["financiado"].astype(bool)
view = test[mask]
avail = [m for m in ORDER if f"yhat_{m}_{train_base}" in test.columns]

tab_claim, tab_map, tab_prop, tab_appraise = st.tabs(
    ["1 · The claim and the scoreboard", "2 · Where the models fail", "3 · Inspect a transaction", "4 · Appraise a property"])


# ==========================================================================
# Tab 1 — claim + scoreboard
# ==========================================================================
with tab_claim:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("""
### The claim, stated so that it can fail

Every appraiser knows the line: three things set the value of a property — *location, location,
location*. Spatial econometrics turned that into machinery: weights matrices, spatial lags, a ρ to
estimate. Gradient boosting needs the same help, hand-fed as neighbourhood features.

**The bet here:** a tabular foundation model that gets latitude and longitude as *two ordinary
numeric columns* — no weights matrix, no lag, no rotated axes, nothing engineered — predicts
spatially correlated prices as well as, or better than, the specialists that model space explicitly.

The comparison is rigged *against* the bet. The baselines keep every spatial advantage; TabPFN-3.5
gets nine raw columns and ten seconds per fit. It loses if it has worse pooled error, loses most
spatial blocks (paired Wilcoxon, block bootstrap), or — the sharpest test — leaves **more spatial
autocorrelation in its residuals** (Moran's I) than the models that see space.
""")
    with c2:
        st.markdown("### Two legs, one verdict each")
        st.markdown("""
| Leg | Setting | Verdict |
|---|---|---|
| **Spatial block CV** on 24 k rows of 2025 | 10 K-means blocks, each held out in turn: *predict a part of the city the model never saw* | Point error: tie with XGBoost+lag, beats SAR. Residual Moran's I: **partly refuted** — more spatial structure left than XGBoost with the explicit lag |
| **Out-of-time** 2025 → 2026 | Fit once on 2025, predict 47,810 transactions of 2026 inside the same city | **Survives every criterion**, on 24 k and 82 k rows, including the residual one — lowest Moran's I of all four models |
""")
        st.caption("Metrics are on ln(R$/m²). Moran's I on a k-NN-8 matrix of the residuals; higher = more spatial structure unexplained.")

    st.markdown("#### Leg 1 — leave-one-block-out CV, Level A (24,000 rows of 2025)")
    cv = pd.DataFrame(metrics["cv"])
    label = {**{k: v[0] for k, v in MODELS.items()}, "tabpfn_think": "TabPFN-3.5, plain table (thinking, medium)"}
    cv_show = pd.DataFrame({
        "Model": cv["model"].map(label),
        "RMSE (ln)": cv["rmse_ln"].round(3), "MAPE": (cv["mape_pct"]).round(1).astype(str) + " %",
        "R² (ln)": cv["r2_ln"].round(3), "Moran's I of residuals": cv["moran_I"].round(3)})
    st.dataframe(cv_show, hide_index=True, width="stretch")

    st.markdown(f"#### Leg 2 — out-of-time, fit on 2025 ({'Level A, 24,000 rows' if train_base == 'level_a' else 'full base, 82,187 rows'}), predict the 47,810 transactions of 2026")
    oot = pd.DataFrame([{k: v for k, v in r.items() if k not in ("per_block", "per_month")} for r in metrics[f"oot_{train_base}"]])
    oot_show = pd.DataFrame({
        "Model": oot["model"].map(label), "RMSE (ln)": oot["rmse_ln"].round(3),
        "MAPE": oot["mape_pct"].round(1).astype(str) + " %", "R² (ln)": oot["r2_ln"].round(3),
        "Bias (ln, + = under-predicted)": oot["bias_ln"].round(3), "Moran's I of residuals": oot["moran_I"].round(3)})
    st.dataframe(oot_show, hide_index=True, width="stretch")

    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure()
        for r in metrics[f"oot_{train_base}"]:
            pb = r["per_block"].sort_values("block")
            fig.add_bar(x=pb["block"].astype(str), y=pb["rmse_ln"], name=MODELS[r["model"]][1],
                        marker_color=MODELS[r["model"]][2], marker_line_width=0,
                        hovertemplate="block %{x} · RMSE %{y:.3f}<extra>" + MODELS[r["model"]][1] + "</extra>")
        fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, title="RMSE (ln) per spatial block — 2026",
                          yaxis_title="RMSE (ln)", xaxis_title=None)
        fig.update_xaxes(tickprefix="block ")
        st.plotly_chart(base_layout(fig, height=400), width="stretch")
    with g2:
        pc = metrics[f"paired_oot_{train_base}"]
        pc = pc[(pc["a"] == "tabpfn_t0") & (pc["metric"] == "rmse_ln") & pc["b"].isin(ORDER)]
        fig = go.Figure()
        for _, r in pc.iterrows():
            fig.add_scatter(x=[r["pooled_diff"]], y=[MODELS[r["b"]][1]], mode="markers",
                            marker=dict(color=MODELS[r["b"]][2], size=12),
                            error_x=dict(type="data", symmetric=False, array=[r["ci_hi"] - r["pooled_diff"]],
                                         arrayminus=[r["pooled_diff"] - r["ci_lo"]], color=MODELS[r["b"]][2], thickness=2),
                            name=MODELS[r["b"]][1], showlegend=False,
                            hovertemplate="ΔRMSE %{x:+.3f}<extra>vs " + MODELS[r["b"]][1] + "</extra>")
        fig.add_vline(x=0, line_color=MUTED, line_dash="dot")
        fig.update_layout(title="TabPFN-3.5 minus each baseline — ΔRMSE (ln), 2026",
                          xaxis_title="ΔRMSE (ln) with block-bootstrap 95 % CI · negative = TabPFN better")
        st.plotly_chart(base_layout(fig, height=400), width="stretch")
    st.caption("Paired over the ten 2026 blocks; Wilcoxon p = 0.002 for every pair shown (10 of 10 blocks). "
               "Numbers are read from `results/oot_*.json` and `results/oot_paired_comparisons_*.csv`.")

    st.markdown("""
**Two honest footnotes.** Every model under-predicts 2026 — prices rose — and the two non-linear
learners more so: they hold the last observed level of the month index instead of extrapolating a
trend. Practice would apply an index; here it is reported, not corrected. And in the block CV the
plain-table model matched the fully spatial XGBoost on error while leaving more spatial structure in
its residuals: the specialist earns its keep when the test is a part of the city nobody has seen.
""")


# ==========================================================================
# Tab 2 — error map
# ==========================================================================
with tab_map:
    left, right = st.columns([1, 3])
    with left:
        model = st.selectbox("Model", avail, format_func=lambda k: MODELS[k][0])
        mode = st.radio("Show", ["Grid cells (≈500 m)", "Individual transactions (sample)"])
        stat = st.radio("Colour by", ["mean residual (ln)", "RMSE (ln)"]) if mode.startswith("Grid") else "residual"
        n_pts = st.slider("Points to draw", 2000, 15000, 6000, step=1000) if mode.startswith("Ind") else None
        st.caption("Residual = observed − predicted, in ln. Positive (red) = the model under-predicted; "
                   "negative (blue) = over-predicted. Grid cells with fewer than 5 transactions are hidden.")
        res_col = f"res_{model}_{train_base}"
        sub = view.dropna(subset=[res_col])
        st.metric("Transactions in view", f"{len(sub):,}")
        st.metric("RMSE (ln) in view", f"{np.sqrt(np.mean(sub[res_col] ** 2)):.3f}")
        st.metric("MAPE in view", f"{100 * np.mean(np.abs(np.expm1(-sub[res_col]))):.1f} %")
    with right:
        if mode.startswith("Grid"):
            cell = 0.005
            g = sub.assign(gx=(sub["lon"] // cell) * cell + cell / 2, gy=(sub["lat"] // cell) * cell + cell / 2)
            agg = g.groupby(["gx", "gy"]).agg(n=(res_col, "size"), mean_res=(res_col, "mean"),
                                              rmse=(res_col, lambda r: float(np.sqrt(np.mean(r ** 2))))).reset_index()
            agg = agg[agg["n"] >= 5]
            if stat.startswith("mean"):
                z, cs, zmin, zmax, ttl = agg["mean_res"], DIVERGING, -0.4, 0.4, "mean residual (ln)"
            else:
                z, cs, zmin, zmax, ttl = agg["rmse"], SEQUENTIAL, 0.1, 0.5, "RMSE (ln)"
            traces = [dict(
                lat=agg["gy"], lon=agg["gx"], mode="markers",
                marker=dict(size=9, color=z, colorscale=cs, cmin=zmin, cmax=zmax,
                            colorbar=dict(title=ttl, thickness=12), opacity=0.85),
                customdata=np.c_[agg["n"], agg["mean_res"], agg["rmse"]], showlegend=False,
                hovertemplate="%{customdata[0]} transactions<br>mean residual %{customdata[1]:+.3f}<br>RMSE %{customdata[2]:.3f}<extra></extra>")]
        else:
            s = sub.sample(min(n_pts, len(sub)), random_state=42)
            traces = [dict(
                lat=s["lat"], lon=s["lon"], mode="markers",
                marker=dict(size=6, color=s[res_col], colorscale=DIVERGING, cmin=-0.6, cmax=0.6,
                            colorbar=dict(title="residual (ln)", thickness=12), opacity=0.75),
                customdata=np.c_[s["sql"], s["tipo_imovel"], s["vu"].round(0), s[res_col]], showlegend=False,
                hovertemplate="SQL %{customdata[0]} · %{customdata[1]}<br>R$/m² %{customdata[2]:,.0f}<br>residual %{customdata[3]:+.3f}<extra></extra>")]
        st.plotly_chart(geo_figure(traces, (-23.60, -46.62), 9.6, 620, basemap, span=0.24), width="stretch")

    st.markdown("#### Same map, all four models — RMSE (ln) per block")
    per_block = []
    for r in metrics[f"oot_{train_base}"]:
        pb = r["per_block"][["block", "n", "rmse_ln", "mape_pct"]].copy()
        pb["model"] = MODELS[r["model"]][1]
        per_block.append(pb)
    pb = pd.concat(per_block)
    pv = pb.pivot(index="block", columns="model", values="rmse_ln").round(3)[[MODELS[m][1] for m in ORDER if MODELS[m][1] in pb["model"].unique()]]
    pv.insert(0, "n (2026)", pb.groupby("block")["n"].first())
    st.dataframe(pv, width="stretch")
    st.caption("Filters in the sidebar do not apply to this block table (it is the versioned per-block result).")


# ==========================================================================
# Tab 3 — inspect one transaction
# ==========================================================================
with tab_prop:
    shap_df, shap_meta = load_shap()
    train = load_train()
    a, b = st.columns([1, 2])
    with a:
        st.markdown("Pick a 2026 transaction by its cadastral key (SQL), or take one of the 50 that were "
                    "explained with SHAP.")
        pick_from = st.radio("Source", ["one of the 50 SHAP-explained properties", "type a SQL"], label_visibility="collapsed")
        if pick_from.startswith("one") and not shap_df.empty:
            opts = shap_df.assign(lbl=lambda d: d["sql"] + " · " + d["tipo_imovel"] + " · block " + d["bloco"].astype(str)
                                  + " · " + d["data_transacao"].dt.strftime("%Y-%m"))
            sel = st.selectbox("Property", opts["lbl"].tolist())
            sql = sel.split(" · ")[0]
            rows = test[test["sql"] == sql]
            srow = shap_df[shap_df["sql"] == sql].iloc[0]
            rows = rows[(rows["data_transacao"] == srow["data_transacao"]) & (np.isclose(rows["valor_transacao"], srow["valor_transacao"]))]
        else:
            sql = st.text_input("SQL (11 digits, e.g. 00301603448)", value="00301603448").strip().zfill(11)
            rows = test[test["sql"] == sql]
            srow = None
        if rows.empty:
            st.warning("No 2026 transaction with that SQL in the cleaned test base — showing the first SHAP property instead.")
            srow = shap_df.iloc[0] if not shap_df.empty else None
            rows = test[test["sql"] == (srow["sql"] if srow is not None else test["sql"].iloc[0])]
        if len(rows) > 1:
            i = st.selectbox("Several transactions share this SQL — choose one",
                             range(len(rows)), format_func=lambda i: f"{rows.iloc[i]['data_transacao']:%Y-%m-%d} · {brl(rows.iloc[i]['valor_transacao'])}")
            row = rows.iloc[i]
        else:
            row = rows.iloc[0]
        st.markdown(f"**{row['tipo_imovel']}** · {row['descricao_padrao'].title()} · block {row['bloco']}  \n"
                    f"{row['logradouro'].title()}, {row['numero']} {('' if pd.isna(row['complemento']) else row['complemento'])} — {('' if pd.isna(row['bairro']) else str(row['bairro']).title())}")
        attrs = pd.DataFrame({
            "attribute": ["built area (m²)", "lot area (m²)", "age (years)", "finish grade (0–5)", "distance to station (m)",
                          "transaction date", "financed", "declared price", "unit value (R$/m²)"],
            "value": [f"{row['area_construida_m2']:,.0f}", f"{row['area_terreno_m2']:,.0f}", f"{row['idade']:.0f}",
                      f"{row['padrao_nivel']:.0f}", f"{row['dist_estacao_m']:,.0f}", f"{row['data_transacao']:%Y-%m-%d}",
                      "yes" if row["financiado"] else "no", brl(row["valor_transacao"]), f"{row['vu']:,.0f}"]})
        st.dataframe(attrs, hide_index=True, width="stretch")

    with b:
        st.markdown("#### What each model said")
        st.caption(f"Predictions from the fit on the {'24,000 Level A rows' if train_base == 'level_a' else 'full 82,187-row 2025 base'} "
                   "(sidebar). Type and financing filters apply to the map, not to a single property.")
        recs = []
        for m in avail:
            yh = row[f"yhat_{m}_{train_base}"]
            recs.append({"Model": MODELS[m][0], "Estimated R$/m²": float(np.exp(yh)),
                         "Estimated total": float(np.exp(yh) * row["area_ref"]),
                         "Error vs declared": float(np.exp(yh) / row["vu"] - 1)})
        recs = pd.DataFrame(recs)
        fig = go.Figure()
        fig.add_bar(x=recs["Model"].map({v[0]: v[1] for v in MODELS.values()}), y=recs["Estimated R$/m²"],
                    marker_color=[MODELS[m][2] for m in avail], marker_line_width=0,
                    text=[f"{v:,.0f}" for v in recs["Estimated R$/m²"]], textposition="outside",
                    hovertemplate="%{x}: R$ %{y:,.0f}/m²<extra></extra>", showlegend=False)
        fig.add_hline(y=row["vu"], line_color=INK, line_dash="dash",
                      annotation_text=f"declared: R$ {row['vu']:,.0f}/m²", annotation_position="top left")
        fig.update_layout(title="Estimated unit value vs the declared price", yaxis_title="R$/m²", bargap=0.35)
        st.plotly_chart(base_layout(fig, height=330), width="stretch")
        show = recs.copy()
        show["Estimated R$/m²"] = show["Estimated R$/m²"].map(lambda v: f"{v:,.0f}")
        show["Estimated total"] = show["Estimated total"].map(brl)
        show["Error vs declared"] = show["Error vs declared"].map(lambda v: f"{100 * v:+.1f} %")
        st.dataframe(show, hide_index=True, width="stretch")

    c, d = st.columns(2)
    with c:
        st.markdown("#### Nearest 2025 transactions (data, not model)")
        st.caption("The eight closest training transactions of the same type by straight-line distance — the k-NN-8 "
                   "neighbourhood the SAR and XGBoost baselines use. Shown for context; TabPFN-3.5 never receives it.")
        same = train[train["tipo_imovel"] == row["tipo_imovel"]]
        dist = haversine_m(row["lat"], row["lon"], same["lat"].to_numpy(), same["lon"].to_numpy())
        nn = same.iloc[np.argsort(dist)[:8]].assign(dist_m=np.sort(dist)[:8])
        nn_show = pd.DataFrame({"SQL": nn["sql"], "distance (m)": nn["dist_m"].round(0).astype(int),
                                "built m²": nn["area_construida_m2"], "age": nn["idade"].round(0).astype(int),
                                "grade": nn["padrao_nivel"], "month": nn["data_transacao"].dt.strftime("%Y-%m"),
                                "R$/m²": np.exp(nn["ln_vu"]).round(0).astype(int)})
        st.dataframe(nn_show, hide_index=True, width="stretch")
        med = float(np.exp(nn["ln_vu"]).median())
        st.caption(f"Median unit value of the eight: R\\$ {med:,.0f}/m² — vs declared R\\$ {row['vu']:,.0f}/m². "
                   "Distance 0 m = same fiscal block: coordinates are block centroids, so a building's own 2025 sales come first.")
        ctx = same.iloc[np.argsort(dist)[:60]]
        traces = [dict(lat=ctx["lat"], lon=ctx["lon"], mode="markers",
                       marker=dict(size=9, color=np.exp(ctx["ln_vu"]), colorscale=SEQUENTIAL, opacity=0.85,
                                   colorbar=dict(title="R$/m²", thickness=10)),
                       name="60 nearest 2025 sales, same type", customdata=np.exp(ctx["ln_vu"]),
                       hovertemplate="R$/m² %{customdata:,.0f}<extra></extra>"),
                  dict(lat=[row["lat"]], lon=[row["lon"]], mode="markers", marker=dict(size=16, color="#eb6834"),
                       name="this property", hovertemplate="this property<extra></extra>")]
        span = max(0.004, 1.2 * float(np.max(np.abs(ctx[["lat", "lon"]].to_numpy() - [row["lat"], row["lon"]]))))
        st.plotly_chart(geo_figure(traces, (row["lat"], row["lon"]), 14, 320, basemap, span=span), width="stretch")
    with d:
        st.markdown("#### Why TabPFN-3.5 landed there — SHAP")
        if srow is None:
            st.info("SHAP values exist for 50 sampled 2026 properties (`data/shap_subsample.csv`); pick one of them "
                    "on the left to see its attribution. Explaining a new property needs the API (see the README).")
        else:
            feats = ["lon", "lat", "tipo_imovel", "area_construida_m2", "idade", "area_terreno_m2",
                     "dist_estacao_m", "padrao_nivel", "mes_idx"]
            names = {"lon": "longitude", "lat": "latitude", "tipo_imovel": "property type", "area_construida_m2": "built area",
                     "idade": "age", "area_terreno_m2": "lot area", "dist_estacao_m": "distance to station",
                     "padrao_nivel": "finish grade", "mes_idx": "month index"}
            vals = pd.Series({names[f]: srow[f"shap_{f}"] for f in feats}).sort_values(key=np.abs)
            base, pred = srow["base_value_ln"], srow["yhat_ln"]
            fig = go.Figure(go.Bar(
                x=vals.values, y=vals.index, orientation="h",
                marker_color=["#b12e2e" if v > 0 else "#0d366b" for v in vals.values], marker_line_width=0,
                text=[f"×{np.exp(v):.2f}" for v in vals.values], textposition="outside",
                hovertemplate="%{y}: %{x:+.3f} ln<extra></extra>"))
            fig.add_vline(x=0, line_color=MUTED)
            lim = 1.45 * float(np.abs(vals.values).max())
            fig.update_xaxes(range=[-lim, lim])
            fig.update_traces(cliponaxis=False)
            fig.update_layout(title=f"Contribution to ln(R$/m²) · base {np.exp(base):,.0f} → estimate {np.exp(pred):,.0f} R$/m²",
                              xaxis_title="SHAP value (ln); label = multiplier on R$/m²")
            st.plotly_chart(base_layout(fig, height=380), width="stretch")
            st.caption("Permutation SHAP against a 20-row background, model reloaded from its cached record. "
                       "Red pushes the unit value up, blue pulls it down; exp(SHAP) is the multiplicative effect.")


# ==========================================================================
# Tab 4 — live appraisal (API, optional)
# ==========================================================================
def _token() -> str | None:
    tok = os.environ.get("TABPFN_TOKEN")
    if not tok:
        try:
            tok = st.secrets["TABPFN_TOKEN"]
        except Exception:
            tok = None
    return tok


LIVE_BASES = {
    "level_a":     ("Level A — 24,000 rows of 2025 (the main analysis)", "oot_tabpfn_t0_level_a"),
    "full":        ("Full 2025 base — 82,187 rows", "oot_tabpfn_t0_full"),
    "level_a_fin": ("Financed deals only — 7,211 rows of Level A (robustness run)", "oot_tabpfn_t0_level_a_fin"),
}


@st.cache_resource(show_spinner=False)
def _live_models() -> dict:
    """base -> (fitted estimator, how it was obtained); shared by the sessions of this server."""
    return {}


def _fresh_fit(base: str):
    from tabpfn_client import TabPFNRegressor  # noqa: WPS433 (optional dependency)
    tr = load_train(base)
    X = tr[T0_NUMERIC + T0_CATEGORICAL].copy()
    X["area_terreno_m2"] = X["area_terreno_m2"].fillna(0.0)
    X["tipo_imovel"] = X["tipo_imovel"].astype("category")
    m = TabPFNRegressor.create_default_for_version("v3.5", random_state=42,
                                                    ignore_pretraining_limits=(base == "full"))
    m.fit(X, tr["ln_vu"].to_numpy(float))
    return m, f"fresh fit on {len(X):,} rows (same features, same seed as the study)"


def live_predict(token: str, base: str, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Predict with the cached fitted record of ``base`` when the server still has it for this
    account, otherwise with a fresh fit on the same rows.

    ``load_model`` makes no request: a record that was fitted under another account, or that the
    server has dropped, only fails at the first ``predict``. That failure is caught here and the
    model is fitted again, once per base and server process.
    """
    os.environ["TABPFN_TOKEN"] = token
    from tabpfn_client import TabPFNRegressor  # noqa: WPS433 (optional dependency)
    store = _live_models()
    if base not in store:
        rec = RESULTS / "tabpfn_cache" / LIVE_BASES[base][1] / "blockall_seed42_model.json"
        try:
            store[base] = (TabPFNRegressor.load_model(rec),
                           "cached record (the exact model evaluated in the study)")
        except Exception:
            store[base] = _fresh_fit(base)
    model, how = store[base]
    try:
        return np.asarray(model.predict(X), dtype=float), how
    except Exception:
        if not how.startswith("cached record"):
            raise
    store[base] = _fresh_fit(base)
    model, how = store[base]
    return np.asarray(model.predict(X), dtype=float), how


with tab_appraise:
    tok = _token()
    st.markdown("#### Appraise a property with TabPFN-3.5 — the plain table, nothing else")
    st.caption("The sidebar filters describe the 2026 test view of tabs 1–3; they do not change what this tab does. "
               "What changes the estimate is the training base chosen below and the property you describe.")
    if not tok:
        st.warning("This tab calls the Prior Labs API and is switched off because no `TABPFN_TOKEN` was found. "
                   "Set the environment variable (or a Streamlit secret) and restart: "
                   "`export TABPFN_TOKEN=...; streamlit run app/app.py`. Everything else in the app works offline.")
    else:
        try:
            import tabpfn_client  # noqa: F401
            have_client = True
        except ImportError:
            have_client = False
            st.error("`tabpfn-client` is not installed in this environment (`pip install -r requirements-tabpfn.txt`).")
        if have_client:
            live_base = st.radio("Fit TabPFN-3.5 on", list(LIVE_BASES), index=list(LIVE_BASES).index(train_base),
                                 format_func=lambda k: LIVE_BASES[k][0], horizontal=True)
            with st.form("appraise"):
                f1, f2, f3 = st.columns(3)
                tipo = f1.selectbox("Property type", ["apartment", "house", "commercial"])
                area_c = f1.number_input("Built area (m²)", 10.0, 20000.0, 70.0, step=1.0)
                area_t = f1.number_input("Lot area (m²) — 0 if unknown / condominium share", 0.0, 500000.0, 0.0, step=10.0)
                idade = f2.number_input("Age (years)", 0.0, 120.0, 15.0, step=1.0)
                padrao = f2.selectbox("Finish grade (IPTU standard level)", [0, 1, 2, 3, 4, 5], index=2)
                dist = f2.number_input("Distance to nearest subway/rail station (m)", 0.0, 20000.0, 800.0, step=10.0)
                lat = f3.number_input("Latitude", -23.90, -23.35, -23.5613, format="%.5f")
                lon = f3.number_input("Longitude", -46.85, -46.35, -46.6560, format="%.5f")
                mes = f3.selectbox("Price level as of", ["Dec 2025 (end of the training window)"])
                go_btn = st.form_submit_button("Estimate")
            if go_btn:
                X = pd.DataFrame([{"area_construida_m2": area_c, "area_terreno_m2": area_t, "idade": idade,
                                   "padrao_nivel": float(padrao), "dist_estacao_m": dist, "mes_idx": 12.0,
                                   "lat": lat, "lon": lon, "tipo_imovel": tipo}])
                # same category levels as the training table, not the single level of this one row
                X["tipo_imovel"] = pd.Categorical(X["tipo_imovel"], categories=PROPERTY_TYPES)
                try:
                    with st.spinner("Asking TabPFN-3.5 (10–20 s the first time a base is used)…"):
                        pred, how = live_predict(tok, live_base, X)
                except Exception as e:  # noqa: BLE001 (quota, network, revoked token: say so, no traceback)
                    st.error(f"The Prior Labs API call failed — {type(e).__name__}: {str(e)[:500]}")
                    st.stop()
                yh = float(pred[0])
                vu = float(np.exp(yh))
                k1, k2, k3 = st.columns(3)
                k1.metric("Unit value", f"R$ {vu:,.0f}/m²")
                k2.metric("Total (× built area)", brl(vu * area_c))
                k3.metric("Trained on", f"{len(load_train(live_base)):,} rows", help=f"Source: {how}")
                st.caption(f"Model: TabPFN-3.5 zero-shot · {LIVE_BASES[live_base][0]} · {how}.")
                tr = load_train(live_base)
                same = tr[tr["tipo_imovel"] == tipo]
                dd = haversine_m(lat, lon, same["lat"].to_numpy(), same["lon"].to_numpy())
                nn = same.iloc[np.argsort(dd)[:8]].assign(dist_m=np.sort(dd)[:8])
                st.markdown("Nearest 2025 transactions of the same type (context only — the model did not use them as neighbours):")
                st.dataframe(pd.DataFrame({"distance (m)": nn["dist_m"].round(0).astype(int), "built m²": nn["area_construida_m2"],
                                           "age": nn["idade"].round(0).astype(int), "grade": nn["padrao_nivel"],
                                           "month": nn["data_transacao"].dt.strftime("%Y-%m"),
                                           "R$/m²": np.exp(nn["ln_vu"]).round(0).astype(int)}),
                             hide_index=True, width="stretch")
                st.caption("Estimate at the December-2025 price level, from declared ITBI prices of 2025. The study measured "
                           "a MAPE of ~20 % for this model on 2026 transactions and a systematic under-prediction of ~5 % "
                           "(rising market). No trend adjustment is applied. Not an appraisal report.")
