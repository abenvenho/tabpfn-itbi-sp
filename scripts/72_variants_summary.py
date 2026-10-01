#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
72_variants_summary.py — every model, both legs, one table each
===============================================================
Collects the finished runs of the main comparison and of the later variants
— geographically weighted XGBoost (``src/model_gxgb.py``), TabPFN-3.5 with
grouped thinking (spatial block as ``group_col``) and the text variant (unit
complement + Referência field) — and pairs each TabPFN-3.5 run with each
spatial baseline over the ten spatial blocks. Runs that are not there yet
are skipped, so the script can be re-run as results arrive.

Reads only the versioned result files (``results/cv_*``, ``oof_*``,
``oot_*``, ``oot_pred_*``); numpy and pandas only, no API, no refit. Block
bootstrap as in ``src/protocol.py`` (2,000 resamples of the blocks, seed 42);
exact Wilcoxon signed-rank test on the per-block RMSEs.

    python scripts/72_variants_summary.py

Writes ``results/variants_summary.md`` and ``.json``.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
SEED = 42
N_BOOT = 2000

MODELS = [  # (key, display name, label stem)
    ("ols", "OLS hedonic", "ols"),
    ("sar", "SAR lag", "sar_gm"),
    ("xgb", "XGBoost + rotated coords + k-NN-8 lag", "xgb_lag"),
    ("gxgb", "Geographically weighted XGBoost", "gxgb"),
    ("tabpfn", "TabPFN-3.5, plain table, zero-shot", "tabpfn_t0"),
    ("think", "TabPFN-3.5, thinking (medium)", "tabpfn_t0_think_medium"),
    ("think_grp", "TabPFN-3.5, thinking (medium), blocks as groups", "tabpfn_t0_think_medium_grp"),
    ("text", "TabPFN-3.5, plain table + unit and building text", "tabpfn_t0_txt_complemento_referencia"),
]
NAME = {k: n for k, n, _ in MODELS}
PAIRS = [  # (a, b): a − b
    ("tabpfn", "gxgb"), ("tabpfn", "xgb"), ("tabpfn", "sar"),
    ("gxgb", "xgb"), ("gxgb", "sar"),
    ("text", "tabpfn"), ("text", "gxgb"), ("text", "xgb"), ("text", "sar"),
    ("think_grp", "think"), ("think_grp", "tabpfn"), ("think_grp", "gxgb"),
    ("think_grp", "xgb"), ("think_grp", "sar"),
]
LEGS = {
    "leg1": ("Leg 1 — leave-one-block-out CV, Level A (24k)",
             lambda stem: f"cv_{stem}_2025_level_a.json",
             lambda stem, s: f"oof_{stem}_2025_level_a_seed{s}.csv", "moran_oof"),
    "leg2_level_a": ("Leg 2 — fit on Level A (24k), predict 2026",
                     lambda stem: f"oot_{stem}_level_a.json",
                     lambda stem, s: f"oot_pred_{stem}_level_a_seed{s}.csv", "moran_test"),
    "leg2_full": ("Leg 2 — fit on the full 2025 base (82k), predict 2026",
                  lambda stem: f"oot_{stem}_full.json",
                  lambda stem, s: f"oot_pred_{stem}_full_seed{s}.csv", "moran_test"),
}


def metrics(y, yhat):
    r = y - yhat
    ape = np.abs(np.exp(yhat) - np.exp(y)) / np.exp(y)
    return {"rmse_ln": float(np.sqrt(np.mean(r ** 2))), "mape_pct": float(100 * ape.mean())}


def boot(y, a, b, blocks, metric):
    rng = np.random.default_rng(SEED)
    uniq = np.unique(blocks)
    idx_b = {k: np.flatnonzero(blocks == k) for k in uniq}
    d = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([idx_b[k] for k in rng.choice(uniq, size=len(uniq), replace=True)])
        d[i] = metrics(y[idx], a[idx])[metric] - metrics(y[idx], b[idx])[metric]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": metrics(y, a)[metric] - metrics(y, b)[metric], "ci95": [float(lo), float(hi)]}


def wilcoxon_exact(d):
    d = d[d != 0]
    if len(d) == 0:
        return 1.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w = ranks[d > 0].sum()
    dist = np.array([sum(r for r, s in zip(ranks, sg) if s)
                     for sg in itertools.product((0, 1), repeat=len(d))])
    return float(min(1.0, 2 * min((dist <= w + 1e-9).mean(), (dist >= w - 1e-9).mean())))


def load_leg(key):
    title, rec_name, pred_name, moran_key = LEGS[key]
    runs = {}
    for k, _, stem in MODELS:
        rec_path = RESULTS / rec_name(stem)
        if not rec_path.exists():
            continue
        rec = json.loads(rec_path.read_text())
        seeds = [int(s) for s in rec["per_seed"]]
        preds = {}
        for s in seeds:
            p = RESULTS / pred_name(stem, s)
            if p.exists():
                preds[s] = pd.read_csv(p, dtype={"sql": str})
        if SEED not in preds:
            continue
        runs[k] = {"rec": rec, "preds": preds, "moran_key": moran_key}
    return title, runs


def row_summary(run):
    rec, preds = run["rec"], run["preds"]
    out = []
    for s, df in preds.items():
        m = metrics(df["y_ln"].to_numpy(float), df["yhat_ln"].to_numpy(float))
        ps = rec["per_seed"][str(s)]
        m["moran_I"] = float(ps[run["moran_key"]]["I"])
        m["bias_ln"] = float(ps.get("bias_ln", np.nan))
        m["seed"] = s
        out.append(m)
    return out


def paired(run_a, run_b):
    """TabPFN-side run (seed 42) against every seed of the other run."""
    a = run_a["preds"][SEED]
    res = []
    for s, b in run_b["preds"].items():
        if not (a["sql"].to_numpy() == b["sql"].to_numpy()).all():
            raise ValueError("row order differs")
        y = a["y_ln"].to_numpy(float)
        blocks = a["bloco"].to_numpy()
        ya, yb = a["yhat_ln"].to_numpy(float), b["yhat_ln"].to_numpy(float)
        pb = np.array([metrics(y[blocks == k], ya[blocks == k])["rmse_ln"]
                       - metrics(y[blocks == k], yb[blocks == k])["rmse_ln"]
                       for k in np.unique(blocks)])
        res.append({"b_seed": s, "rmse_ln": boot(y, ya, yb, blocks, "rmse_ln"),
                    "mape_pct": boot(y, ya, yb, blocks, "mape_pct"),
                    "blocks_won": int((pb < 0).sum()), "n_blocks": int(len(pb)),
                    "wilcoxon_p": wilcoxon_exact(pb)})
    return res


def fmt_range(vals, f):
    lo, hi = min(vals), max(vals)
    return f.format(lo) if f.format(lo) == f.format(hi) else f"{f.format(lo)} to {f.format(hi)}"


def main():
    out = {"legs": {}}
    md = ["# Every model, both legs — including geographically weighted XGBoost, "
          "grouped thinking and the text variant", "",
          "Pooled metrics of seed 42 (range over seeds where a model has more than one). "
          "Paired rows: Δ = first − second in RMSE_ln (negative favours the first), CI95 by block "
          f"bootstrap ({N_BOOT:,} resamples), blocks won by the first, exact Wilcoxon p on the "
          "per-block RMSEs; a TabPFN-3.5 run (seed 42) is paired with every seed of a stochastic "
          "baseline and the range is shown. Generated by `scripts/72_variants_summary.py` from the "
          "versioned predictions.", ""]
    for key in LEGS:
        title, runs = load_leg(key)
        if not runs:
            continue
        leg = {"title": title, "models": {}, "pairs": []}
        md += [f"## {title}", "", "| Model | RMSE_ln | MAPE | Moran I | bias ln | seeds |",
               "|---|---|---|---|---|---|"]
        for k, name, _ in MODELS:
            if k not in runs:
                continue
            rows = row_summary(runs[k])
            leg["models"][k] = rows
            bias = [r["bias_ln"] for r in rows if not np.isnan(r["bias_ln"])]
            md.append(f"| {name} | {fmt_range([r['rmse_ln'] for r in rows], '{:.4f}')} "
                      f"| {fmt_range([r['mape_pct'] for r in rows], '{:.1f} %')} "
                      f"| {fmt_range([r['moran_I'] for r in rows], '{:.3f}')} "
                      f"| {fmt_range(bias, '{:+.3f}') if bias else '—'} | {len(rows)} |")
        md += ["", "| First vs second | ΔRMSE_ln [CI95] | ΔMAPE pp | blocks | p |", "|---|---|---|---|---|"]
        for a, b in PAIRS:
            if a not in runs or b not in runs:
                continue
            res = paired(runs[a], runs[b])
            leg["pairs"].append({"a": a, "b": b, "per_seed": res})
            ci = lambda r: f"{r['rmse_ln']['diff']:+.4f} [{r['rmse_ln']['ci95'][0]:+.4f}; {r['rmse_ln']['ci95'][1]:+.4f}]"
            txt = " / ".join(ci(r) for r in res)
            mp = fmt_range([r["mape_pct"]["diff"] for r in res], "{:+.2f}")
            won = fmt_range([r["blocks_won"] for r in res], "{:d}")
            p = fmt_range([r["wilcoxon_p"] for r in res], "{:.3f}")
            seeds = f" (seeds {', '.join(str(r['b_seed']) for r in res)})" if len(res) > 1 else ""
            md.append(f"| {NAME[a]} vs {NAME[b]}{seeds} | {txt} | {mp} | {won}/10 | {p} |")
        md.append("")
        out["legs"][key] = leg
    (RESULTS / "variants_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RESULTS / "variants_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
