#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
out_of_time.py — train on 2025, predict the 2026 transactions
=============================================================
The second leg of the protocol: every model is fitted once on the 2025 base
(Level A sample or the full 82k base) and evaluated on the 47,810 cleaned
transactions from the 2026 ITBI forms, which post-date the training data by
one to seven months (``mes_idx`` 13–19 vs 1–12 in training). Nothing from
2026 touches fitting or tuning.

What is reported (``results/oot_<label>.json``):

* pooled metrics on 2026 (RMSE/MAE in ln, MAPE, R² in ln);
* metrics **per spatial block** of 2026 (the K-means blocks fitted on 2025
  and applied to 2026 by the cleaning pipeline) — the resampling unit of
  the block bootstrap and the pairs of the Wilcoxon test in
  ``compare_holdout``;
* metrics and **mean residual (bias) per month ahead** — models must
  extrapolate the month index beyond the training range, and they do it
  differently (linear models extrapolate the trend, trees hold the last
  level, TabPFN does whatever it does); this table makes that visible;
* Moran's I of the 2026 residuals on a k-NN-8 weights matrix built on the
  2026 coordinates (same sub-metre jitter rule as the CV protocol).

Models (all through the same ``fit(train_df)/predict(test_df)`` interface):
``ols``, ``sar`` (GM_Lag), ``xgb`` (spatial features; hyperparameters tuned
by leave-one-block-out on the *training* base only, cached in
``results/xgb_params_oot_<train>.json``) and ``tabpfn`` (plain table,
zero-shot or thinking; API responses cached under
``results/tabpfn_cache/oot_<label>/``).

Run (baselines here; TabPFN on a machine with API access):
    python -m src.out_of_time --model ols --train level_a
    python -m src.out_of_time --model sar --train level_a
    python -m src.out_of_time --model xgb --train level_a --seeds 42 43 44
    python -m src.out_of_time --model tabpfn --train level_a --thinking off
    python -m src.out_of_time --compare tabpfn_t0_level_a xgb_lag_level_a
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import (BLOCK_COL, KNN_K, SEED, TARGET, block_bootstrap_diff,
                       jittered_coords, knn_weights, load_base,
                       moran_of_residuals, paired_wilcoxon, regression_metrics)

TRAIN_FILES = {"level_a": "data/itbi_sp_2025_level_a.csv",
               "full": "data/itbi_sp_2025_train.csv.gz"}
TEST_FILE = "data/itbi_sp_2026_test.csv"
TRAIN_LAST_MONTH = 12          # mes_idx of December 2025


# --------------------------------------------------------------------------
# Hold-out evaluation
# --------------------------------------------------------------------------
def months_ahead(test_df: pd.DataFrame) -> np.ndarray:
    return (test_df["mes_idx"].to_numpy(dtype=int) - TRAIN_LAST_MONTH)


def run_holdout(model_factory, train_df: pd.DataFrame, test_df: pd.DataFrame,
                label: str, seeds: tuple[int, ...] = (SEED,),
                out_dir: str | Path = "results") -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y = test_df[TARGET].to_numpy(dtype=float)
    blocks = test_df[BLOCK_COL].to_numpy()
    ahead = months_ahead(test_df)
    w = knn_weights(jittered_coords(test_df), k=KNN_K)

    result = {"label": label, "n_train": int(len(train_df)),
              "n_test": int(len(test_df)), "seeds": list(seeds),
              "knn_k": KNN_K, "per_seed": {}}
    for seed in seeds:
        t0 = time.time()
        model = model_factory(seed)
        model.fit(train_df)
        yhat = np.asarray(model.predict(test_df), dtype=float)
        assert len(yhat) == len(y) and not np.isnan(yhat).any()
        resid = y - yhat

        per_block = []
        for b in sorted(np.unique(blocks)):
            m = regression_metrics(y[blocks == b], yhat[blocks == b])
            m["block"] = int(b)
            per_block.append(m)
        per_month = []
        for h in sorted(np.unique(ahead)):
            idx = ahead == h
            if idx.sum() < 30:            # a handful of back-dated forms
                continue
            m = regression_metrics(y[idx], yhat[idx])
            m["months_ahead"] = int(h)
            m["bias_ln"] = float(resid[idx].mean())
            per_month.append(m)
        result["per_seed"][str(seed)] = {
            "pooled": regression_metrics(y, yhat),
            "bias_ln": float(resid.mean()),
            "per_block": per_block, "per_month_ahead": per_month,
            "moran_test": moran_of_residuals(resid, w),
            "runtime_s": round(time.time() - t0, 1)}
        pred = test_df[["sql", "tipo_imovel", BLOCK_COL, "mes_idx"]].copy()
        pred["y_ln"] = y
        pred["yhat_ln"] = yhat
        pred.to_csv(out_dir / f"oot_pred_{label}_seed{seed}.csv", index=False)

    if len(seeds) > 1:
        summ = {}
        for m in ["rmse_ln", "mae_ln", "mape_pct", "r2_ln"]:
            vals = np.array([result["per_seed"][str(s)]["pooled"][m] for s in seeds])
            half = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
            summ[m] = {"mean": float(vals.mean()),
                       "ci95": [float(vals.mean() - half), float(vals.mean() + half)]}
        result["seed_summary"] = summ
    (out_dir / f"oot_{label}.json").write_text(json.dumps(result, indent=2),
                                               encoding="utf-8")
    return result


def compare_holdout(label_a: str, label_b: str, test_df: pd.DataFrame,
                    seed: int = SEED, metric: str = "rmse_ln",
                    out_dir: str | Path = "results") -> dict:
    """Paired Wilcoxon over the 2026 blocks + block bootstrap (A − B)."""
    out_dir = Path(out_dir)
    y = test_df[TARGET].to_numpy(dtype=float)
    blocks = test_df[BLOCK_COL].to_numpy()

    def read(label):
        res = json.loads((out_dir / f"oot_{label}.json").read_text())
        pred = pd.read_csv(out_dir / f"oot_pred_{label}_seed{seed}.csv")
        pb = [b[metric] for b in res["per_seed"][str(seed)]["per_block"]]
        return pb, pred["yhat_ln"].to_numpy(dtype=float)

    pb_a, yhat_a = read(label_a)
    pb_b, yhat_b = read(label_b)
    out = {"a": label_a, "b": label_b, "metric": metric, "seed": seed,
           "wilcoxon_blocks": paired_wilcoxon(pb_a, pb_b),
           "block_bootstrap": block_bootstrap_diff(y, yhat_a, yhat_b, blocks, metric)}
    suffix = "" if metric == "rmse_ln" else f"_{metric}"
    (out_dir / f"oot_compare_{label_a}_vs_{label_b}{suffix}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# Model factories
# --------------------------------------------------------------------------
def make_factory(args, train_df: pd.DataFrame, out_dir: Path, label: str):
    if args.model == "ols":
        from .model_sar import OLSHedonic
        return lambda seed: OLSHedonic(seed=seed)
    if args.model == "sar":
        from .model_sar import SARLagModel
        return lambda seed: SARLagModel(seed=seed, estimator="gm")
    if args.model == "xgb":
        from .model_xgb import XGBSpatialModel, tune_on
        cache = out_dir / f"xgb_params_oot_{args.train}.json"
        if cache.exists():
            tuned = json.loads(cache.read_text())
        else:
            t0 = time.time()
            tuned = tune_on(train_df, args.n_trials, seed=SEED, use_lag=True)
            tuned["tuning_s"] = round(time.time() - t0, 1)
            cache.write_text(json.dumps(tuned, indent=2))
            print(f"  XGB tuned on {args.train} blocks in {tuned['tuning_s']:.0f}s "
                  f"(inner RMSE_ln={tuned['inner_rmse_ln']:.4f})", flush=True)
        params = tuned["params"]
        return lambda seed: XGBSpatialModel(seed=seed, params=params, use_lag=True)
    if args.model == "tabpfn":
        from .model_tabpfn import TabPFNT0Model
        cache_dir = out_dir / "tabpfn_cache" / f"oot_{label}"
        group_col = BLOCK_COL if args.group_col else None
        return lambda seed: TabPFNT0Model(cache_dir, "all", seed=seed,
                                          thinking=args.thinking, group_col=group_col,
                                          quantiles=args.quantiles, chunk=args.chunk,
                                          ignore_pretraining_limits=args.ignore_limits)
    raise ValueError(args.model)


def default_label(args) -> str:
    if args.model == "ols":
        base = "ols"
    elif args.model == "sar":
        base = "sar_gm"
    elif args.model == "xgb":
        base = "xgb_lag"
    else:
        base = ("tabpfn_t0" + ("" if args.thinking == "off" else f"_think_{args.thinking}")
                + ("_grp" if args.group_col else ""))
    return f"{base}_{args.train}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Out-of-time evaluation: fit on 2025, predict 2026")
    ap.add_argument("--model", choices=["ols", "sar", "xgb", "tabpfn"])
    ap.add_argument("--train", choices=list(TRAIN_FILES), default="level_a")
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    ap.add_argument("--n-trials", type=int, default=30, help="XGB: Optuna trials")
    ap.add_argument("--thinking", choices=["off", "medium", "high"], default="off")
    ap.add_argument("--group-col", action="store_true")
    ap.add_argument("--quantiles", type=float, nargs="*", default=None)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--ignore-limits", action="store_true",
                    help="TabPFN: ignore_pretraining_limits (needed for the full 82k base)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("LABEL_A", "LABEL_B"),
                    help="paired tests between two finished hold-out runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out_dir = Path(args.out)
    test_df = load_base(TEST_FILE)
    if args.compare:
        a, b = args.compare
        for metric in ("rmse_ln", "mape_pct"):
            r = compare_holdout(a, b, test_df, metric=metric, out_dir=out_dir)
            w, bb = r["wilcoxon_blocks"], r["block_bootstrap"]
            print(f"{a} vs {b} [{metric}]: per-block mean diff {w['mean_diff']:+.4f} "
                  f"(Wilcoxon p={w['p_value']:.3f}) | pooled {bb['diff']:+.4f} "
                  f"CI95 [{bb['ci95'][0]:+.4f}, {bb['ci95'][1]:+.4f}]")
        return

    if not args.model:
        ap.error("--model is required unless --compare is given")
    train_df = load_base(TRAIN_FILES[args.train])
    label = args.label or default_label(args)
    print(f"Out-of-time | model={args.model} train={args.train} ({len(train_df):,} rows) "
          f"-> test 2026 ({len(test_df):,} rows) | label '{label}'", flush=True)
    factory = make_factory(args, train_df, out_dir, label)
    res = run_holdout(factory, train_df, test_df, label, seeds=tuple(args.seeds),
                      out_dir=out_dir)
    s0 = str(args.seeds[0])
    p, m = res["per_seed"][s0]["pooled"], res["per_seed"][s0]["moran_test"]
    print(f"2026: RMSE_ln={p['rmse_ln']:.4f} MAE_ln={p['mae_ln']:.4f} MAPE={p['mape_pct']:.1f}% "
          f"R2_ln={p['r2_ln']:.3f} bias_ln={res['per_seed'][s0]['bias_ln']:+.4f} "
          f"| Moran I={m['I']:.3f} (z={m['z_norm']:.1f}) | {res['per_seed'][s0]['runtime_s']}s")
    print("months ahead:", " ".join(f"{r['months_ahead']}:{r['rmse_ln']:.3f}/{r['bias_ln']:+.3f}"
                                    for r in res["per_seed"][s0]["per_month_ahead"]))
    if "seed_summary" in res:
        ss = res["seed_summary"]["rmse_ln"]
        print(f"RMSE_ln over seeds: {ss['mean']:.4f} CI95 [{ss['ci95'][0]:.4f}, {ss['ci95'][1]:.4f}]")


if __name__ == "__main__":
    main()
