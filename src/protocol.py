#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
protocol.py — spatially honest evaluation protocol
===================================================
Shared evaluation machinery for the model comparison (SAR, G-XGBoost,
TabPFN-3.5). It mirrors — and reinforces — the protocol of the reference
study on Belo Horizonte apartments:

* **Spatial-block cross-validation**: 10 folds = the K-means blocks fitted on
  the 2025 UTM coordinates by the cleaning pipeline (column ``bloco``). Each
  fold holds one block out for testing, so models are always judged on
  spatially disjoint areas.
* **Out-of-sample metrics**: RMSE and MAE on the log response, MAPE on the
  price level (R$), and R² on the log response.
* **Moran's I of out-of-fold residuals** with a row-standardised k-NN-8
  weights matrix — the paper's spatial-autocorrelation check.
* **Paired Wilcoxon** (per-fold metric pairs) **plus a block bootstrap** of
  the pooled metric difference (the paper's acknowledged limitation that
  fold pairs are not independent).
* **Multiple seeds** with confidence intervals across seeds (the reference
  study used a single seed).

Coincident coordinates: every transaction inherits the centroid of its
fiscal block, so exact coordinate ties are the rule, not the exception. For
k-NN construction (weights matrix and spatial-lag feature) we apply a
deterministic sub-metre jitter (uniform in ±0.5 m, seeded) that breaks ties
without materially moving any point; this choice is recorded here and in the
results metadata.

Model interface
---------------
A *model factory* is ``factory(seed) -> model`` where the model implements::

    model.fit(train_df)              # train_df has all data columns
    model.predict(test_df) -> np.ndarray   # predictions of ln_vu

Everything a model may not use (the leakage columns listed in the data
dictionary) is the model's responsibility; the protocol only feeds data
frames and collects ln-scale predictions.

Smoke test
----------
``python -m src.protocol --data data/itbi_sp_2025_level_a.csv --smoke``
runs a trivial per-type median predictor through the full machinery and
writes ``results/smoke_protocol.json`` — useful to check the environment
before spending API credits.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from esda.moran import Moran
from libpysal.weights import KNN
from scipy import stats
from scipy.spatial import cKDTree

SEED = 42
JITTER_M = 0.5          # sub-metre tie-breaking jitter for k-NN (metres)
KNN_K = 8               # neighbours in W and in the spatial-lag feature
BLOCK_COL = "bloco"
TARGET = "ln_vu"

DTYPES = {"sql": str, "cep": str, "setor": str, "setor_quadra": str,
          "padrao_iptu": str, "padrao_tipo": str}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def load_base(path: str | Path, financed_only: bool = False,
              types: list[str] | None = None) -> pd.DataFrame:
    """Load a cleaned base; optionally restrict to financed deals or types."""
    df = pd.read_csv(path, dtype=DTYPES, parse_dates=["data_transacao"])
    if financed_only:
        df = df[df["financiado"].astype(bool)]
    if types:
        df = df[df["tipo_imovel"].isin(types)]
    return df.reset_index(drop=True)


def spatial_folds(df: pd.DataFrame, block_col: str = BLOCK_COL):
    """Leave-one-block-out folds over the K-means spatial blocks."""
    folds = []
    for b in sorted(df[block_col].unique()):
        test = np.flatnonzero((df[block_col] == b).to_numpy())
        train = np.flatnonzero((df[block_col] != b).to_numpy())
        folds.append((int(b), train, test))
    return folds


def jittered_coords(df: pd.DataFrame, seed: int = SEED,
                    scale: float = JITTER_M) -> np.ndarray:
    """UTM coordinates with a deterministic sub-metre tie-breaking jitter.

    Many records share the exact fiscal-block centroid; the jitter breaks
    k-NN ties deterministically without materially moving any point.
    """
    rng = np.random.default_rng(seed)
    xy = df[["x_utm", "y_utm"]].to_numpy(dtype=float)
    return xy + rng.uniform(-scale, scale, size=xy.shape)


def knn_weights(xy: np.ndarray, k: int = KNN_K) -> KNN:
    """Row-standardised k-NN spatial weights matrix.

    A k-NN graph is directed and may split into disconnected components
    (libpysal warns about it); that is expected here and unproblematic for
    Moran's I, so the warning is silenced.
    """
    w = KNN.from_array(xy, k=k, silence_warnings=True)
    w.transform = "R"
    return w


def knn_spatial_lag(train_xy: np.ndarray, train_y: np.ndarray,
                    query_xy: np.ndarray, k: int = KNN_K,
                    exclude_self: bool = False) -> np.ndarray:
    """Mean of the k nearest TRAINING responses for each query point.

    Used (a) as the T1 spatial-lag feature and (b) in the SAR out-of-sample
    predictor. With ``exclude_self=True`` (for training rows themselves) the
    nearest neighbour — the point itself — is dropped.
    """
    tree = cKDTree(train_xy)
    kk = k + 1 if exclude_self else k
    _, idx = tree.query(query_xy, k=kk)
    if exclude_self:
        idx = idx[:, 1:]
    return train_y[idx].mean(axis=1)


# --------------------------------------------------------------------------
# Metrics and tests
# --------------------------------------------------------------------------
def regression_metrics(y_ln: np.ndarray, yhat_ln: np.ndarray) -> dict:
    """RMSE/MAE on the log response, MAPE on the price level, R² on logs."""
    resid = y_ln - yhat_ln
    ape = np.abs(np.exp(yhat_ln) - np.exp(y_ln)) / np.exp(y_ln)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_ln - y_ln.mean()) ** 2))
    return {
        "n": int(len(y_ln)),
        "rmse_ln": float(np.sqrt(np.mean(resid ** 2))),
        "mae_ln": float(np.mean(np.abs(resid))),
        "mape_pct": float(100 * ape.mean()),
        "median_ape_pct": float(100 * np.median(ape)),
        "r2_ln": float(1 - ss_res / ss_tot),
    }


def moran_of_residuals(resid: np.ndarray, w: KNN,
                       permutations: int = 0) -> dict:
    """Moran's I of residuals; analytical z by default (permutations=0)."""
    m = Moran(np.asarray(resid, dtype=float), w,
              permutations=permutations)
    out = {"I": float(m.I), "z_norm": float(m.z_norm),
           "p_norm": float(m.p_norm)}
    if permutations:
        out["p_sim"] = float(m.p_sim)
    return out


def paired_wilcoxon(per_fold_a: list[float], per_fold_b: list[float]) -> dict:
    """Paired Wilcoxon signed-rank test on per-fold metric pairs (A vs B)."""
    a, b = np.asarray(per_fold_a), np.asarray(per_fold_b)
    stat, p = stats.wilcoxon(a, b)
    return {"n_folds": int(len(a)), "statistic": float(stat),
            "p_value": float(p), "mean_diff": float((a - b).mean())}


def block_bootstrap_diff(y_ln: np.ndarray,
                         yhat_a: np.ndarray, yhat_b: np.ndarray,
                         blocks: np.ndarray, metric: str = "rmse_ln",
                         n_boot: int = 2000, seed: int = SEED) -> dict:
    """Bootstrap CI of the pooled metric difference (A − B), resampling
    spatial blocks with replacement — respects spatial dependence better
    than resampling rows."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    by_block = {b: np.flatnonzero(blocks == b) for b in uniq}

    def pooled(idx: np.ndarray, yhat: np.ndarray) -> float:
        return regression_metrics(y_ln[idx], yhat[idx])[metric]

    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_block[b] for b in sample])
        diffs[i] = pooled(idx, yhat_a) - pooled(idx, yhat_b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    full = np.arange(len(y_ln))
    return {"metric": metric, "n_boot": int(n_boot),
            "diff": float(pooled(full, yhat_a) - pooled(full, yhat_b)),
            "ci95": [float(lo), float(hi)],
            "p_boot_two_sided": float(2 * min((diffs > 0).mean(),
                                              (diffs < 0).mean()))}


# --------------------------------------------------------------------------
# Cross-validation runner
# --------------------------------------------------------------------------
def run_cv(model_factory, df: pd.DataFrame, label: str,
           seeds: tuple[int, ...] = (SEED,),
           w: KNN | None = None,
           moran_permutations: int = 0,
           out_dir: str | Path = "results") -> dict:
    """Run spatial-block CV for one model over one or more seeds.

    Returns (and writes to ``results/cv_<label>.json``) a result dict with,
    per seed: pooled metrics, per-fold metrics, Moran's I of the pooled
    out-of-fold residuals, and the out-of-fold predictions themselves
    (saved separately as ``results/oof_<label>_seed<seed>.csv`` for the
    pairwise tests and error analysis).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = spatial_folds(df)
    y = df[TARGET].to_numpy(dtype=float)
    if w is None:
        w = knn_weights(jittered_coords(df))

    result = {"label": label, "n": int(len(df)), "seeds": list(seeds),
              "folds": [{"block": b, "n_test": int(len(te))}
                        for b, _, te in folds],
              "knn_k": KNN_K, "jitter_m": JITTER_M,
              "per_seed": {}}

    for seed in seeds:
        t0 = time.time()
        yhat = np.full(len(df), np.nan)
        per_fold = []
        for b, tr, te in folds:
            model = model_factory(seed)
            model.fit(df.iloc[tr])
            pred = np.asarray(model.predict(df.iloc[te]), dtype=float)
            yhat[te] = pred
            fm = regression_metrics(y[te], pred)
            fm["block"] = b
            per_fold.append(fm)
        assert not np.isnan(yhat).any(), "some rows never predicted"

        pooled = regression_metrics(y, yhat)
        moran = moran_of_residuals(y - yhat, w,
                                   permutations=moran_permutations)
        result["per_seed"][str(seed)] = {
            "pooled": pooled, "per_fold": per_fold, "moran_oof": moran,
            "runtime_s": round(time.time() - t0, 1)}

        oof = df[["sql", "tipo_imovel", BLOCK_COL]].copy()
        oof["y_ln"] = y
        oof["yhat_ln"] = yhat
        oof.to_csv(out_dir / f"oof_{label}_seed{seed}.csv", index=False)

    # cross-seed summary (CI over seeds, when more than one)
    if len(seeds) > 1:
        summ = {}
        for m in ["rmse_ln", "mae_ln", "mape_pct", "r2_ln"]:
            vals = np.array([result["per_seed"][str(s)]["pooled"][m]
                             for s in seeds])
            half = 1.96 * vals.std(ddof=1) / np.sqrt(len(vals))
            summ[m] = {"mean": float(vals.mean()),
                       "ci95": [float(vals.mean() - half),
                                float(vals.mean() + half)]}
        result["seed_summary"] = summ

    (out_dir / f"cv_{label}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    return result


def compare_models(label_a: str, label_b: str, df: pd.DataFrame,
                   seed: int = SEED, metric: str = "rmse_ln",
                   out_dir: str | Path = "results") -> dict:
    """Paired Wilcoxon + block bootstrap between two finished CV runs."""
    out_dir = Path(out_dir)
    y = df[TARGET].to_numpy(dtype=float)
    blocks = df[BLOCK_COL].to_numpy()

    def read(label):
        res = json.loads((out_dir / f"cv_{label}.json").read_text())
        oof = pd.read_csv(out_dir / f"oof_{label}_seed{seed}.csv")
        per_fold = [f[metric] for f in res["per_seed"][str(seed)]["per_fold"]]
        return per_fold, oof["yhat_ln"].to_numpy(dtype=float)

    pf_a, yhat_a = read(label_a)
    pf_b, yhat_b = read(label_b)
    out = {"a": label_a, "b": label_b, "metric": metric, "seed": seed,
           "wilcoxon": paired_wilcoxon(pf_a, pf_b),
           "block_bootstrap": block_bootstrap_diff(y, yhat_a, yhat_b,
                                                   blocks, metric)}
    suffix = "" if metric == "rmse_ln" else f"_{metric}"
    (out_dir / f"compare_{label_a}_vs_{label_b}{suffix}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    return out


# --------------------------------------------------------------------------
# Smoke test — trivial baseline through the full machinery
# --------------------------------------------------------------------------
class MedianByTypeModel:
    """Predicts the training median ln_vu of the property type (fallback:
    global median). Exists only to exercise the protocol end to end."""

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit(self, train_df: pd.DataFrame):
        self.by_type_ = train_df.groupby("tipo_imovel")[TARGET].median()
        self.global_ = float(train_df[TARGET].median())
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return (test_df["tipo_imovel"].map(self.by_type_)
                .fillna(self.global_).to_numpy(dtype=float))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--data", default="data/itbi_sp_2025_level_a.csv")
    ap.add_argument("--financed-only", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="run the median-by-type baseline through the protocol")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = load_base(args.data, financed_only=args.financed_only)
    print(f"Base: {len(df):,} rows | blocks: {df[BLOCK_COL].nunique()}"
          + (" | financed only" if args.financed_only else ""))

    if args.smoke:
        label = "smoke_median_by_type" + ("_fin" if args.financed_only else "")
        res = run_cv(MedianByTypeModel, df, label=label,
                     seeds=tuple(args.seeds), out_dir=args.out)
        pooled = res["per_seed"][str(args.seeds[0])]["pooled"]
        moran = res["per_seed"][str(args.seeds[0])]["moran_oof"]
        print(f"Smoke baseline pooled: RMSE_ln={pooled['rmse_ln']:.4f} "
              f"MAE_ln={pooled['mae_ln']:.4f} MAPE={pooled['mape_pct']:.1f}% "
              f"R2_ln={pooled['r2_ln']:.3f} | Moran I={moran['I']:.3f} "
              f"(z={moran['z_norm']:.1f})")
        print(f"Wrote {args.out}/cv_{label}.json")


if __name__ == "__main__":
    main()
