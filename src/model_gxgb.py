#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_gxgb.py — geographically weighted XGBoost, calibrated at regression points
===============================================================================
The second non-linear spatial baseline: Geographical-XGBoost (G-XGBoost,
Grekousis 2025, *Journal of Geographical Systems*; package ``geoxgboost``),
the tree-based analogue of geographically weighted regression and one of the
methods of the reference study on Belo Horizonte.

The estimator, as in ``geoxgboost``
-----------------------------------
* **Local models.** Around a location, an XGBoost is fitted on the ``k``
  nearest training rows (adaptive bandwidth), each weighted by the bi-square
  kernel ``(1 - d²/h²)²`` with ``h`` the distance to the k-th neighbour.
* **Features.** The hedonic design matrix shared by OLS/SAR/XGBoost
  (``model_sar.design_matrix``). Location enters only through the
  geographic weighting — no coordinates, no lag — which is what makes the
  model *geographically weighted* rather than a global learner that sees
  space.
* **Ensemble with a global model.** The local prediction is combined with
  the prediction of a global XGBoost on the same features:
  ``ŷ = α·ŷ_local + (1 − α)·ŷ_global`` (``alpha_wt`` in ``geoxgboost``).
* **Bandwidth by cross-validation** (``optimize_bw`` in ``geoxgboost``).

What changes, and why
---------------------
``geoxgboost`` calibrates one local model — with its own GridSearchCV — at
*every training row*, over a dense n × n distance matrix. At 19–21k rows per
fold that is ~0.5 h per bandwidth candidate per fold and ~2.9 GB of matrix
(``src/pilot_gxgb_timing.py``, ``results/gxgb_timing_pilot.json``). Here the
local models are calibrated at **regression points**, as geographically
weighted regression allows (calibration locations need not be data points;
Fotheringham, Brunsdon & Charlton 2002; ``regression.points`` in GWmodel):

* ``M`` regression points = K-means centres of the training coordinates
  (``M = 1000``, about one per 1.5 km² of the city);
* one local XGBoost per regression point, with shared hyperparameters
  instead of a grid search per local model;
* a query location is predicted by the local models of its 3 nearest
  regression points, averaged with inverse-distance weights, then blended
  with the global model;
* k-d trees instead of a dense distance matrix.

The bandwidth ``k`` and the weight ``α`` are chosen **nested**, like the
XGBoost baseline: for each outer fold, an inner CV over the training blocks
(3 inner folds, three blocks each) scores every (k, α) by pooled inner
RMSE_ln; the outer test block never touches the choice. Candidates:
``k ∈ {50, 100, 200, 400, 800, 1600, 3200} × n_train / 24,000`` (the same spatial reach on
the 82k base as on Level A) and ``α ∈ {0, 0.25, 0.5, 0.75, 1}`` (the grid was widened downward, from
400 to 50, after the first searches kept choosing its smallest value; 50
neighbours is taken as the floor, the smallest sample on which the local
model is still a model, and the choice is still made on the inner CV alone). The choice is
cached in ``results/gxgb_params_<label>.json``, so extra seeds reuse it.

Run:
    python -m src.model_gxgb --data data/itbi_sp_2025_level_a.csv --seeds 42 43 44
    python -m src.out_of_time --model gxgb --train level_a --seeds 42 43 44
    python -m src.out_of_time --model gxgb --train full --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from .model_sar import X_NAMES, design_matrix
from .protocol import (BLOCK_COL, SEED, jittered_coords, load_base,
                       regression_metrics, run_cv)

TARGET = "ln_vu"
N_POINTS = 1000                       # regression points
N_NEAR = 3                            # regression points blended per query
BASE_BANDWIDTHS = (50, 100, 200, 400, 800, 1600, 3200)   # neighbours, for 24,000 training rows
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
INNER_FOLDS = 3
N_WORKERS = 2
CACHE_ROOT = Path("results/gxgb_cache")   # resumable intermediate predictions (not versioned)

LOCAL_PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
                    min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                    reg_lambda=1.0)
GLOBAL_PARAMS = dict(n_estimators=800, learning_rate=0.05, max_depth=6,
                     min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                     reg_lambda=1.0)


def bandwidth_grid(n_train: int) -> list[int]:
    return [int(round(b * n_train / 24000)) for b in BASE_BANDWIDTHS]


def bisquare(d: np.ndarray) -> np.ndarray:
    """Adaptive bi-square weights; h = distance to the farthest neighbour."""
    h = d.max() * (1 + 1e-9) + 1e-9
    return (1.0 - (d / h) ** 2) ** 2


class GWXGBoost:
    """Geographically weighted XGBoost at regression points (see module doc)."""

    def __init__(self, seed: int = SEED, bandwidth: int = 1600, alpha: float = 0.5,
                 n_points: int = N_POINTS, n_near: int = N_NEAR,
                 n_workers: int = N_WORKERS):
        self.seed, self.bandwidth, self.alpha = seed, bandwidth, alpha
        self.n_points, self.n_near, self.n_workers = n_points, n_near, n_workers

    # -- pieces ------------------------------------------------------------
    def _fit_global(self, X, y):
        return xgb.XGBRegressor(tree_method="hist", n_jobs=self.n_workers,
                                random_state=self.seed, verbosity=0,
                                **GLOBAL_PARAMS).fit(X, y)

    def _regression_points(self, xy):
        m = min(self.n_points, len(np.unique(xy.round(0), axis=0)))
        km = KMeans(n_clusters=m, n_init=1, random_state=self.seed).fit(xy)
        return km.cluster_centers_

    def _fit_locals(self, X, y, tree, points, k):
        k = min(k, len(y))

        def fit_one(p):
            d, idx = tree.query(p, k=k)
            return xgb.XGBRegressor(tree_method="hist", n_jobs=1,
                                    random_state=self.seed, verbosity=0,
                                    **LOCAL_PARAMS).fit(X[idx], y[idx],
                                                        sample_weight=bisquare(d))

        with ThreadPoolExecutor(max_workers=self.n_workers) as ex:
            return list(ex.map(fit_one, points))

    def _local_predict(self, models, points, Xq, xyq):
        """IDW blend of the local models of the n_near nearest regression points."""
        d, idx = cKDTree(points).query(xyq, k=self.n_near)
        d, idx = np.atleast_2d(d), np.atleast_2d(idx)
        w = 1.0 / (d + 1.0)
        w /= w.sum(axis=1, keepdims=True)
        pred = np.zeros(len(xyq))
        for j in np.unique(idx):
            rows, cols = np.nonzero(idx == j)
            pred[rows] += w[rows, cols] * models[j].predict(Xq[rows])
        return pred

    # -- protocol interface --------------------------------------------------
    def fit(self, train_df: pd.DataFrame):
        X = design_matrix(train_df)
        y = train_df[TARGET].to_numpy(dtype=float)
        xy = jittered_coords(train_df, seed=self.seed)
        self.points_ = self._regression_points(xy)
        self.global_ = self._fit_global(X, y)
        self.locals_ = self._fit_locals(X, y, cKDTree(xy), self.points_, self.bandwidth)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = design_matrix(test_df)
        xy = jittered_coords(test_df, seed=self.seed)
        g = self.global_.predict(X).astype(float)
        if self.alpha == 0:
            return g
        loc = self._local_predict(self.locals_, self.points_, X, xy)
        return self.alpha * loc + (1 - self.alpha) * g


# --------------------------------------------------------------------------
# Nested choice of bandwidth and alpha
# --------------------------------------------------------------------------
def inner_splits(blocks: np.ndarray, n_folds: int = INNER_FOLDS):
    """Group the training blocks into n_folds inner folds (sorted, round-robin)."""
    uniq = np.sort(np.unique(blocks))
    for f in range(n_folds):
        held = uniq[f::n_folds]
        yield np.flatnonzero(~np.isin(blocks, held)), np.flatnonzero(np.isin(blocks, held))


def _cached(path: Path | None, compute):
    """np.load(path) if it exists, else compute(), save and return."""
    if path is not None and path.exists():
        return np.load(path)
    arr = np.asarray(compute(), dtype=float)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, arr)
    return arr


def tune_on(train_df: pd.DataFrame, seed: int = SEED,
            cache_dir: Path | None = None) -> dict:
    """Pooled inner-CV RMSE_ln for every (bandwidth, alpha); returns the best.

    With ``cache_dir`` every inner prediction (global model, and local models
    per bandwidth) is saved as it is produced, so an interrupted search
    resumes where it stopped."""
    train_df = train_df.reset_index(drop=True)
    X = design_matrix(train_df)
    y = train_df[TARGET].to_numpy(dtype=float)
    xy = jittered_coords(train_df, seed=seed)
    blocks = train_df[BLOCK_COL].to_numpy()
    grid = bandwidth_grid(len(train_df))

    y_all, g_all = [], []
    loc_all = {k: [] for k in grid}
    for f, (tr, va) in enumerate(inner_splits(blocks)):
        m = GWXGBoost(seed=seed)
        c = (lambda name: cache_dir / f"inner{f}_{name}.npy") if cache_dir else (lambda name: None)
        g_all.append(_cached(c("global"), lambda: m._fit_global(X[tr], y[tr]).predict(X[va])))
        state = {}
        for k in grid:
            def local_pred(k=k):
                if not state:
                    state["points"] = m._regression_points(xy[tr])
                    state["tree"] = cKDTree(xy[tr])
                models = m._fit_locals(X[tr], y[tr], state["tree"], state["points"], k)
                return m._local_predict(models, state["points"], X[va], xy[va])
            loc_all[k].append(_cached(c(f"k{k}"), local_pred))
        y_all.append(y[va])

    y_all, g_all = np.concatenate(y_all), np.concatenate(g_all)
    scores = {}
    for k in grid:
        loc = np.concatenate(loc_all[k])
        for a in ALPHAS:
            scores[f"{k}|{a}"] = regression_metrics(y_all, a * loc + (1 - a) * g_all)["rmse_ln"]
    best = min(scores, key=scores.get)
    k, a = best.split("|")
    return {"bandwidth": int(k), "alpha": float(a), "inner_rmse_ln": scores[best],
            "inner_scores": scores, "bandwidth_grid": grid, "alphas": list(ALPHAS),
            "inner_folds": INNER_FOLDS, "n_points": N_POINTS, "n_near": N_NEAR}


class TunedGWXGBFactory:
    """``factory(seed)`` for ``run_cv``: tunes once per outer fold (cached by
    the set of training blocks), then fits with the chosen (k, alpha)."""

    def __init__(self, label: str, out_dir: Path):
        self.label = label
        self.cache_path = out_dir / f"gxgb_params_{label}.json"
        self.cache = (json.loads(self.cache_path.read_text())
                      if self.cache_path.exists() else {})

    def __call__(self, seed: int):
        return _OuterFold(self, seed)


class _OuterFold:
    def __init__(self, factory: TunedGWXGBFactory, seed: int):
        self.f, self.seed = factory, seed

    def fit(self, train_df: pd.DataFrame):
        key = ",".join(str(b) for b in sorted(train_df[BLOCK_COL].unique()))
        tag = "blocks" + key.replace(",", "")
        self.pred_path = CACHE_ROOT / self.f.label / f"{tag}_seed{self.seed}_pred.npy"
        if self.pred_path.exists():
            return self
        if key not in self.f.cache:
            t0 = time.time()
            res = tune_on(train_df, seed=SEED,
                          cache_dir=CACHE_ROOT / self.f.label / f"tune_{tag}")
            res["tuning_s"] = round(time.time() - t0, 1)
            self.f.cache[key] = res
            self.f.cache_path.write_text(json.dumps(self.f.cache, indent=2))
            print(f"  training blocks {key}: k={res['bandwidth']} alpha={res['alpha']} "
                  f"inner RMSE_ln={res['inner_rmse_ln']:.4f} ({res['tuning_s']:.0f}s)", flush=True)
        p = self.f.cache[key]
        self.m = GWXGBoost(seed=self.seed, bandwidth=p["bandwidth"],
                           alpha=p["alpha"]).fit(train_df)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _cached(self.pred_path, lambda: self.m.predict(test_df))


class CachedHoldout:
    """A fitted-once hold-out run whose predictions are saved (resumable)."""

    def __init__(self, pred_path: Path, model: GWXGBoost):
        self.pred_path, self.model = pred_path, model

    def fit(self, train_df: pd.DataFrame):
        if not self.pred_path.exists():
            self.model.fit(train_df)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return _cached(self.pred_path, lambda: self.model.predict(test_df))


def main() -> None:
    ap = argparse.ArgumentParser(description="Geographically weighted XGBoost through the protocol")
    ap.add_argument("--data", default="data/itbi_sp_2025_level_a.csv")
    ap.add_argument("--label", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    ap.add_argument("--financed-only", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = load_base(args.data, financed_only=args.financed_only)
    label = args.label or ("gxgb_" + Path(args.data).stem.replace("itbi_sp_", "")
                           + ("_fin" if args.financed_only else ""))
    print(f"GW-XGBoost: {len(df):,} rows -> label '{label}' | {N_POINTS} regression points, "
          f"bandwidths {bandwidth_grid(int(len(df) * 0.9))} (inner), alphas {ALPHAS} | "
          f"seeds {args.seeds}", flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    factory = TunedGWXGBFactory(label, out_dir)
    t0 = time.time()
    res = run_cv(factory, df, label=label, seeds=tuple(args.seeds), out_dir=out_dir)
    s0 = str(args.seeds[0])
    pooled, moran = res["per_seed"][s0]["pooled"], res["per_seed"][s0]["moran_oof"]
    print(f"CV done in {time.time() - t0:,.0f}s | RMSE_ln={pooled['rmse_ln']:.4f} "
          f"MAPE={pooled['mape_pct']:.1f}% R2_ln={pooled['r2_ln']:.3f} | Moran I={moran['I']:.3f}")
    meta = json.loads((out_dir / f"cv_{label}.json").read_text())
    meta["features"] = X_NAMES
    meta["design"] = {"estimator": "geographically weighted XGBoost (Grekousis 2025) at "
                                   "regression points", "n_points": N_POINTS, "n_near": N_NEAR,
                      "kernel": "adaptive bi-square", "local_params": LOCAL_PARAMS,
                      "global_params": GLOBAL_PARAMS, "params_file": str(factory.cache_path)}
    (out_dir / f"cv_{label}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
