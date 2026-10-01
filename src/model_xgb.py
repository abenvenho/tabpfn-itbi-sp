#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_xgb.py — gradient-boosting baseline with fold-internal spatial features
=============================================================================
Global XGBoost with spatial context, the machine-learning baseline of the
comparison (SAR vs boosting vs TabPFN-3.5).

Why a global learner here, and where the geographically weighted one is
------------------------------------------------------------------------
The reference study on Belo Horizonte used G-XGBoost (Grekousis 2025,
``pip install geoxgboost``). As shipped, ``optimize_bw`` fits one local
XGBoost with a GridSearchCV per *training row and per bandwidth candidate*
over a dense n x n distance matrix: on a 200-row subsample one candidate took
18.7 s (0.093 s per spatial unit), so a real fold of ~19-21k rows costs
~0.5 h per candidate and ~2.9 GB of distance matrix
(``results/gxgb_timing_pilot.json``). The geographically weighted estimator
is therefore run in ``src/model_gxgb.py``, calibrated at regression points
instead of at every row, as a second, separate baseline.

This module is the other kind of spatial learner: one global XGBoost that is
*shown* space through fold-internal features.

Its spatial inputs:

* **Spatial-lag feature**: mean ``ln_vu`` of the 8 nearest TRAINING
  neighbours (``protocol.knn_spatial_lag``); for training rows the point
  itself is excluded (leave-one-out lag) so the feature is honest; for test
  rows only training neighbours are used. Built inside each fold.
* **Rotated coordinates**: UTM x, y plus their projections on 30°, 45° and
  60° axes, so axis-aligned tree splits can carve oblique spatial regions
  (a standard trick for boosted trees on coordinates).
* **Hedonic block**: the same design matrix as the SAR/OLS baselines
  (``model_sar.design_matrix``), so the three models share the same
  information set apart from how space enters.

Hyperparameters: **nested** Optuna tuning under spatial-block CV. For each
outer fold, an inner leave-one-block-out CV over the 9 training blocks
selects the hyperparameters (objective: pooled inner RMSE on ln_vu, with
early stopping on the held-out inner block); the model is then refitted on
the whole outer training set and evaluated once on the outer test block.
The outer test block never touches tuning. Tuned parameters are cached in
``results/xgb_params_<label>.json`` so multi-seed runs reuse them.

Run (Level A):
    python -m src.model_xgb --data data/itbi_sp_2025_level_a.csv
    python -m src.model_xgb --data data/itbi_sp_2025_level_a.csv --no-lag   # ablation
    python -m src.model_xgb --seeds 42 43 44                                   # CI over seeds
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb

from .model_sar import X_NAMES, design_matrix
from .protocol import (BLOCK_COL, SEED, jittered_coords, knn_spatial_lag,
                       load_base, regression_metrics, run_cv)

TARGET = "ln_vu"
ROTATIONS_DEG = (30.0, 45.0, 60.0)
COORD_NAMES = ["x_utm", "y_utm"] + [f"rot{int(a)}" for a in ROTATIONS_DEG]
LAG_NAME = "knn8_lag_ln_vu"

DEFAULT_PARAMS = dict(n_estimators=800, learning_rate=0.05, max_depth=6,
                      min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                      reg_lambda=1.0, reg_alpha=0.0)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------
def rotated_coords(xy: np.ndarray) -> np.ndarray:
    """[x, y, x cos a + y sin a for a in ROTATIONS_DEG]."""
    cols = [xy[:, 0], xy[:, 1]]
    for a in ROTATIONS_DEG:
        r = np.deg2rad(a)
        cols.append(xy[:, 0] * np.cos(r) + xy[:, 1] * np.sin(r))
    return np.column_stack(cols)


def feature_names(use_lag: bool = True) -> list[str]:
    return X_NAMES + COORD_NAMES + ([LAG_NAME] if use_lag else [])


class XGBSpatialModel:
    """Global XGBoost on hedonic + coordinate features + fold-internal lag."""

    def __init__(self, seed: int = SEED, params: dict | None = None,
                 use_lag: bool = True, k: int = 8, n_jobs: int = 2):
        self.seed = seed
        self.params = dict(DEFAULT_PARAMS if params is None else params)
        self.use_lag = use_lag
        self.k = k
        self.n_jobs = n_jobs

    # -- feature construction -------------------------------------------
    def _features(self, df: pd.DataFrame, xy: np.ndarray,
                  lag: np.ndarray | None) -> np.ndarray:
        blocks = [design_matrix(df), rotated_coords(xy)]
        if self.use_lag:
            blocks.append(lag[:, None])
        return np.column_stack(blocks)

    def train_features(self, train_df: pd.DataFrame):
        xy = jittered_coords(train_df, seed=self.seed)
        y = train_df[TARGET].to_numpy(dtype=float)
        lag = (knn_spatial_lag(xy, y, xy, k=self.k, exclude_self=True)
               if self.use_lag else None)
        return self._features(train_df, xy, lag), y, xy

    def test_features(self, test_df: pd.DataFrame, train_xy: np.ndarray,
                      train_y: np.ndarray) -> np.ndarray:
        xy = jittered_coords(test_df, seed=self.seed)
        lag = (knn_spatial_lag(train_xy, train_y, xy, k=self.k)
               if self.use_lag else None)
        return self._features(test_df, xy, lag)

    # -- sklearn-like interface used by the protocol ---------------------
    def _make(self, **override) -> xgb.XGBRegressor:
        p = {**self.params, **override}
        return xgb.XGBRegressor(tree_method="hist", n_jobs=self.n_jobs,
                                random_state=self.seed, verbosity=0, **p)

    def fit(self, train_df: pd.DataFrame):
        X, y, xy = self.train_features(train_df)
        self._train_xy, self._train_y = xy, y
        self.model_ = self._make().fit(X, y)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = self.test_features(test_df, self._train_xy, self._train_y)
        return self.model_.predict(X).astype(float)


# --------------------------------------------------------------------------
# Nested tuning (inner leave-one-block-out on the outer training set)
# --------------------------------------------------------------------------
def tune_on(train_df: pd.DataFrame, n_trials: int, seed: int = SEED,
            use_lag: bool = True, max_rounds: int = 3000,
            early_stopping: int = 100) -> dict:
    """Optuna search under inner leave-one-block-out CV.

    Each trial trains on 8 of the 9 training blocks with early stopping on
    the 9th (rotating), and is scored by the pooled inner RMSE_ln. The
    returned parameter set uses the median best iteration across inner
    folds (scaled by 9/8 for the refit on all nine blocks).
    """
    train_df = train_df.reset_index(drop=True)
    blocks = train_df[BLOCK_COL].to_numpy()
    uniq = np.unique(blocks)

    # inner-fold features are built ONCE per inner fold (they do not depend
    # on the hyperparameters), which keeps the search cheap
    proto = XGBSpatialModel(seed=seed, use_lag=use_lag)
    inner = []
    for b in uniq:
        tr = np.flatnonzero(blocks != b)
        va = np.flatnonzero(blocks == b)
        Xtr, ytr, xytr = proto.train_features(train_df.iloc[tr])
        Xva = proto.test_features(train_df.iloc[va], xytr, ytr)
        inner.append((Xtr, ytr, Xva, train_df[TARGET].to_numpy(float)[va]))

    def objective(trial: optuna.Trial) -> float:
        p = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            gamma=trial.suggest_float("gamma", 1e-4, 1.0, log=True),
        )
        yhat_all, y_all, best_its = [], [], []
        for Xtr, ytr, Xva, yva in inner:
            m = xgb.XGBRegressor(tree_method="hist", n_jobs=2, random_state=seed,
                                 verbosity=0, n_estimators=max_rounds,
                                 early_stopping_rounds=early_stopping, **p)
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            yhat_all.append(m.predict(Xva))
            y_all.append(yva)
            best_its.append(int(m.best_iteration) + 1)
        rmse = regression_metrics(np.concatenate(y_all),
                                  np.concatenate(yhat_all))["rmse_ln"]
        trial.set_user_attr("best_iterations", best_its)
        return rmse

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params)
    its = study.best_trial.user_attrs["best_iterations"]
    best["n_estimators"] = int(round(np.median(its) * len(uniq) / (len(uniq) - 1)))
    return {"params": best, "inner_rmse_ln": float(study.best_value),
            "inner_best_iterations": its, "n_trials": n_trials}


class TunedXGBFactory:
    """``factory(seed)`` for ``run_cv``: tunes once per outer fold (cached)."""

    def __init__(self, label: str, n_trials: int, use_lag: bool,
                 out_dir: Path, folds_order: list[int]):
        self.label, self.n_trials, self.use_lag = label, n_trials, use_lag
        self.cache_path = out_dir / f"xgb_params_{label}.json"
        self.cache = (json.loads(self.cache_path.read_text())
                      if self.cache_path.exists() else {})
        self.folds_order = folds_order          # outer folds in run_cv order
        self.call = 0

    def __call__(self, seed: int):
        block = str(self.folds_order[self.call % len(self.folds_order)])
        self.call += 1
        outer = _OuterFoldModel(self, block, seed)
        return outer


class _OuterFoldModel:
    """Tunes on the outer training set (once, cached by block), then fits."""

    def __init__(self, factory: TunedXGBFactory, block: str, seed: int):
        self.f, self.block, self.seed = factory, block, seed

    def fit(self, train_df: pd.DataFrame):
        if self.block not in self.f.cache:
            t0 = time.time()
            res = tune_on(train_df, self.f.n_trials, seed=SEED,
                          use_lag=self.f.use_lag)
            res["tuning_s"] = round(time.time() - t0, 1)
            self.f.cache[self.block] = res
            self.f.cache_path.write_text(json.dumps(self.f.cache, indent=2))
            print(f"  fold block={self.block}: tuned in {res['tuning_s']:.0f}s "
                  f"inner RMSE_ln={res['inner_rmse_ln']:.4f} "
                  f"n_estimators={res['params']['n_estimators']}", flush=True)
        params = self.f.cache[self.block]["params"]
        self.m = XGBSpatialModel(seed=self.seed, params=params,
                                 use_lag=self.f.use_lag).fit(train_df)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        return self.m.predict(test_df)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="XGBoost + spatial features through the protocol")
    ap.add_argument("--data", default="data/itbi_sp_2025_level_a.csv")
    ap.add_argument("--label", default=None)
    ap.add_argument("--no-lag", action="store_true", help="ablation without the k-NN lag")
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    ap.add_argument("--financed-only", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = load_base(args.data, financed_only=args.financed_only)
    use_lag = not args.no_lag
    label = args.label or ("xgb" + ("_lag" if use_lag else "_nolag") + "_"
                           + Path(args.data).stem.replace("itbi_sp_", "")
                           + ("_fin" if args.financed_only else ""))
    print(f"XGB{' + lag' if use_lag else ' (no lag)'}: {len(df):,} rows -> label '{label}' "
          f"| nested Optuna {args.n_trials} trials | seeds {args.seeds}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    folds_order = sorted(df[BLOCK_COL].unique().tolist())
    factory = TunedXGBFactory(label, args.n_trials, use_lag, out_dir, folds_order)

    t0 = time.time()
    res = run_cv(factory, df, label=label, seeds=tuple(args.seeds), out_dir=out_dir)
    s0 = str(args.seeds[0])
    pooled, moran = res["per_seed"][s0]["pooled"], res["per_seed"][s0]["moran_oof"]
    print(f"CV done in {time.time() - t0:,.0f}s | RMSE_ln={pooled['rmse_ln']:.4f} "
          f"MAE_ln={pooled['mae_ln']:.4f} MAPE={pooled['mape_pct']:.1f}% "
          f"R2_ln={pooled['r2_ln']:.3f} | Moran I={moran['I']:.3f} (z={moran['z_norm']:.1f})")
    if "seed_summary" in res:
        ss = res["seed_summary"]["rmse_ln"]
        print(f"RMSE_ln over seeds: {ss['mean']:.4f} CI95 [{ss['ci95'][0]:.4f}, {ss['ci95'][1]:.4f}]")

    # record the feature set and tuning design in the result file
    meta = json.loads((out_dir / f"cv_{label}.json").read_text())
    meta["features"] = feature_names(use_lag)
    meta["tuning"] = {"method": "nested Optuna TPE, inner leave-one-block-out on the "
                                "outer training blocks, early stopping on the inner "
                                "held-out block", "n_trials": args.n_trials,
                      "params_file": str(factory.cache_path)}
    (out_dir / f"cv_{label}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
