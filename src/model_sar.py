#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_sar.py — spatial autoregressive (SAR) lag baseline
=========================================================
Hedonic SAR lag model, the spatial-econometrics baseline of the comparison,
estimated with ``spreg`` (PySAL) on a k-NN-8 row-standardised weights matrix
built **within each training fold** (the test block never enters W).

Specification (pooled across property types, with type dummies):

    ln_vu = rho * W ln_vu + b0 + X b + e

    X = [ln(built area), ln(1 + lot area), age, age^2, finish grade,
         ln(1 + distance to nearest rail/subway station), month index,
         house dummy, commercial dummy]        (apartment = reference)

Out-of-sample prediction follows the reference study: for a test point,
``yhat = X beta + rho * (mean ln_vu of its 8 nearest TRAINING neighbours)``
— the conditional-on-observed-neighbours predictor, using
``protocol.knn_spatial_lag`` (same deterministic sub-metre jitter).

Estimator: the reference study used maximum likelihood on 5,005 rows. Here
each training fold holds ~21k rows and ``ML_Lag``'s inference builds the
DENSE inverse of (I - rho W) — an n x n matrix (~3.5 GB at this size) — which
exhausts memory; this was measured, not assumed. The default is therefore
``GM_Lag`` (Kelejian–Prucha two-stage least squares with spatially lagged
instruments), the standard estimator for large samples; ``--estimator ml``
remains available for smaller bases. An OLS hedonic with the same design
matrix (``--estimator ols``) provides the non-spatial reference, as in the
study. All estimators are deterministic — the protocol seed only affects the
k-NN tie-breaking jitter — so one seed suffices.

Run (Level A):
    python -m src.model_sar --data data/itbi_sp_2025_level_a.csv
    python -m src.model_sar --data data/itbi_sp_2025_level_a.csv --estimator ols
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from spreg import GM_Lag, ML_Lag

from .protocol import (SEED, jittered_coords, knn_spatial_lag, knn_weights,
                       load_base, run_cv)

TARGET = "ln_vu"
TYPE_REF = "apartment"          # reference category for the type dummies
TYPE_DUMMIES = ["house", "commercial"]

X_NAMES = ["ln_built_area", "ln1p_lot_area", "age", "age_sq",
           "finish_grade", "ln1p_dist_station", "month_index",
           "d_house", "d_commercial"]


def design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Hedonic design matrix (no constant — spreg adds it)."""
    grade = pd.to_numeric(df["padrao_nivel"], errors="coerce")
    grade = grade.fillna(grade.median())
    age = df["idade"].to_numpy(dtype=float)
    X = np.column_stack([
        np.log(df["area_construida_m2"].to_numpy(dtype=float)),
        np.log1p(df["area_terreno_m2"].fillna(0).to_numpy(dtype=float)),
        age,
        age ** 2,
        grade.to_numpy(dtype=float),
        np.log1p(df["dist_estacao_m"].to_numpy(dtype=float)),
        df["mes_idx"].to_numpy(dtype=float),
        (df["tipo_imovel"] == "house").to_numpy(dtype=float),
        (df["tipo_imovel"] == "commercial").to_numpy(dtype=float),
    ])
    return X


class OLSHedonic:
    """Non-spatial OLS on the same design matrix (reference baseline)."""

    def __init__(self, seed: int = SEED):
        self.seed = seed

    def fit(self, train_df: pd.DataFrame):
        X = np.column_stack([np.ones(len(train_df)), design_matrix(train_df)])
        y = train_df[TARGET].to_numpy(dtype=float)
        self.betas_, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = np.column_stack([np.ones(len(test_df)), design_matrix(test_df)])
        return X @ self.betas_


class SARLagModel:
    """SAR lag model with fold-internal W and neighbour-lag prediction."""

    def __init__(self, seed: int = SEED, estimator: str = "gm", k: int = 8):
        self.seed = seed
        self.estimator = estimator
        self.k = k

    def fit(self, train_df: pd.DataFrame):
        y = train_df[TARGET].to_numpy(dtype=float)[:, None]
        X = design_matrix(train_df)
        self._train_xy = jittered_coords(train_df, seed=self.seed)
        self._train_y = y.ravel()
        w = knn_weights(self._train_xy, k=self.k)
        if self.estimator == "ml":
            m = ML_Lag(y, X, w=w, method="LU",
                       name_y=TARGET, name_x=X_NAMES)
        else:
            m = GM_Lag(y, X, w=w, w_lags=2,
                       name_y=TARGET, name_x=X_NAMES)
        betas = np.asarray(m.betas, dtype=float).ravel()
        self.betas_ = betas[:-1]             # constant + X coefficients
        self.rho_ = float(betas[-1])
        self.model_ = m
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        X = design_matrix(test_df)
        xb = self.betas_[0] + X @ self.betas_[1:]
        lag = knn_spatial_lag(self._train_xy, self._train_y,
                              jittered_coords(test_df, seed=self.seed),
                              k=self.k)
        return xb + self.rho_ * lag


def full_fit_summary(df: pd.DataFrame, estimator: str,
                     out_path: Path) -> dict:
    """Fit once on the full base and save the coefficient table."""
    if estimator == "ols":
        model = OLSHedonic().fit(df)
        lines = ["OLS hedonic — full-sample coefficients",
                 f"n = {len(df):,}", ""]
        for name, b in zip(["const"] + X_NAMES, model.betas_):
            lines.append(f"{name:>20s}  {b: .6f}")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return {"rho": None, "n": int(len(df)), "summary_file": str(out_path)}
    model = SARLagModel(estimator=estimator).fit(df)
    m = model.model_
    out_path.write_text(str(m.summary), encoding="utf-8")
    return {"rho": model.rho_,
            "pseudo_r2": float(getattr(m, "pr2", np.nan)),
            "n": int(m.n), "summary_file": str(out_path)}


def main() -> None:
    ap = argparse.ArgumentParser(description="SAR lag baseline through the protocol")
    ap.add_argument("--data", default="data/itbi_sp_2025_level_a.csv")
    ap.add_argument("--label", default=None)
    ap.add_argument("--estimator", choices=["gm", "ml", "ols"], default="gm")
    ap.add_argument("--financed-only", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = load_base(args.data, financed_only=args.financed_only)
    prefix = "ols" if args.estimator == "ols" else f"sar_{args.estimator}"
    label = args.label or (prefix + "_"
                           + Path(args.data).stem.replace("itbi_sp_", "")
                           + ("_fin" if args.financed_only else ""))
    print(f"{prefix.upper()}: {len(df):,} rows -> label '{label}'")

    if args.estimator == "ols":
        factory = lambda seed: OLSHedonic(seed=seed)          # noqa: E731
    else:
        factory = lambda seed: SARLagModel(seed=seed,          # noqa: E731
                                           estimator=args.estimator)
    t0 = time.time()
    res = run_cv(factory, df, label=label, seeds=(SEED,), out_dir=args.out)
    pooled = res["per_seed"][str(SEED)]["pooled"]
    moran = res["per_seed"][str(SEED)]["moran_oof"]
    rhos = "n/a"
    print(f"CV done in {time.time() - t0:,.0f}s | RMSE_ln={pooled['rmse_ln']:.4f} "
          f"MAE_ln={pooled['mae_ln']:.4f} MAPE={pooled['mape_pct']:.1f}% "
          f"R2_ln={pooled['r2_ln']:.3f} | Moran I={moran['I']:.3f} "
          f"(z={moran['z_norm']:.1f})")

    summ = full_fit_summary(df, args.estimator,
                            Path(args.out) / f"{label}_full_fit.txt")
    rho_txt = "n/a" if summ["rho"] is None else f"{summ['rho']:.4f}"
    print(f"Full-sample fit: rho={rho_txt} (summary -> {summ['summary_file']})")
    meta = json.loads((Path(args.out) / f"cv_{label}.json").read_text())
    meta["full_fit"] = summ
    (Path(args.out) / f"cv_{label}.json").write_text(json.dumps(meta, indent=2),
                                                     encoding="utf-8")


if __name__ == "__main__":
    main()
