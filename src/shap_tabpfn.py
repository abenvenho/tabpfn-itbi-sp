#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shap_tabpfn.py — SHAP attributions for the TabPFN-3.5 valuation model
=====================================================================
What an appraiser needs to defend an estimate is not only the number but
*why*: which attributes moved this property away from the typical unit value
and by how much. TabPFN-3.5 is served through an API, so the explanation is
model-agnostic: permutation SHAP (Shapley values estimated over random
feature orderings, ``shap.PermutationExplainer``) with an independent
background sample drawn from the training base.

The model is **not refitted**: the record of the zero-shot model fitted on
the 2025 Level A base for the out-of-time test
(``results/tabpfn_cache/oot_tabpfn_t0_level_a/blockall_seed42_model.json``)
is reloaded with ``TabPFNRegressor.load_model``. The rows explained are 2026
transactions — properties the model has never seen, valued with the model of
the previous year, which is exactly the appraisal use case.

Everything is small by design (API calls cost credits):

* ``data/shap_subsample.csv``     — the rows explained (default 50 from the
  2026 test base, stratified by property type in the Level A proportions);
* ``data/shap_background.csv``    — the background sample (default 20 rows
  of the training base, same stratification);
* one permutation forward and backward per row: ``2 * n_features + 1 = 19``
  masked evaluations plus shap's 10-mask probe of which inputs vary, each
  averaged over the background → at most 580 predicted rows per explained
  property (≈ 27,000 for the default 50 rows, two API calls per row);
  ``--n-permutations 2`` adds 380 rows per property.

Every API call is cached under ``results/shap/cache/`` keyed by the SHA-1 of
the masked input matrix, so a repeated run (same seed, same rows) costs
nothing and the attributions are reproducible offline.

Outputs (``results/shap/``):

* ``shap_values_<label>.csv``  — one row per explained property: ``sql``,
  type, block, observed and predicted ``ln_vu``, the base value (expected
  prediction over the background), and one SHAP column per feature, in ln
  units (``exp(shap)`` is the multiplicative effect on the unit value);
* ``shap_summary_<label>.json`` — mean |SHAP| per feature, overall and by
  property type, plus the run configuration.

Figures are produced separately (``python -m src.make_figures --shap``).

Run (machine with API access; ``TABPFN_TOKEN`` exported)::

    python -m src.shap_tabpfn --dry-run        # rows, cost estimate, no quota
    python -m src.shap_tabpfn                  # explain (≈ 27,000 predicted rows)
    python -m src.shap_tabpfn --smoke          # local mock model, no API (pipeline test)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .model_tabpfn import T0_CATEGORICAL, T0_FEATURES, t0_features
from .protocol import BLOCK_COL, SEED, TARGET, load_base

TRAIN_FILE = "data/itbi_sp_2025_level_a.csv"
TEST_FILE = "data/itbi_sp_2026_test.csv"
MODEL_RECORD = "results/tabpfn_cache/oot_tabpfn_t0_level_a/blockall_seed42_model.json"
SUBSAMPLE_FILE = "data/shap_subsample.csv"
BACKGROUND_FILE = "data/shap_background.csv"
KEEP_COLS = ["sql", "tipo_imovel", BLOCK_COL, "bairro", "data_transacao", "mes_idx",
             "area_construida_m2", "area_terreno_m2", "idade", "padrao_nivel",
             "descricao_padrao", "dist_estacao_m", "lat", "lon", "valor_transacao",
             "area_ref", TARGET]


# --------------------------------------------------------------------------
# Samples
# --------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, n: int, seed: int,
                      weights: dict[str, float]) -> pd.DataFrame:
    """``n`` rows, per-type counts proportional to ``weights`` (at least 1 each)."""
    types = list(weights)
    tot = sum(weights.values())
    counts = {t: max(1, int(round(n * weights[t] / tot))) for t in types}
    while sum(counts.values()) > n:                      # rounding overshoot
        t = max(counts, key=counts.get); counts[t] -= 1
    parts = []
    for i, t in enumerate(types):
        sub = df[df["tipo_imovel"] == t]
        parts.append(sub.sample(n=min(counts[t], len(sub)), random_state=seed + i))
    out = pd.concat(parts).sort_values(["tipo_imovel", "sql"]).reset_index(drop=True)
    return out


def build_samples(n_explain: int, n_background: int, seed: int,
                  overwrite: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write (or reuse) the explained rows and the background sample."""
    train = load_base(TRAIN_FILE)
    weights = train["tipo_imovel"].value_counts(normalize=True).to_dict()
    sub_p, bg_p = Path(SUBSAMPLE_FILE), Path(BACKGROUND_FILE)
    if sub_p.exists() and bg_p.exists() and not overwrite:
        sub = pd.read_csv(sub_p, dtype={"sql": str})
        bg = pd.read_csv(bg_p, dtype={"sql": str})
        print(f"reusing {sub_p} ({len(sub)} rows) and {bg_p} ({len(bg)} rows)")
        return sub, bg
    test = load_base(TEST_FILE)
    sub = stratified_sample(test, n_explain, seed, weights)[KEEP_COLS]
    bg = stratified_sample(train, n_background, seed + 100, weights)[KEEP_COLS]
    sub.to_csv(sub_p, index=False)
    bg.to_csv(bg_p, index=False)
    print(f"wrote {sub_p}: {len(sub)} rows "
          f"({sub['tipo_imovel'].value_counts().to_dict()}) | "
          f"{bg_p}: {len(bg)} rows ({bg['tipo_imovel'].value_counts().to_dict()})")
    return sub, bg


# --------------------------------------------------------------------------
# Model wrapper: numpy matrix from shap -> feature table -> cached predict
# --------------------------------------------------------------------------
class CachedPredictor:
    """Callable ``f(X: np.ndarray) -> np.ndarray`` for shap, cached per call."""

    def __init__(self, model, categories: dict[str, list], cache_dir: Path,
                 chunk: int = 5000):
        self.model, self.categories, self.chunk = model, categories, chunk
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.n_calls = self.n_rows = self.n_cached = 0

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        """Feature table -> float matrix for shap (categories as integer codes)."""
        M = X.copy()
        for c in T0_CATEGORICAL:
            M[c] = M[c].astype(str).map({v: i for i, v in enumerate(self.categories[c])})
        return M[T0_FEATURES].to_numpy(dtype=float)

    def frame(self, X: np.ndarray) -> pd.DataFrame:
        """Float matrix from shap -> feature table with the fitted dtypes."""
        df = pd.DataFrame(np.asarray(X, dtype=float), columns=T0_FEATURES)
        for c in T0_CATEGORICAL:
            cats = self.categories[c]
            df[c] = pd.Categorical([cats[int(round(v))] for v in df[c]], categories=cats)
        return df

    def __call__(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        key = hashlib.sha1(np.ascontiguousarray(X).tobytes()).hexdigest()
        p = self.cache_dir / f"{key}.npy"
        self.n_calls += 1; self.n_rows += len(X)
        if p.exists():
            self.n_cached += 1
            return np.load(p)
        df = self.frame(X)
        out = [np.asarray(self.model.predict(df.iloc[i:i + self.chunk]), dtype=float)
               for i in range(0, len(df), self.chunk)]
        pred = np.concatenate(out)
        np.save(p, pred)
        return pred


class MockModel:
    """Local stand-in for ``--smoke``: a fixed hedonic rule, no API."""

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        t = X["tipo_imovel"].astype(str).map({"apartment": 8.9, "house": 8.3,
                                              "commercial": 8.6}).fillna(8.6)
        return (t.to_numpy() - 0.004 * X["idade"].to_numpy() + 0.12 * X["padrao_nivel"].to_numpy()
                - 0.00005 * X["dist_estacao_m"].to_numpy() + 0.6 * (X["lon"].to_numpy() + 46.65)
                - 0.4 * (X["lat"].to_numpy() + 23.58))


# --------------------------------------------------------------------------
# Explanation
# --------------------------------------------------------------------------
def explain(sub: pd.DataFrame, bg: pd.DataFrame, predictor: CachedPredictor,
            n_permutations: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    import shap
    X_bg = predictor.encode(t0_features(bg))
    X_ex = predictor.encode(t0_features(sub))
    masker = shap.maskers.Independent(X_bg, max_samples=len(X_bg))
    explainer = shap.PermutationExplainer(predictor, masker, feature_names=T0_FEATURES,
                                          seed=seed)
    n_masks = 2 * len(T0_FEATURES) + 1
    exp = explainer(X_ex, max_evals=n_permutations * n_masks,
                    batch_size=n_permutations * n_masks, silent=True)
    return np.asarray(exp.values, dtype=float), np.asarray(exp.base_values, dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser(description="Permutation SHAP for the TabPFN-3.5 valuation model")
    ap.add_argument("--n-explain", type=int, default=50, help="2026 rows explained")
    ap.add_argument("--n-background", type=int, default=20, help="training rows in the background")
    ap.add_argument("--n-permutations", type=int, default=1)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--model-record", default=MODEL_RECORD)
    ap.add_argument("--label", default="tabpfn_t0_level_a")
    ap.add_argument("--resample", action="store_true", help="redraw the two samples")
    ap.add_argument("--dry-run", action="store_true", help="samples + cost estimate, no explanation")
    ap.add_argument("--smoke", action="store_true", help="local mock model; tests the pipeline only")
    ap.add_argument("--out", default="results/shap")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub, bg = build_samples(args.n_explain, args.n_background, args.seed, args.resample)
    n_masks = 2 * len(T0_FEATURES) + 1
    n_probe = len(T0_FEATURES) + 1          # shap's varying-inputs probe, once per row
    n_rows = len(sub) * (args.n_permutations * n_masks + n_probe) * len(bg)
    print(f"{len(sub)} rows explained x ({args.n_permutations} permutation(s) x {n_masks} masks "
          f"+ {n_probe} probe masks) x {len(bg)} background rows = {n_rows:,} predicted rows "
          f"(upper bound; identical rows are deduplicated) in {2 * len(sub)} API calls")

    label = args.label + ("_smoke" if args.smoke else "")
    if args.smoke:
        model = MockModel()
    else:
        if args.dry_run:
            if os.environ.get("TABPFN_TOKEN"):
                from tabpfn_client import estimate_cost, get_api_usage
                meta = json.loads(Path(args.model_record).with_name(
                    Path(args.model_record).name.replace("model.json", "fit_meta.json")).read_text())
                r = estimate_cost(np.zeros((meta["n_train"], len(T0_FEATURES))),
                                  np.zeros((n_rows, len(T0_FEATURES))),
                                  model_version="v3.5", operation="predict")
                print(f"cost estimate, {meta['n_train']:,} train x {n_rows:,} predicted rows: "
                      f"{r.estimated_cost} ({r.pricing_version})")
                print(get_api_usage())
            else:
                print("TABPFN_TOKEN not set: skipping cost estimate")
            return
        if not os.environ.get("TABPFN_TOKEN"):
            sys.exit("TABPFN_TOKEN is not set (export TABPFN_TOKEN=...)")
        from tabpfn_client import TabPFNRegressor
        model = TabPFNRegressor.load_model(args.model_record)
        print(f"model reloaded from {args.model_record} (no refit)")

    train_cats = {c: sorted(load_base(TRAIN_FILE)[c].astype(str).unique().tolist())
                  for c in T0_CATEGORICAL}
    predictor = CachedPredictor(model, train_cats, out_dir / "cache" / label)
    t0 = time.time()
    values, base = explain(sub, bg, predictor, args.n_permutations, args.seed)
    yhat = base + values.sum(axis=1)
    print(f"explained in {time.time() - t0:,.0f}s | {predictor.n_calls} model calls "
          f"({predictor.n_cached} from cache), {predictor.n_rows:,} rows")

    res = sub[["sql", "tipo_imovel", BLOCK_COL, "bairro", "data_transacao",
               "valor_transacao", "area_ref", TARGET]].copy()
    res["yhat_ln"] = yhat
    res["base_value_ln"] = base
    for j, f in enumerate(T0_FEATURES):
        res[f"shap_{f}"] = values[:, j]
    res.to_csv(out_dir / f"shap_values_{label}.csv", index=False)

    abs_mean = {f: float(np.abs(values[:, j]).mean()) for j, f in enumerate(T0_FEATURES)}
    by_type = {t: {f: float(np.abs(values[(sub["tipo_imovel"] == t).to_numpy(), j]).mean())
                   for j, f in enumerate(T0_FEATURES)}
               for t in sorted(sub["tipo_imovel"].unique())}
    summary = {"label": label, "model_record": None if args.smoke else args.model_record,
               "n_explained": int(len(sub)), "n_background": int(len(bg)),
               "n_permutations": args.n_permutations, "seed": args.seed,
               "features": T0_FEATURES, "target": TARGET,
               "explainer": "shap.PermutationExplainer, Independent masker",
               "predicted_rows": int(predictor.n_rows),
               "mean_abs_shap_ln": dict(sorted(abs_mean.items(), key=lambda kv: -kv[1])),
               "mean_abs_shap_ln_by_type": by_type,
               "rmse_ln_on_subsample": float(np.sqrt(np.mean((yhat - sub[TARGET].to_numpy()) ** 2)))}
    (out_dir / f"shap_summary_{label}.json").write_text(json.dumps(summary, indent=2),
                                                        encoding="utf-8")
    print("mean |SHAP| (ln units), descending:")
    for f, v in summary["mean_abs_shap_ln"].items():
        print(f"  {f:<20s} {v:.4f}  (x{np.exp(v):.3f} typical effect on the unit value)")
    print(f"RMSE_ln on the explained rows: {summary['rmse_ln_on_subsample']:.4f}")


if __name__ == "__main__":
    main()
