#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
model_tabpfn.py — TabPFN-3.5 *without* explicit spatial modelling
==================================================================
The central hypothesis of this project, stated so that it can fail:

    A tabular foundation model that receives only the property attributes
    and the coordinates as two ordinary numeric columns — no spatial weights
    matrix, no spatial lag, no rotated axes, no neighbourhood features —
    predicts spatially correlated prices as well as, or better than,
    specialised models that encode spatial dependence explicitly
    (SAR lag with W and rho; XGBoost with a k-NN-8 lag, rotated coordinates
    and nested tuning).

The design is deliberately asymmetric *against* the hypothesis: the
baselines keep every spatial advantage; TabPFN-3.5 gets the plain table.
Refutation criteria, per the shared protocol (``src/protocol.py``):
worse pooled RMSE_ln / MAPE, losses in most spatial blocks (paired
Wilcoxon, block bootstrap), and — the sharpest one — more spatial
autocorrelation left in the out-of-fold residuals (Moran's I).

Feature set ("T0", the reference study's replica)
--------------------------------------------------
Raw columns, no transformations (TabPFN handles scale and non-linearity):
built area, lot area, age, finish grade, distance to the nearest station,
month index, property type (categorical), latitude, longitude. Exactly the
information set of the SAR/OLS design matrix, minus the spatial machinery.

Runs
----
* ``--thinking off``      zero-shot TabPFN-3.5 (the model as shipped)
* ``--thinking medium|high``  thinking mode: the fit explores configurations
  by internal validation on the training fold, optimising RMSE
  (``thinking_metric="rmse"``). Main runs use **no** ``group_col`` (the model
  sees nothing about the blocks); ``--group-col`` adds the spatial block as
  the grouping column of the internal validation, a robustness variant only
  (it is spatial information entering through the tuning, hence not the
  main analysis).

Ablation: nominal location (``--nominal bairro cep``)
-----------------------------------------------------
Labelled apart from the main analysis, because it changes the information
set. The T0 table describes location with two numbers; the ITBI form also
carries it as *names*: the district field (``bairro``) and the postal code
(``cep``). Both are high-cardinality strings — about 3,900 and 10,000
distinct values in the 24,000 rows of Level A — the kind of column
TabPFN-3.5 is advertised to take as-is. The ablation adds them to T0,
unchanged otherwise, and asks how much nominal location adds to raw
coordinates.

The columns go in *as filed*: ``bairro`` is a free-text field (39 % empty in
Level A, spelling variants, and tower/block labels typed into it in roughly
a quarter of the filled rows); it is not cleaned, and missing stays missing.
``cep`` is written in its usual form (``01310-100``) so that it cannot be
read as a number. Both are sent as plain strings — not as pandas categories,
whose integer codes would differ between a training fold and its test fold.

What to expect, stated before the runs:

* leg 1 (leave-one-block-out): a test block's postal codes are absent from
  the training folds by construction (97–100 % unseen; 25–51 % of district
  names), so the nominal columns cannot carry much there. A gain in this leg
  would need an explanation; it is a negative control.
* leg 2 (2025 -> 2026): 78 % of the 2026 rows have a postal code present in
  Level A (90 % in the full base). If names add anything to coordinates, it
  shows here, and it should concentrate in the rows whose code was seen in
  training (``src/nominal_ablation.py`` splits the error that way).

The baselines are not re-run with these columns: this is an ablation of the
TabPFN input, not a new comparison.

Second ablation: a name in place of the coordinates (``--no-coords``)
---------------------------------------------------------------------
The first ablation found that names add nothing *to* coordinates. This one
asks the other question: how good is each location descriptor *on its own*?
Latitude and longitude are dropped, and the table carries, in turn: nothing
(the floor: what the attributes and the distance to a station say without
any location), the postal code as a string, the district field, both, and
the postal code read as a number (``cep_num``; Brazilian postal codes are
assigned geographically, so the number is an ordinal, one-dimensional
coordinate that has a value for codes never seen in training). The distance
to the nearest station stays in these variants: it is an attribute of T0 and
a weak spatial signal in its own right.

The strict version (``--no-coords --no-station``) removes that too, because
the distance is computed from the coordinates. The postal code is then the
only spatial information TabPFN-3.5 has, while SAR keeps its weights matrix
and XGBoost its k-NN lag, rotated coordinates and station distance. Three
inputs: nothing (its own floor), the code as a string, the code as a number.

Expectations, written before the runs (from a location-only check on the
2026 rows: the training mean by postal code reaches RMSE 0.397 / 0.358 in ln
with 24k / 82k training rows, the mean of the 8 nearest training rows by
coordinates 0.365 / 0.343, the mean by district 0.426 / 0.410):

* leg 1: a held-out block's postal codes are new, so the code as a string
  should fall to the floor; the district field likewise for most rows. The
  numeric code is the only name-based input with a chance there.
* leg 2: every name-based input lands between the floor and the plain table,
  the postal code ahead of the district field, and closer to the plain table
  on 82k rows than on 24k.
* strict version: the floor rises (no spatial column left), and the postal
  code recovers less than half of the way to the plain table; we expect it
  to lose to XGBoost+lag in both legs and make no call against SAR.
* what would change the reading of the main result: a name-based input that
  matches the plain table in leg 2. The claim "two raw coordinate columns
  carry the spatial signal" would then have to become "any location
  identifier does".

Text variant: the unit and the building as written (``--text``)
-----------------------------------------------------------------
TabPFN-3.5 reads free-text columns natively. The ITBI form has two that no
run above uses: the unit complement (``complemento``: "AP 201 E 3VGS",
"CJ 1904 TORRE B", "CASA 3") and the *Referência* field (the building or
development name on about 40 % of the forms: "EDIFICIO THE PARK",
"CJ HAB SAFIRA IV", "LIVING HEREDITA"; ``pipeline/05_reference_field.py``).
``--text complemento referencia`` adds both, as filed, to the plain table
with its coordinates: the question is whether the words carry value
information the table lacks (parking spaces, towers, penthouses, the
development and its market segment, social-housing complexes). Strings are
sent as they are; the server decides between category and text.

Expectations, written before the runs: a small gain at most. Within the
same building, the floor read from the unit number moves the unit price by
about 0.01 % per floor (51,495 apartments in 9,801 buildings of the full 2025
base), and units whose complement mentions parking differ from their
neighbours in the same building by −0.026 in ln; the building name is the
open question. A gain would show in leg 2 (the same buildings sell again);
leg 1 is again a near-negative control.

Predictive quantiles (for the 80 % intervals used later) are requested only
from zero-shot fits: the server returns point predictions only for models
fitted in thinking mode.

Every API response is cached under ``results/tabpfn_cache/<label>/`` —
fitted-model records (``save_model``) and predictions per fold — so a
configuration is paid for once and the protocol can be re-run offline.
``--dry-run`` builds the features and prints the server's cost estimate
(``estimate_cost``: only row/column counts are sent, no quota used).

The API is not reachable from the development sandbox; these runs are
executed on the author's machine (``scripts/10_tabpfn_level_a.sh``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .protocol import BLOCK_COL, SEED, load_base, run_cv, compare_models

TARGET = "ln_vu"
MODEL_VERSION = "v3.5"

T0_NUMERIC = ["area_construida_m2", "area_terreno_m2", "idade", "padrao_nivel",
              "dist_estacao_m", "mes_idx", "lat", "lon"]
T0_CATEGORICAL = ["tipo_imovel"]
T0_FEATURES = T0_NUMERIC + T0_CATEGORICAL
NOMINAL_COLUMNS = ("bairro", "cep", "cep_num")   # ablations only, never in the main runs
TEXT_COLUMNS = ("complemento", "referencia")      # text variant only, never in the main runs
REFERENCE_FILE = Path("data/itbi_sp_reference_field.csv.gz")
_REF_CACHE: dict = {}
COORDINATES = ["lat", "lon"]
STATION = "dist_estacao_m"              # computed from the coordinates


def nominal_column(df: pd.DataFrame, col: str) -> pd.Series:
    """One nominal-location column as plain strings (missing stays missing).

    ``cep_num`` is the exception: the postal code as a number.
    """
    if col == "cep_num":
        return pd.to_numeric(df["cep"].astype(str).str.replace(r"\D", "", regex=True),
                             errors="coerce").astype(float)
    s = df[col]
    out = pd.Series([None] * len(s), index=s.index, dtype=object)
    ok = s.notna().to_numpy()
    if col == "cep":
        digits = (s[ok].astype(str).str.replace(r"\D", "", regex=True)
                  .str.replace(r"^(\d{1,7})$", lambda m: m.group(1).zfill(8), regex=True))
        out[ok] = (digits.str[:5] + "-" + digits.str[5:]).to_numpy()
    else:
        txt = s[ok].astype(str).str.strip()
        out[ok] = txt.where(txt != "", None).to_numpy()
    return out


def _reference_lookup() -> pd.Series:
    """*Referência* by (sql, date, price), from pipeline/05_reference_field.py."""
    if "s" not in _REF_CACHE:
        ref = pd.read_csv(REFERENCE_FILE, dtype={"sql": str, "referencia": str})
        ref["key"] = (ref["sql"] + "|" + ref["data_transacao"] + "|"
                      + ref["valor_transacao"].map(repr))
        _REF_CACHE["s"] = ref.drop_duplicates("key").set_index("key")["referencia"]
    return _REF_CACHE["s"]


def text_column(df: pd.DataFrame, col: str) -> pd.Series:
    """A free-text column as filed (whitespace collapsed; empty -> missing)."""
    if col == "referencia" and "referencia" not in df.columns:
        key = (df["sql"].astype(str) + "|"
               + pd.to_datetime(df["data_transacao"]).dt.strftime("%Y-%m-%d") + "|"
               + df["valor_transacao"].astype(float).map(repr))
        s = pd.Series(_reference_lookup().reindex(key).to_numpy(), index=df.index)
    else:
        s = df[col]
    out = pd.Series([None] * len(s), index=s.index, dtype=object)
    ok = s.notna().to_numpy()
    txt = s[ok].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    out[ok] = txt.where(txt != "", None).to_numpy()
    return out


def feature_names(nominal: tuple[str, ...] = (), coords: bool = True,
                  station: bool = True, text: tuple[str, ...] = ()) -> list[str]:
    """Column order of the table sent to the model."""
    base = [c for c in T0_FEATURES
            if (coords or c not in COORDINATES) and (station or c != STATION)]
    return base + list(nominal) + list(text)


def t0_features(df: pd.DataFrame, group_col: str | None = None,
                nominal: tuple[str, ...] = (), coords: bool = True,
                station: bool = True, text: tuple[str, ...] = ()) -> pd.DataFrame:
    """Plain feature table for TabPFN (raw columns, categorical as category).

    ``nominal`` appends the ablation columns (``bairro``, ``cep`` as strings,
    ``cep_num`` as a number); ``coords=False`` drops latitude and longitude,
    ``station=False`` the distance to the nearest station.
    """
    X = pd.DataFrame(index=df.index)
    for c in T0_NUMERIC:
        if (not coords and c in COORDINATES) or (not station and c == STATION):
            continue
        X[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    X["area_terreno_m2"] = X["area_terreno_m2"].fillna(0.0)
    for c in T0_CATEGORICAL:
        X[c] = df[c].astype("category")
    for c in nominal:
        if c not in NOMINAL_COLUMNS:
            raise ValueError(f"unknown nominal column {c!r}; choose from {NOMINAL_COLUMNS}")
        X[c] = nominal_column(df, c)
    for c in text:
        if c not in TEXT_COLUMNS:
            raise ValueError(f"unknown text column {c!r}; choose from {TEXT_COLUMNS}")
        X[c] = text_column(df, c)
    if group_col:
        X[group_col] = df[group_col].astype(int)
    return X.reset_index(drop=True)


def nominal_tag(nominal: tuple[str, ...], as_categorical: bool = False,
                coords: bool = True, station: bool = True,
                text: tuple[str, ...] = ()) -> str:
    """Label fragment for an ablation run, e.g. ``_nom_bairro_cep``,
    ``_nocoord_nom_cep`` or ``_nocoord_nostation_nom_cep_num``."""
    tag = ("" if coords else "_nocoord") + ("" if station else "_nostation")
    if nominal:
        tag += "_nom_" + "_".join(nominal) + ("_cat" if as_categorical else "")
    if text:
        tag += "_txt_" + "_".join(text)
    return tag


# --------------------------------------------------------------------------
# Model with per-fold cache
# --------------------------------------------------------------------------
class TabPFNT0Model:
    """fit/predict for the protocol; every server call cached on disk."""

    def __init__(self, cache_dir: Path, block: str, seed: int = SEED,
                 thinking: str = "off", group_col: str | None = None,
                 thinking_timeout_s: float = 2400.0,
                 quantiles: list[float] | None = None, chunk: int = 5000,
                 ignore_pretraining_limits: bool = False,
                 nominal: tuple[str, ...] = (), nominal_categorical: bool = False,
                 coords: bool = True, station: bool = True,
                 text: tuple[str, ...] = ()):
        self.cache_dir, self.block, self.seed = cache_dir, block, seed
        self.text = tuple(text)
        self.nominal, self.nominal_categorical = tuple(nominal), nominal_categorical
        self.coords, self.station = coords, station
        self.thinking, self.group_col = thinking, group_col
        self.timeout, self.quantiles, self.chunk = thinking_timeout_s, quantiles, chunk
        self.ignore_limits = ignore_pretraining_limits
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # cache paths -------------------------------------------------------
    def _p(self, kind: str) -> Path:
        return self.cache_dir / f"block{self.block}_seed{self.seed}_{kind}"

    def _make(self):
        from tabpfn_client import TabPFNRegressor
        kw = dict(random_state=self.seed,
                  ignore_pretraining_limits=self.ignore_limits)
        if self.thinking != "off":
            kw.update(thinking_mode=True, thinking_effort=self.thinking,
                      thinking_timeout_s=self.timeout, thinking_metric="rmse")
            if self.group_col:
                kw["group_col"] = self.group_col
        if self.nominal and self.nominal_categorical:
            # variant: declare the string columns categorical instead of
            # leaving the server to decide between category and text
            cols = feature_names(self.nominal, self.coords, self.station)
            kw["categorical_features_indices"] = [
                cols.index(c) for c in T0_CATEGORICAL + list(self.nominal) if c != "cep_num"]
        return TabPFNRegressor.create_default_for_version(MODEL_VERSION, **kw)

    def fit(self, train_df: pd.DataFrame):
        self.n_train_ = len(train_df)
        if self._p("pred_mean.npy").exists():
            self.model_ = None                      # nothing to fit: cached
            return self
        rec = self._p("model.json")
        from tabpfn_client import TabPFNRegressor
        if rec.exists():
            self.model_ = TabPFNRegressor.load_model(rec)
            return self
        X = t0_features(train_df, self.group_col, self.nominal, self.coords, self.station,
                        self.text)
        y = train_df[TARGET].to_numpy(dtype=float)
        t0 = time.time()
        self.model_ = self._make().fit(X, y)
        self.model_.save_model(rec)
        meta = {"block": self.block, "seed": self.seed, "n_train": int(len(X)),
                "thinking": self.thinking, "group_col": self.group_col,
                "nominal": list(self.nominal), "coordinates": self.coords,
                "station_distance": self.station,
                "nominal_categorical": self.nominal_categorical,
                "text": list(self.text),
                "fit_s": round(time.time() - t0, 1)}
        self._p("fit_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"  block {self.block}: fit {meta['fit_s']:.0f}s "
              f"(thinking={self.thinking})", flush=True)
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        p = self._p("pred_mean.npy")
        if p.exists():
            return np.load(p)
        X = t0_features(test_df, self.group_col, self.nominal, self.coords, self.station,
                        self.text)
        out = []
        for i in range(0, len(X), self.chunk):
            out.append(np.asarray(self.model_.predict(X.iloc[i:i + self.chunk]),
                                  dtype=float))
        pred = np.concatenate(out)
        np.save(p, pred)
        # the server returns point predictions only for thinking-fitted models
        # (HTTP 422 on output_type="quantiles"); quantiles come from the
        # zero-shot fit of the same fold
        if self.quantiles and self.thinking == "off":
            qs = []
            for i in range(0, len(X), self.chunk):
                q = self.model_.predict(X.iloc[i:i + self.chunk],
                                        output_type="quantiles",
                                        quantiles=self.quantiles)
                qs.append(np.column_stack([np.asarray(a, dtype=float) for a in q]))
            np.save(self._p("pred_quantiles.npy"), np.vstack(qs))
            self._p("quantiles.json").write_text(json.dumps(self.quantiles))
        return pred


class TabPFNFactory:
    """``factory(seed)`` for ``run_cv``; folds arrive in sorted block order."""

    def __init__(self, cache_dir: Path, folds_order: list[int], **kw):
        self.cache_dir, self.folds_order, self.kw = cache_dir, folds_order, kw
        self.call = 0

    def __call__(self, seed: int):
        block = str(self.folds_order[self.call % len(self.folds_order)])
        self.call += 1
        return TabPFNT0Model(self.cache_dir, block, seed=seed, **self.kw)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="TabPFN-3.5 (no explicit spatial modelling) through the protocol")
    ap.add_argument("--data", default="data/itbi_sp_2025_level_a.csv")
    ap.add_argument("--thinking", choices=["off", "medium", "high"], default="off")
    ap.add_argument("--group-col", action="store_true",
                    help="robustness variant: spatial block as thinking group_col")
    ap.add_argument("--thinking-timeout-s", type=float, default=2400.0)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    ap.add_argument("--quantiles", type=float, nargs="*", default=None,
                    help="also cache predictive quantiles, e.g. 0.1 0.25 0.5 0.75 0.9")
    ap.add_argument("--chunk", type=int, default=5000, help="test rows per predict call")
    ap.add_argument("--label", default=None)
    ap.add_argument("--financed-only", action="store_true")
    ap.add_argument("--nominal", nargs="*", default=[], choices=list(NOMINAL_COLUMNS),
                    help="ABLATION: add nominal-location columns as raw strings")
    ap.add_argument("--nominal-categorical", action="store_true",
                    help="ablation variant: declare the nominal columns categorical")
    ap.add_argument("--no-coords", action="store_true",
                    help="ABLATION: drop latitude and longitude (a name in place of the coordinates)")
    ap.add_argument("--no-station", action="store_true",
                    help="ABLATION: also drop the distance to the nearest station (strict version)")
    ap.add_argument("--text", nargs="*", default=[], choices=list(TEXT_COLUMNS),
                    help="TEXT VARIANT: add the unit complement and/or the Referência field as free text")
    ap.add_argument("--ignore-limits", action="store_true",
                    help="ignore_pretraining_limits (if the server objects to the cardinality)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print features and the server's cost estimate; no fit")
    ap.add_argument("--compare-with", nargs="*", default=[],
                    help="labels of finished CV runs to compare against (Wilcoxon + block bootstrap)")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    df = load_base(args.data, financed_only=args.financed_only)
    group_col = BLOCK_COL if args.group_col else None
    nominal = tuple(c for c in NOMINAL_COLUMNS if c in args.nominal)   # fixed order
    coords, station = not args.no_coords, not args.no_station
    text = tuple(c for c in TEXT_COLUMNS if c in args.text)            # fixed order
    features = feature_names(nominal, coords, station, text)
    variant = bool(nominal or text or not coords or not station)
    label = args.label or ("tabpfn_t0"
                           + ("" if args.thinking == "off" else f"_think_{args.thinking}")
                           + ("_grp" if group_col else "")
                           + nominal_tag(nominal, args.nominal_categorical, coords, station, text) + "_"
                           + Path(args.data).stem.replace("itbi_sp_", "")
                           + ("_fin" if args.financed_only else ""))
    folds_order = sorted(df[BLOCK_COL].unique().tolist())
    n_test_max = int(df[BLOCK_COL].value_counts().max())
    print(f"TabPFN-{MODEL_VERSION} T0 | thinking={args.thinking} group_col={group_col} "
          f"| {len(df):,} rows, {len(folds_order)} blocks -> label '{label}'", flush=True)
    print(f"features ({len(features)}): {features}"
          + ("  [ABLATION / VARIANT]" if variant else ""))

    if args.dry_run:
        X = t0_features(df.iloc[:100], nominal=nominal, coords=coords, station=station,
                        text=text)
        print(X.dtypes.to_string())
        if not os.environ.get("TABPFN_TOKEN"):
            print("TABPFN_TOKEN not set: skipping cost estimate"); return
        from tabpfn_client import estimate_cost, get_api_usage
        Xtr = np.zeros((len(df) - n_test_max, len(features)))
        Xte = np.zeros((n_test_max, len(features)))
        r = estimate_cost(Xtr, Xte, model_version=MODEL_VERSION, operation="predict")
        print(f"predict, largest fold ({Xtr.shape[0]:,} x {Xte.shape[0]:,}): "
              f"{r.estimated_cost} ({r.pricing_version}) -> x{len(folds_order)} folds x{len(args.seeds)} seeds")
        if args.thinking != "off":
            r = estimate_cost(Xtr, None, model_version=MODEL_VERSION,
                              operation="thinking_fit", thinking_effort=args.thinking)
            print(f"thinking_fit ({args.thinking}): {r.estimated_cost} per fold "
                  f"-> x{len(folds_order)} folds x{len(args.seeds)} seeds")
        print(get_api_usage())
        return

    if not os.environ.get("TABPFN_TOKEN"):
        sys.exit("TABPFN_TOKEN is not set (export TABPFN_TOKEN=... ); "
                 "see scripts/00_check_api.py")

    out_dir = Path(args.out)
    cache_dir = out_dir / "tabpfn_cache" / label
    factory = TabPFNFactory(cache_dir, folds_order, thinking=args.thinking,
                            group_col=group_col,
                            thinking_timeout_s=args.thinking_timeout_s,
                            quantiles=args.quantiles, chunk=args.chunk,
                            ignore_pretraining_limits=args.ignore_limits,
                            nominal=nominal, nominal_categorical=args.nominal_categorical,
                            coords=coords, station=station, text=text)
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

    meta = json.loads((out_dir / f"cv_{label}.json").read_text())
    meta["model"] = {"family": "TabPFN", "version": MODEL_VERSION,
                     "thinking": args.thinking, "group_col": group_col,
                     "thinking_metric": "rmse" if args.thinking != "off" else None,
                     "features": features, "explicit_spatial_modelling": False,
                     "nominal_location": list(nominal), "coordinates": coords,
                     "station_distance": station,
                     "nominal_as": (None if not nominal else
                                    "categorical" if args.nominal_categorical else "string"),
                     "text_columns": list(text),
                     "analysis": "ablation" if variant else "main"}
    (out_dir / f"cv_{label}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for other in args.compare_with:
        for metric in ("rmse_ln", "mape_pct"):
            r = compare_models(label, other, df, seed=args.seeds[0], metric=metric,
                               out_dir=out_dir)
            w, bb = r["wilcoxon"], r["block_bootstrap"]
            print(f"{label} vs {other} [{metric}]: per-fold mean diff {w['mean_diff']:+.4f} "
                  f"(Wilcoxon p={w['p_value']:.3f}) | pooled {bb['diff']:+.4f} "
                  f"CI95 [{bb['ci95'][0]:+.4f}, {bb['ci95'][1]:+.4f}]")


if __name__ == "__main__":
    main()
