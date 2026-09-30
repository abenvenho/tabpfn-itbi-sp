#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
nominal_ablation.py — how much does nominal location add to raw coordinates?
============================================================================
An ablation of the TabPFN-3.5 input, kept apart from the main analysis. The
main runs describe location with latitude and longitude only. Here the same
table also carries the district field (``bairro``) and the postal code
(``cep``) of the ITBI form, as raw high-cardinality strings
(``src/model_tabpfn.py``, ``--nominal``). Nothing else changes: same rows,
same folds, same seed, zero-shot.

Two commands:

``probe``  (API, a few hundred rows)
    Fits TabPFN-3.5 on 600 Level A rows with and without the string columns
    and predicts 100 rows of a held-out spatial block, where every postal
    code is new and part of the district field is empty. It answers, before
    the real runs, whether the server takes the columns, copes with unseen
    and missing values, and accepts ``categorical_features_indices``.

``summary``  (offline)
    Reads whatever ablation runs are in ``results/`` and writes
    ``results/nominal_ablation_summary.{json,md}``: each variant against the
    plain table in both legs (pooled metrics, Moran's I of the residuals,
    paired Wilcoxon over blocks, block-bootstrap CI of the difference), and,
    for the out-of-time leg, the mechanism check — the error split by
    whether the row's postal code / district name occurs in the training
    base. If names carry information beyond coordinates, the gain has to sit
    in the "seen" rows; a gain of the same size in the "unseen" rows would
    point at something else.

``descriptors``  (offline)
    The second ablation (``--no-coords``): a name *in place of* the
    coordinates. Writes ``results/location_descriptor_summary.{json,md}``:
    for each leg, the three baselines, TabPFN-3.5 on the plain table, and
    TabPFN-3.5 with no location at all (the floor), with the postal code, the
    district field, both, and the postal code as a number — each against the
    plain table, XGBoost+lag and SAR (block-bootstrap CI), plus the share of
    the location signal it recovers (0 % = the floor, 100 % = the two
    coordinates) and, for the out-of-time leg, the split by whether the
    row's postal code / district occurs in the training base. The strict
    family (``--no-station`` as well: the postal code as the only spatial
    information) is reported the same way, against its own floor.

``usage``  (API)
    Prints the account's credit usage.

Runs are produced by ``scripts/50_nominal_location_ablation.sh``.
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

from .model_tabpfn import (MODEL_VERSION, T0_CATEGORICAL, T0_FEATURES,
                           nominal_column, nominal_tag, t0_features)
from .out_of_time import TEST_FILE, TRAIN_FILES, compare_holdout
from .protocol import (BLOCK_COL, SEED, TARGET, block_bootstrap_diff,
                       compare_models, load_base, regression_metrics)

VARIANTS = [("bairro",), ("cep",), ("bairro", "cep")]                 # added to the coordinates
# in place of the coordinates: (nominal columns, station distance kept?, description)
DESCRIPTORS = [((), True, "no location (floor)"), (("cep",), True, "postal code"),
               (("bairro",), True, "district field"), (("bairro", "cep"), True, "district + postal code"),
               (("cep_num",), True, "postal code as a number"),
               ((), False, "nothing spatial (strict floor)"), (("cep",), False, "postal code only"),
               (("cep_num",), False, "postal code as a number only")]
FAMILY = {True: "no lat/lon", False: "no lat/lon, no station distance"}
BASELINES = [("ols", "OLS hedonic"), ("sar_gm", "SAR lag"), ("xgb_lag", "XGBoost + rotated coords + k-NN-8 lag")]
LEVEL_A = TRAIN_FILES["level_a"]


# --------------------------------------------------------------------------
# probe (API)
# --------------------------------------------------------------------------
def probe(n_train: int = 600, n_test: int = 100) -> int:
    if not os.environ.get("TABPFN_TOKEN"):
        sys.exit("TABPFN_TOKEN is not set (export TABPFN_TOKEN=...)")
    from tabpfn_client import TabPFNRegressor, get_api_usage

    df = load_base(LEVEL_A)
    rng = np.random.default_rng(SEED)
    held = int(sorted(df[BLOCK_COL].unique())[0])
    tr = df[df[BLOCK_COL] != held]
    te = df[df[BLOCK_COL] == held]
    tr = tr.iloc[rng.choice(len(tr), n_train, replace=False)]
    te = te.iloc[rng.choice(len(te), n_test, replace=False)]
    y_tr, y_te = tr[TARGET].to_numpy(float), te[TARGET].to_numpy(float)
    nom = ("bairro", "cep")
    unseen = {c: float((~nominal_column(te, c).isin(nominal_column(tr, c).dropna())).mean())
              for c in nom}
    print(f"probe: {n_train} train rows (blocks != {held}), {n_test} test rows (block {held})")
    print("  test rows with a value absent from training or missing: "
          + ", ".join(f"{c} {100 * v:.0f}%" for c, v in unseen.items()))
    print("  " + str(get_api_usage()))

    cols = T0_FEATURES + list(nom)
    cat_idx = [cols.index(c) for c in T0_CATEGORICAL + list(nom)]
    cases = [("plain table (T0)", (), {}),
             ("T0 + bairro + cep as strings", nom, {}),
             ("T0 + bairro + cep declared categorical", nom,
              {"categorical_features_indices": cat_idx})]
    failed = []
    for name, nominal, kw in cases:
        t0 = time.time()
        try:
            m = TabPFNRegressor.create_default_for_version(MODEL_VERSION, random_state=SEED, **kw)
            m.fit(t0_features(tr, nominal=nominal), y_tr)
            pred = np.asarray(m.predict(t0_features(te, nominal=nominal)), dtype=float)
            if len(pred) != n_test or not np.isfinite(pred).all():
                raise RuntimeError(f"bad predictions: len={len(pred)}, "
                                   f"finite={int(np.isfinite(pred).sum())}")
            rmse = regression_metrics(y_te, pred)["rmse_ln"]
            print(f"  OK   {name}: RMSE_ln={rmse:.3f} in {time.time() - t0:.0f}s")
        except Exception as e:                                   # noqa: BLE001
            failed.append(name)
            print(f"  FAIL {name}: {type(e).__name__}: {str(e)[:600]}")
    print("  " + str(get_api_usage()))
    if "T0 + bairro + cep as strings" in failed:
        print("The string columns were not accepted: do not start the ablation runs.")
        return 1
    if failed:
        print("The main ablation can run; the failed variant(s) above will be skipped or fail.")
    else:
        print("All good: the ablation runs can start.")
    return 0


# --------------------------------------------------------------------------
# summary (offline)
# --------------------------------------------------------------------------
def _fmt_ci(d: dict) -> str:
    return f"{d['diff']:+.4f} [{d['ci95'][0]:+.4f}; {d['ci95'][1]:+.4f}]"


def _leg1(out_dir: Path, seed: int) -> list[dict]:
    ref = "tabpfn_t0_2025_level_a"
    if not (out_dir / f"cv_{ref}.json").exists():
        return []
    df = load_base(LEVEL_A)
    ref_res = json.loads((out_dir / f"cv_{ref}.json").read_text())["per_seed"][str(seed)]
    rows = [{"leg": "1 spatial CV (Level A)", "variant": "plain table", "label": ref,
             **{k: ref_res["pooled"][k] for k in ("rmse_ln", "mape_pct", "r2_ln")},
             "moran_I": ref_res["moran_oof"]["I"]}]
    for v in VARIANTS:
        for cat in (False, True):
            label = f"tabpfn_t0{nominal_tag(v, cat)}_2025_level_a"
            if not (out_dir / f"cv_{label}.json").exists():
                continue
            res = json.loads((out_dir / f"cv_{label}.json").read_text())["per_seed"][str(seed)]
            c = compare_models(label, ref, df, seed=seed, metric="rmse_ln", out_dir=out_dir)
            cm = compare_models(label, ref, df, seed=seed, metric="mape_pct", out_dir=out_dir)
            pf_a = [f["rmse_ln"] for f in res["per_fold"]]
            pf_b = [f["rmse_ln"] for f in ref_res["per_fold"]]
            rows.append({"leg": "1 spatial CV (Level A)",
                         "variant": "+ " + " + ".join(v) + (" (categorical)" if cat else ""),
                         "label": label,
                         **{k: res["pooled"][k] for k in ("rmse_ln", "mape_pct", "r2_ln")},
                         "moran_I": res["moran_oof"]["I"],
                         "d_rmse": c["block_bootstrap"], "d_mape": cm["block_bootstrap"],
                         "wilcoxon_p": c["wilcoxon"]["p_value"],
                         "blocks_better": int(sum(a < b for a, b in zip(pf_a, pf_b))),
                         "n_blocks": len(pf_a)})
    return rows


def _seen_masks(train_df: pd.DataFrame, test_df: pd.DataFrame, col: str) -> dict:
    tr, te = nominal_column(train_df, col), nominal_column(test_df, col)
    missing = te.isna().to_numpy()
    seen = te.isin(tr.dropna()).to_numpy() & ~missing
    out = {f"{col} seen in training": seen, f"{col} not seen": ~seen & ~missing}
    if missing.any():
        out[f"{col} empty"] = missing
    return out


def _leg2(out_dir: Path, seed: int) -> tuple[list[dict], list[dict]]:
    rows, mech = [], []
    test_df = load_base(TEST_FILE)
    y = test_df[TARGET].to_numpy(float)
    blocks = test_df[BLOCK_COL].to_numpy()
    for train in TRAIN_FILES:
        ref = f"tabpfn_t0_{train}"
        if not (out_dir / f"oot_{ref}.json").exists():
            continue
        leg = f"2 out-of-time ({'Level A 24k' if train == 'level_a' else 'full 82k'} -> 2026)"
        ref_res = json.loads((out_dir / f"oot_{ref}.json").read_text())["per_seed"][str(seed)]
        yhat_ref = pd.read_csv(out_dir / f"oot_pred_{ref}_seed{seed}.csv")["yhat_ln"].to_numpy(float)
        rows.append({"leg": leg, "variant": "plain table", "label": ref,
                     **{k: ref_res["pooled"][k] for k in ("rmse_ln", "mape_pct", "r2_ln")},
                     "moran_I": ref_res["moran_test"]["I"]})
        train_df = None
        for v in VARIANTS:
            for cat in (False, True):
                label = f"tabpfn_t0{nominal_tag(v, cat)}_{train}"
                if not (out_dir / f"oot_{label}.json").exists():
                    continue
                res = json.loads((out_dir / f"oot_{label}.json").read_text())["per_seed"][str(seed)]
                c = compare_holdout(label, ref, test_df, seed=seed, metric="rmse_ln", out_dir=out_dir)
                cm = compare_holdout(label, ref, test_df, seed=seed, metric="mape_pct", out_dir=out_dir)
                pb_a = [b["rmse_ln"] for b in res["per_block"]]
                pb_b = [b["rmse_ln"] for b in ref_res["per_block"]]
                row = {"leg": leg,
                       "variant": "+ " + " + ".join(v) + (" (categorical)" if cat else ""),
                       "label": label,
                       **{k: res["pooled"][k] for k in ("rmse_ln", "mape_pct", "r2_ln")},
                       "moran_I": res["moran_test"]["I"],
                       "d_rmse": c["block_bootstrap"], "d_mape": cm["block_bootstrap"],
                       "wilcoxon_p": c["wilcoxon_blocks"]["p_value"],
                       "blocks_better": int(sum(a < b for a, b in zip(pb_a, pb_b))),
                       "n_blocks": len(pb_a)}
                xgb = f"xgb_lag_{train}"
                if (out_dir / f"oot_{xgb}.json").exists():
                    row["d_rmse_vs_xgb_lag"] = compare_holdout(
                        label, xgb, test_df, seed=seed, metric="rmse_ln",
                        out_dir=out_dir)["block_bootstrap"]
                rows.append(row)

                # mechanism: does the gain sit where the name was seen in training?
                if train_df is None:
                    train_df = load_base(TRAIN_FILES[train])
                yhat = pd.read_csv(out_dir / f"oot_pred_{label}_seed{seed}.csv")["yhat_ln"].to_numpy(float)
                for col in v:
                    for name, mask in _seen_masks(train_df, test_df, col).items():
                        if mask.sum() < 200:
                            continue
                        bb = block_bootstrap_diff(y[mask], yhat[mask], yhat_ref[mask],
                                                  blocks[mask], "rmse_ln")
                        mech.append({"leg": leg, "variant": row["variant"], "subset": name,
                                     "n": int(mask.sum()), "share": float(mask.mean()),
                                     "rmse_plain": regression_metrics(y[mask], yhat_ref[mask])["rmse_ln"],
                                     "rmse_variant": regression_metrics(y[mask], yhat[mask])["rmse_ln"],
                                     "d_rmse": bb})
    return rows, mech


def summary(out: str = "results", seed: int = SEED) -> None:
    out_dir = Path(out)
    rows = _leg1(out_dir, seed)
    rows2, mech = _leg2(out_dir, seed)
    rows += rows2
    if not any("d_rmse" in r for r in rows):
        sys.exit("no ablation runs found in results/ (run scripts/50_nominal_location_ablation.sh)")

    md = ["# Ablation — nominal location on top of raw coordinates (TabPFN-3.5, zero-shot)", "",
          "Difference = variant − plain table; negative is better. CI95 by block bootstrap "
          "(2,000 resamples of the spatial blocks); Wilcoxon paired over blocks.", "",
          "| Leg | Input | RMSE_ln | MAPE | R²_ln | Moran I | ΔRMSE_ln [CI95] | ΔMAPE p.p. [CI95] | Wilcoxon p | blocks better |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "d_rmse" in r:
            dm = r["d_mape"]
            tail = (f"{_fmt_ci(r['d_rmse'])} | {dm['diff']:+.2f} [{dm['ci95'][0]:+.2f}; {dm['ci95'][1]:+.2f}] | "
                    f"{r['wilcoxon_p']:.3f} | {r['blocks_better']}/{r['n_blocks']}")
        else:
            tail = "— | — | — | —"
        md.append(f"| {r['leg']} | {r['variant']} | {r['rmse_ln']:.4f} | {r['mape_pct']:.1f}% | "
                  f"{r['r2_ln']:.3f} | {r['moran_I']:.3f} | {tail} |")
    if mech:
        md += ["", "## Mechanism check (out-of-time leg)", "",
               "The same difference, split by whether the 2026 row's value occurs in the training base.", "",
               "| Leg | Input | Subset of 2026 | n (share) | RMSE plain | RMSE variant | ΔRMSE_ln [CI95] |",
               "|---|---|---|---|---|---|---|"]
        for m in mech:
            md.append(f"| {m['leg']} | {m['variant']} | {m['subset']} | {m['n']:,} ({100 * m['share']:.0f}%) | "
                      f"{m['rmse_plain']:.4f} | {m['rmse_variant']:.4f} | {_fmt_ci(m['d_rmse'])} |")
    text = "\n".join(md) + "\n"
    (out_dir / "nominal_ablation_summary.md").write_text(text, encoding="utf-8")
    (out_dir / "nominal_ablation_summary.json").write_text(
        json.dumps({"seed": seed, "runs": rows, "mechanism": mech}, indent=2), encoding="utf-8")
    print(text)
    print(f"Wrote {out_dir}/nominal_ablation_summary.md and .json")


# --------------------------------------------------------------------------
# descriptors (offline): a name in place of the coordinates, against every model
# --------------------------------------------------------------------------
def _leg_frames(out_dir: Path, seed: int):
    """Yield (leg name, y, blocks, test frame, train file, reader) for the three settings.

    ``reader(label)`` returns (metrics dict, per-block RMSE list, predictions) or None.
    """
    df = load_base(LEVEL_A)

    def read_cv(label):
        f = out_dir / f"cv_{label}_2025_level_a.json"
        if not f.exists():
            return None
        r = json.loads(f.read_text())["per_seed"][str(seed)]
        yhat = pd.read_csv(out_dir / f"oof_{label}_2025_level_a_seed{seed}.csv")["yhat_ln"].to_numpy(float)
        return ({**r["pooled"], "moran_I": r["moran_oof"]["I"]},
                [x["rmse_ln"] for x in r["per_fold"]], yhat)

    yield ("Leg 1 — leave-one-block-out CV, Level A (24k)", df[TARGET].to_numpy(float),
           df[BLOCK_COL].to_numpy(), None, None, read_cv)

    test_df = load_base(TEST_FILE)
    for train, name in [("level_a", "Leg 2 — fit on Level A (24k), predict 2026"),
                        ("full", "Leg 2 — fit on the full 2025 base (82k), predict 2026")]:
        def read_oot(label, train=train):
            f = out_dir / f"oot_{label}_{train}.json"
            if not f.exists():
                return None
            r = json.loads(f.read_text())["per_seed"][str(seed)]
            yhat = pd.read_csv(out_dir / f"oot_pred_{label}_{train}_seed{seed}.csv")["yhat_ln"].to_numpy(float)
            return ({**r["pooled"], "moran_I": r["moran_test"]["I"], "bias_ln": r["bias_ln"]},
                    [x["rmse_ln"] for x in r["per_block"]], yhat)

        yield (name, test_df[TARGET].to_numpy(float), test_df[BLOCK_COL].to_numpy(),
               test_df, TRAIN_FILES[train], read_oot)


def descriptors(out: str = "results", seed: int = SEED) -> None:
    out_dir = Path(out)
    legs, md = [], [
        "# A name in place of the coordinates — TabPFN-3.5 zero-shot against every model", "",
        "TabPFN-3.5 rows marked *no lat/lon* drop latitude and longitude from the plain table and "
        "carry the named descriptor instead; rows marked *no station distance* also drop the "
        "distance to the nearest station, which is computed from the coordinates, so that the "
        "postal code is the only spatial information left. The baselines are untouched: SAR keeps "
        "its weights matrix, XGBoost its k-NN lag, rotated coordinates and station distance. "
        "Everything else is unchanged (same rows, folds, seed). "
        "Differences are RMSE_ln of the row minus RMSE_ln of the reference, negative is better, "
        "CI95 by block bootstrap (2,000 resamples of the spatial blocks). *Signal recovered* places "
        "the row between the floor of its own family (0 %) and the plain table (100 %) on the "
        "RMSE_ln scale.", ""]
    found = False
    for name, y, blocks, test_df, train_file, read in _leg_frames(out_dir, seed):
        plain = read("tabpfn_t0")
        if plain is None:
            continue
        refs = {"plain": plain, "xgb": read("xgb_lag"), "sar": read("sar_gm")}
        floors = {st: read("tabpfn_t0" + nominal_tag((), coords=False, station=st)) for st in (True, False)}
        rows = []
        for key, label in BASELINES:
            r = read(key)
            if r is not None:
                rows.append({"model": label, "label": key, **_pick(r[0])})
        rows.append({"model": "**TabPFN-3.5, plain table (lat/lon)**", "label": "tabpfn_t0",
                     **_pick(plain[0]), "recovered": 1.0 if any(floors.values()) else None})
        for nominal, st, text in DESCRIPTORS:
            tag = "tabpfn_t0" + nominal_tag(nominal, coords=False, station=st)
            r = read(tag)
            if r is None:
                continue
            found = True
            floor = floors[st]
            row = {"model": f"TabPFN-3.5, {FAMILY[st]}: {text}", "label": tag, **_pick(r[0])}
            for ref_key, ref in refs.items():
                if ref is None:
                    continue
                row[f"d_vs_{ref_key}"] = block_bootstrap_diff(y, r[2], ref[2], blocks, "rmse_ln")
                row[f"blocks_better_vs_{ref_key}"] = int(sum(a < b for a, b in zip(r[1], ref[1])))
                row["n_blocks"] = len(r[1])
            if floor is not None:
                span = floor[0]["rmse_ln"] - plain[0]["rmse_ln"]
                row["recovered"] = (floor[0]["rmse_ln"] - r[0]["rmse_ln"]) / span if span > 0 else None
            rows.append(row)

        md += [f"## {name}", "",
               "| Model | RMSE_ln | MAPE | R²_ln | Moran I | signal recovered | Δ vs TabPFN plain [CI95] | "
               "Δ vs XGBoost+lag [CI95] | Δ vs SAR [CI95] |", "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            rec = "—" if r.get("recovered") is None else f"{100 * r['recovered']:.0f} %"
            cells = [(_fmt_ci(r[k]) + f" ({r['blocks_better_vs_' + k[5:]]}/{r['n_blocks']})") if k in r else "—"
                     for k in ("d_vs_plain", "d_vs_xgb", "d_vs_sar")]
            md.append(f"| {r['model']} | {r['rmse_ln']:.4f} | {r['mape_pct']:.1f}% | {r['r2_ln']:.3f} | "
                      f"{r['moran_I']:.3f} | {rec} | " + " | ".join(cells) + " |")
        md.append("")
        md.append("(n/10) = spatial blocks in which the row has the lower RMSE.")
        md.append("")

        mech = []
        if test_df is not None and any(floors.values()):
            train_df = load_base(train_file)
            for nominal, st, text in DESCRIPTORS:
                r = read("tabpfn_t0" + nominal_tag(nominal, coords=False, station=st))
                floor = floors[st]
                if r is None or not nominal or floor is None:
                    continue
                col = "cep" if nominal[-1].startswith("cep") else nominal[-1]
                for subset, mask in _seen_masks(train_df, test_df, col).items():
                    if mask.sum() < 200:
                        continue
                    f_, v_, p_ = (regression_metrics(y[mask], a[2][mask])["rmse_ln"] for a in (floor, r, plain))
                    mech.append({"descriptor": f"{text} ({FAMILY[st]})", "subset": subset, "n": int(mask.sum()),
                                 "share": float(mask.mean()), "rmse_floor": f_, "rmse_descriptor": v_,
                                 "rmse_plain": p_,
                                 "recovered": (f_ - v_) / (f_ - p_) if f_ > p_ else None})
            if mech:
                md += ["Split of the 2026 rows by whether the value occurs in the training base:", "",
                       "| Descriptor | Subset of 2026 | n (share) | RMSE floor | RMSE descriptor | "
                       "RMSE plain table | signal recovered |", "|---|---|---|---|---|---|---|"]
                for m in mech:
                    rec = "—" if m["recovered"] is None else f"{100 * m['recovered']:.0f} %"
                    md.append(f"| {m['descriptor']} | {m['subset']} | {m['n']:,} ({100 * m['share']:.0f}%) | "
                              f"{m['rmse_floor']:.4f} | {m['rmse_descriptor']:.4f} | {m['rmse_plain']:.4f} | {rec} |")
                md.append("")
        legs.append({"leg": name, "rows": rows, "seen_unseen": mech})
    if not found:
        sys.exit("no --no-coords runs found in results/ (run scripts/50_nominal_location_ablation.sh replace)")
    text = "\n".join(md) + "\n"
    (out_dir / "location_descriptor_summary.md").write_text(text, encoding="utf-8")
    (out_dir / "location_descriptor_summary.json").write_text(
        json.dumps({"seed": seed, "legs": legs}, indent=2), encoding="utf-8")
    print(text)
    print(f"Wrote {out_dir}/location_descriptor_summary.md and .json")


def _pick(m: dict) -> dict:
    return {k: m[k] for k in ("rmse_ln", "mape_pct", "r2_ln", "moran_I", "bias_ln") if k in m}


def usage() -> None:
    if not os.environ.get("TABPFN_TOKEN"):
        sys.exit("TABPFN_TOKEN is not set (export TABPFN_TOKEN=...)")
    from tabpfn_client import get_api_usage
    print(get_api_usage())


def main() -> None:
    ap = argparse.ArgumentParser(description="Location ablations: API probe and offline summaries")
    ap.add_argument("command", choices=["probe", "summary", "descriptors", "usage"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.command == "probe":
        sys.exit(probe())
    if args.command == "usage":
        usage()
    elif args.command == "descriptors":
        descriptors(args.out)
    else:
        summary(args.out)


if __name__ == "__main__":
    main()
