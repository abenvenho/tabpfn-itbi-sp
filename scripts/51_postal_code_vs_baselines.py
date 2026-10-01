#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
51_postal_code_vs_baselines.py — the postal code in place of the coordinates,
against every seed of the spatial baselines
=============================================================================
``results/location_descriptor_summary.md`` pairs each TabPFN-3.5 variant with
the seed-42 run of XGBoost + k-NN lag, the run behind every paired test in the
README. XGBoost is stochastic and was run with three seeds (42, 43, 44), and on
the full 2025 base they spread from 0.283 to 0.294 in RMSE_ln. This script
repeats the paired comparison of the two postal-code variants against each of
the three seeds, against each seed of the geographically weighted XGBoost
(``src/model_gxgb.py``) and against SAR, so that no claim rests on the least
favourable seed of the opponent.

TabPFN-3.5 variants (zero-shot, seed 42, latitude and longitude removed):

* ``tabpfn_t0_nocoord_nostation_nom_cep`` — the postal code as a string is the
  only spatial information (the distance to a station is also removed, since
  it is computed from the coordinates);
* ``tabpfn_t0_nocoord_nom_cep`` — the postal code as a string, plus the
  distance to the nearest station that XGBoost also receives.

The baselines are the ones of the main analysis, untouched.

Nothing is refitted and no API call is made: the script reads the versioned
predictions (``results/oof_*`` for leg 1, ``results/oot_pred_*`` for leg 2)
and the Moran's I already stored in the run records. It needs numpy and pandas
only. The block bootstrap is the one of ``src/protocol.py`` (2,000 resamples of
the ten spatial blocks, seed 42); the Wilcoxon signed-rank test on the ten
per-block RMSEs is computed exactly, by enumeration.

Usage, from the repository root:

    python scripts/51_postal_code_vs_baselines.py

Writes ``results/postal_code_vs_baselines.md`` and ``.json``.
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

TABPFN = {
    "tabpfn_t0_nocoord_nostation_nom_cep": "TabPFN-3.5, postal code only",
    "tabpfn_t0_nocoord_nom_cep": "TabPFN-3.5, postal code + station distance",
}
OPPONENTS = [("xgb_lag", 42), ("xgb_lag", 43), ("xgb_lag", 44),
             ("gxgb", 42), ("gxgb", 43), ("gxgb", 44), ("sar_gm", 42)]
OPPONENT_NAME = {"xgb_lag": "XGBoost + rotated coords + k-NN-8 lag",
                 "gxgb": "Geographically weighted XGBoost", "sar_gm": "SAR lag"}

LEGS = {
    "leg1": {
        "title": "Leg 1 — leave-one-block-out CV, Level A (24k)",
        "pred": lambda lbl, s: RESULTS / f"oof_{lbl}_2025_level_a_seed{s}.csv",
        "record": lambda lbl: RESULTS / f"cv_{lbl}_2025_level_a.json",
        "moran_key": "moran_oof",
    },
    "leg2_level_a": {
        "title": "Leg 2 — fit on Level A (24k), predict the 47,810 transactions of 2026",
        "pred": lambda lbl, s: RESULTS / f"oot_pred_{lbl}_level_a_seed{s}.csv",
        "record": lambda lbl: RESULTS / f"oot_{lbl}_level_a.json",
        "moran_key": "moran_test",
    },
    "leg2_full": {
        "title": "Leg 2 — fit on the full 2025 base (82k), predict the 47,810 transactions of 2026",
        "pred": lambda lbl, s: RESULTS / f"oot_pred_{lbl}_full_seed{s}.csv",
        "record": lambda lbl: RESULTS / f"oot_{lbl}_full.json",
        "moran_key": "moran_test",
    },
}


def metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    """Same definitions as ``src.protocol.regression_metrics``."""
    resid = y - yhat
    ape = np.abs(np.exp(yhat) - np.exp(y)) / np.exp(y)
    return {"rmse_ln": float(np.sqrt(np.mean(resid ** 2))),
            "mape_pct": float(100 * ape.mean())}


def block_bootstrap(y, a, b, blocks, metric):
    """Pooled difference A − B and its 95 % CI, resampling spatial blocks
    (as ``src.protocol.block_bootstrap_diff``)."""
    rng = np.random.default_rng(SEED)
    uniq = np.unique(blocks)
    by_block = {k: np.flatnonzero(blocks == k) for k in uniq}
    diffs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        idx = np.concatenate([by_block[k] for k in rng.choice(uniq, size=len(uniq), replace=True)])
        diffs[i] = metrics(y[idx], a[idx])[metric] - metrics(y[idx], b[idx])[metric]
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": metrics(y, a)[metric] - metrics(y, b)[metric],
            "ci95": [float(lo), float(hi)]}


def wilcoxon_exact(d: np.ndarray) -> float:
    """Two-sided exact Wilcoxon signed-rank p-value (zeros dropped)."""
    d = d[d != 0]
    n = len(d)
    if n == 0:
        return 1.0
    ranks = pd.Series(np.abs(d)).rank().to_numpy()
    w_obs = ranks[d > 0].sum()
    dist = np.array([sum(r for r, s in zip(ranks, signs) if s)
                     for signs in itertools.product((0, 1), repeat=n)])
    p = 2 * min((dist <= w_obs + 1e-9).mean(), (dist >= w_obs - 1e-9).mean())
    return float(min(1.0, p))


def load(leg: dict, label: str, seed: int) -> pd.DataFrame:
    return pd.read_csv(leg["pred"](label, seed), dtype={"sql": str})


def moran(leg: dict, label: str, seed: int) -> float:
    rec = json.loads(leg["record"](label).read_text())
    return float(rec["per_seed"][str(seed)][leg["moran_key"]]["I"])


def compare(leg: dict, a_label: str, b_label: str, b_seed: int) -> dict:
    a = load(leg, a_label, SEED)
    b = load(leg, b_label, b_seed)
    if not (a["sql"].to_numpy() == b["sql"].to_numpy()).all():
        raise ValueError(f"row order differs: {a_label} vs {b_label} seed {b_seed}")
    y = a["y_ln"].to_numpy(float)
    blocks = a["bloco"].to_numpy()
    ya, yb = a["yhat_ln"].to_numpy(float), b["yhat_ln"].to_numpy(float)
    per_block = [(metrics(y[blocks == k], ya[blocks == k])["rmse_ln"],
                  metrics(y[blocks == k], yb[blocks == k])["rmse_ln"])
                 for k in np.unique(blocks)]
    d_blocks = np.array([pa - pb for pa, pb in per_block])
    return {
        "a": a_label, "b": b_label, "b_seed": b_seed,
        "a_metrics": {**metrics(y, ya), "moran_I": moran(leg, a_label, SEED)},
        "b_metrics": {**metrics(y, yb), "moran_I": moran(leg, b_label, b_seed)},
        "rmse_ln": block_bootstrap(y, ya, yb, blocks, "rmse_ln"),
        "mape_pct": block_bootstrap(y, ya, yb, blocks, "mape_pct"),
        "blocks_won": int((d_blocks < 0).sum()),
        "n_blocks": int(len(d_blocks)),
        "wilcoxon_p": wilcoxon_exact(d_blocks),
    }


def main() -> None:
    out = {"seed_tabpfn": SEED, "n_boot": N_BOOT, "legs": {}}
    md = ["# The postal code in place of the coordinates — against every seed of the baselines", "",
          "TabPFN-3.5 zero-shot without latitude and longitude, carrying the raw postal code (CEP) "
          "as a string, paired with each seed of XGBoost + k-NN-8 lag, each seed of the geographically "
          "weighted XGBoost, and SAR. The baselines "
          "keep all of their spatial inputs. Δ = TabPFN − opponent, negative favours TabPFN; CI95 "
          f"by block bootstrap ({N_BOOT:,} resamples of the ten spatial blocks); blocks = spatial "
          "blocks in which TabPFN has the lower RMSE; p = exact Wilcoxon signed-rank test on the "
          "ten per-block RMSEs. Generated by `scripts/51_postal_code_vs_baselines.py` from the "
          "versioned predictions; no model refitted.", ""]
    for key, leg in LEGS.items():
        rows = []
        for a_label in TABPFN:
            for b_label, b_seed in OPPONENTS:
                if not leg["pred"](b_label, b_seed).exists():
                    continue                     # baseline not run for this leg (yet)
                rows.append(compare(leg, a_label, b_label, b_seed))
        out["legs"][key] = {"title": leg["title"], "comparisons": rows}
        md += [f"## {leg['title']}", "",
               "| TabPFN-3.5 variant | Opponent | RMSE_ln TabPFN · opp. | MAPE TabPFN · opp. "
               "| Moran I TabPFN · opp. | ΔRMSE_ln [CI95] | ΔMAPE pp [CI95] | blocks | p |",
               "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            am, bm = r["a_metrics"], r["b_metrics"]
            dr, dm = r["rmse_ln"], r["mape_pct"]
            opp = f"{OPPONENT_NAME[r['b']]}" + (f", seed {r['b_seed']}" if r["b"] != "sar_gm" else "")
            md.append(
                f"| {TABPFN[r['a']]} | {opp} "
                f"| {am['rmse_ln']:.4f} · {bm['rmse_ln']:.4f} "
                f"| {am['mape_pct']:.1f} % · {bm['mape_pct']:.1f} % "
                f"| {am['moran_I']:.3f} · {bm['moran_I']:.3f} "
                f"| {dr['diff']:+.4f} [{dr['ci95'][0]:+.4f}; {dr['ci95'][1]:+.4f}] "
                f"| {dm['diff']:+.2f} [{dm['ci95'][0]:+.2f}; {dm['ci95'][1]:+.2f}] "
                f"| {r['blocks_won']}/{r['n_blocks']} | {r['wilcoxon_p']:.3f} |")
        md.append("")
    (RESULTS / "postal_code_vs_baselines.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (RESULTS / "postal_code_vs_baselines.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
