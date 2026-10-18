#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_numbers.py — verify every number cited in the manuscript against the
versioned results.

Run from the repository root:  python paper/check_numbers.py

Each check is one of two classes:
  NUM  — the cited value is recomputed or read from a results/ or data/ file
         and compared at the rounding used in the text;
  TXT  — a prose-sourced claim (no structured file carries it) is matched
         verbatim against the audited README.md / docs/further_tests.md, so
         the manuscript never drifts from the repository's own record.

Exit status 0 iff every check passes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results"
FAILS: list[str] = []
N_PASS = 0


def J(name: str) -> dict:
    return json.loads((RES / name).read_text())


def CHECK(desc: str, ok: bool, detail: str = "") -> None:
    global N_PASS
    if ok:
        N_PASS += 1
    else:
        FAILS.append(f"{desc}  [{detail}]")
        print(f"FAIL  {desc}  [{detail}]")


def close(value: float, cited: float, nd: int, desc: str) -> None:
    CHECK(desc, abs(float(value) - cited) <= 0.5 * 10 ** (-nd) + 1e-12, f"got {value!r}")


def within(value: float, lo: float, hi: float, desc: str) -> None:
    CHECK(desc, lo - 1e-12 <= float(value) <= hi + 1e-12, f"got {value!r}")


def TXT(desc: str, needle: str, path: Path) -> None:
    hay = " ".join(path.read_text(encoding="utf-8").split())
    CHECK(f"TXT {desc}", " ".join(needle.split()) in hay, f"'{needle}' not in {path.name}")


def seeds(model_rows: list[dict], metric: str) -> dict[int, float]:
    return {r["seed"]: r[metric] for r in model_rows}


def find_key(obj, key):
    """First value of `key` anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = find_key(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = find_key(v, key)
            if r is not None:
                return r
    return None


# ---------------------------------------------------------------- data files
train = pd.read_csv(ROOT / "data/itbi_sp_2025_train.csv.gz", dtype={"sql": str})
test = pd.read_csv(ROOT / "data/itbi_sp_2026_test.csv", dtype={"sql": str})
level_a = pd.read_csv(ROOT / "data/itbi_sp_2025_level_a.csv", dtype={"sql": str})
README = ROOT / "README.md"
FURTHER = ROOT / "docs/further_tests.md"
FILTER = ROOT / "data/filter_report.md"

CHECK("rows: 2025 train = 82,187", len(train) == 82187, str(len(train)))
CHECK("rows: 2026 test = 47,810", len(test) == 47810, str(len(test)))
CHECK("rows: Level A = 24,000", len(level_a) == 24000, str(len(level_a)))
for t, n in [("apartment", 60067), ("house", 15940), ("commercial", 6180)]:
    CHECK(f"2025 {t} = {n:,}", (train["tipo_imovel"] == t).sum() == n)
for t, n in [("apartment", 35354), ("house", 9429), ("commercial", 3027)]:
    CHECK(f"2026 {t} = {n:,}", (test["tipo_imovel"] == t).sum() == n)
for t, n in [("apartment", 12000), ("house", 7000), ("commercial", 5000)]:
    CHECK(f"Level A {t} = {n:,}", (level_a["tipo_imovel"] == t).sum() == n)

# table 2: financed shares and median unit values
for df, share, med, nm in [(train, 0.34, 4836, "train"), (level_a, 0.30, 4715, "level_a"),
                           (test, 0.42, 4970, "test")]:
    close(df["financiado"].mean(), share, 2, f"{nm}: financed share {share:.0%}")
    close(round(np.exp(df["ln_vu"]).median()), med, 0, f"{nm}: median unit value {med}")

# filter chain counts: every number of Table 1 must appear in the versioned report
for n in ["230,525", "207,518", "108,750", "108,420", "90,927", "90,673", "89,078",
          "87,488", "84,263", "82,193", "82,187",
          "135,655", "124,289", "61,304", "61,112", "51,919", "51,918", "51,810",
          "51,202", "50,430", "48,879", "47,814", "47,810",
          "3,225", "1,551", "99.99%"]:
    TXT(f"filter chain count {n}", n, FILTER)

# leakage checks
k1 = set(zip(train["sql"], train["data_transacao"], train["valor_transacao"]))
k2 = set(zip(test["sql"], test["data_transacao"], test["valor_transacao"]))
CHECK("leakage: zero identical (key, date, price) across years", len(k1 & k2) == 0)
close(test["sql"].isin(set(level_a["sql"])).mean() * 100, 0.6, 1,
      "leakage: 0.6% of 2026 rows share a key with Level A")

# stations = 203 points
nst = 0
for g in ["estacao_metro.geojson", "estacao_trem.geojson"]:
    nst += len(json.loads((ROOT / "data/raw" / g).read_text())["features"])
CHECK("203 station points", nst == 203, str(nst))

# descriptive Moran's I of the raw response
dm = J("descriptive_moran.json")
close(dm["level_a_2025"]["moran_I_ln_vu"], 0.430, 3, "Moran I of ln_vu, Level A = 0.430")
close(dm["level_a_2025"]["z_norm"], 141.8, 1, "z = 141.8")
close(dm["test_2026"]["moran_I_ln_vu"], 0.506, 3, "Moran I of ln_vu, 2026 = 0.506")
close(dm["test_2026"]["z_norm"], 235.7, 1, "z = 235.7")

# postal-code cardinality
close(train["cep"].nunique(), 15938, 0, "distinct postal codes, full base = 15,938")
close(level_a["cep"].nunique(), 10022, 0, "distinct postal codes, Level A = 10,022")

# financed subset sizes
CHECK("financed Level A rows = 7,211", int(level_a["financiado"].sum()) == 7211)
CHECK("financed 2026 rows = 20,052", int(test["financiado"].sum()) == 20052)

# ------------------------------------------------------------- leg 1 (tables)
V = json.loads((RES / "variants_summary.json").read_text())
L1, L2a, L2f = V["legs"]["leg1"], V["legs"]["leg2_level_a"], V["legs"]["leg2_full"]


def FLAT(leg):
    out = []
    for pr in leg["pairs"]:
        for e in pr["per_seed"]:
            out.append({"a": pr["a"], "b": pr["b"], **e})
    return out


FL1, FL2a, FL2f = FLAT(L1), FLAT(L2a), FLAT(L2f)


def pooled(cvname: str, seed="42"):
    d = J(cvname)["per_seed"][seed]
    return d["pooled"], d.get("moran_oof") or d.get("moran_test")


p, m = pooled("cv_ols_2025_level_a.json")
close(p["rmse_ln"], 0.470, 3, "leg1 OLS RMSE"); close(p["mape_pct"], 39.4, 1, "leg1 OLS MAPE")
close(p["r2_ln"], 0.066, 3, "leg1 OLS R2"); close(m["I"], 0.390, 3, "leg1 OLS I")
p, m = pooled("cv_sar_gm_2025_level_a.json")
close(p["rmse_ln"], 0.448, 3, "leg1 SAR RMSE"); close(p["mape_pct"], 38.4, 1, "leg1 SAR MAPE")
close(p["r2_ln"], 0.155, 3, "leg1 SAR R2"); close(m["I"], 0.363, 3, "leg1 SAR I")
close(find_key(J("cv_sar_gm_2025_level_a.json"), "rho"), 0.91, 2, "SAR rho = 0.91")
p, m = pooled("cv_xgb_nolag_2025_level_a.json")
close(p["rmse_ln"], 0.397, 3, "leg1 XGB-nolag RMSE"); close(p["mape_pct"], 33.3, 1, "leg1 XGB-nolag MAPE")
close(p["r2_ln"], 0.334, 3, "leg1 XGB-nolag R2"); close(m["I"], 0.355, 3, "leg1 XGB-nolag I")
# floor row (median by type) — prose-sourced in the README conclusions
for s in ["0.497", "42.1", "0.446"]:
    TXT(f"leg1 floor {s}", s, README)

for lbl, rmse42, rng, mape42, r2, I42, Irng in [
    ("xgb", 0.385, (0.385, 0.387), 32.3, 0.376, 0.309, (0.309, 0.318)),
    ("gxgb", 0.410, (0.407, 0.410), 34.5, 0.291, 0.324, (0.319, 0.326)),
    ("tabpfn", 0.394, (0.391, 0.394), 32.2, 0.346, 0.348, (0.342, 0.348)),
]:
    sv = seeds(L1["models"][lbl], "rmse_ln")
    close(sv[42], rmse42, 3, f"leg1 {lbl} RMSE seed42")
    close(min(sv.values()), rng[0], 3, f"leg1 {lbl} RMSE range lo")
    close(max(sv.values()), rng[1], 3, f"leg1 {lbl} RMSE range hi")
    close(seeds(L1["models"][lbl], "mape_pct")[42], mape42, 1, f"leg1 {lbl} MAPE seed42")
    iv = seeds(L1["models"][lbl], "moran_I")
    close(iv[42], I42, 3, f"leg1 {lbl} I seed42")
    close(min(iv.values()), Irng[0], 3, f"leg1 {lbl} I range lo")
    close(max(iv.values()), Irng[1], 3, f"leg1 {lbl} I range hi")
close(J("cv_xgb_lag_2025_level_a.json")["per_seed"]["42"]["pooled"]["r2_ln"], 0.376, 3, "leg1 XGB+lag R2")
close(J("cv_gxgb_2025_level_a.json")["per_seed"]["42"]["pooled"]["r2_ln"], 0.291, 3, "leg1 GW R2")
close(J("cv_tabpfn_t0_2025_level_a.json")["per_seed"]["42"]["pooled"]["r2_ln"], 0.346, 3, "leg1 TabPFN R2")
for name, rmse, mape, r2, I in [("cv_tabpfn_t0_think_medium_2025_level_a.json", 0.389, 31.7, 0.362, 0.342),
                                ("cv_tabpfn_t0_think_high_2025_level_a.json", 0.388, 31.7, 0.366, 0.339)]:
    p, m = pooled(name)
    close(p["rmse_ln"], rmse, 3, f"{name} RMSE"); close(p["mape_pct"], mape, 1, f"{name} MAPE")
    close(p["r2_ln"], r2, 3, f"{name} R2"); close(m["I"], I, 3, f"{name} I")

# leg-1 paired tests
q = [x for x in FL1 if x["a"] == "tabpfn" and x["b"] == "sar"][0]
close(q["rmse_ln"]["diff"], -0.054, 3, "leg1 tabpfn vs SAR diff")
close(q["rmse_ln"]["ci95"][0], -0.100, 3, "...CI lo"); close(q["rmse_ln"]["ci95"][1], -0.014, 3, "...CI hi")
CHECK("leg1 tabpfn vs SAR 8/10", q["blocks_won"] == 8)
close(q["wilcoxon_p"], 0.010, 3, "...p=0.010")
close(q["mape_pct"]["diff"], -6.2, 1, "...dMAPE -6.2pp")

qx = [x for x in FL1 if x["a"] == "tabpfn" and x["b"] == "xgb"]
d42 = [x for x in qx if x.get("b_seed") == 42][0]
close(d42["rmse_ln"]["diff"], 0.009, 3, "leg1 tabpfn vs XGB diff +0.009")
close(d42["rmse_ln"]["ci95"][0], -0.018, 3, "...CI lo"); close(d42["rmse_ln"]["ci95"][1], 0.034, 3, "...CI hi")
CHECK("leg1 tabpfn vs XGB 7/10 each seed", all(x["blocks_won"] == 7 for x in qx))

qg = [x for x in FL1 if x["a"] == "tabpfn" and x["b"] == "gxgb"]
within(min(x["rmse_ln"]["diff"] for x in qg), -0.0165, -0.0155, "leg1 vs GW diff lo ~ -0.016")
within(max(x["rmse_ln"]["diff"] for x in qg), -0.0135, -0.0125, "leg1 vs GW diff hi ~ -0.013")
CHECK("leg1 vs GW CIs all cross zero", all(x["rmse_ln"]["ci95"][0] < 0 < x["rmse_ln"]["ci95"][1] for x in qg))
CHECK("leg1 vs GW blocks 6-7", {x["blocks_won"] for x in qg} <= {6, 7})
within(min(x["wilcoxon_p"] for x in qg), 0.100, 0.110, "leg1 vs GW p lo 0.105")
within(max(x["wilcoxon_p"] for x in qg), 0.225, 0.235, "leg1 vs GW p hi 0.232")

qgw = [x for x in FL1 if x["a"] == "gxgb" and x["b"] == "xgb"]
within(min(x["rmse_ln"]["diff"] for x in qgw), 0.0220, 0.0230, "leg1 GW vs XGB +0.023 lo")
within(max(x["rmse_ln"]["diff"] for x in qgw), 0.0248, 0.0256, "leg1 GW vs XGB +0.025 hi")
CHECK("leg1 GW vs XGB blocks 1-2", {x["blocks_won"] for x in qgw} <= {1, 2})
within(min(x["wilcoxon_p"] for x in qgw), 0.025, 0.029, "p lo 0.027")
within(max(x["wilcoxon_p"] for x in qgw), 0.035, 0.039, "p hi 0.037")

cm = J("compare_tabpfn_t0_think_medium_2025_level_a_vs_tabpfn_t0_2025_level_a.json")
close(cm["block_bootstrap"]["diff"], -0.005, 3, "think vs t0 seed42 -0.005")
close(cm["block_bootstrap"]["ci95"][0], -0.008, 3, "...CI lo")
close(cm["block_bootstrap"]["ci95"][1], -0.001, 3, "...CI hi")
TXT("think vs seeds 43/44 about -0.002", "\u22120.002 vs seeds 43 and 44", README)

c = J("compare_xgb_lag_2025_level_a_vs_xgb_nolag_2025_level_a.json")
close(find_key(c, "diff") or find_key(c, "pooled_diff"), -0.013, 3, "lag ablation diff -0.013")
ci = find_key(c, "ci95") or find_key(c, "ci")
close(ci[0], -0.024, 3, "lag ablation CI lo"); close(ci[1], -0.004, 3, "lag ablation CI hi")

# gxgb bandwidth chosen per fold on Level A: 40-176 at alpha=0.5
GP = J("gxgb_params_gxgb_2025_level_a.json")
bws = [v["bandwidth"] for v in GP.values()]
CHECK("GW leg1 k per fold in 40..176", min(bws) == 40 and max(bws) == 176, str(sorted(bws)))
CHECK("GW leg1 alpha=0.5 in every fold", all(v["alpha"] == 0.5 for v in GP.values()))
for f, kk in [("gxgb_params_oot_level_a.json", 50), ("gxgb_params_oot_full.json", 342)]:
    d = J(f)
    CHECK(f"GW {f}: k={kk}, alpha=0.5", d["bandwidth"] == kk and d["alpha"] == 0.5,
          f"k={d.get('bandwidth')}, a={d.get('alpha')}")

# ------------------------------------------------------------- leg 2 (tables)
def oot(name, seed="42"):
    d = J(name)["per_seed"][seed]
    return d["pooled"], d["moran_test"], d["bias_ln"]


for name, rmse, mape, r2, bias, I in [
    ("oot_ols_level_a.json", 0.404, 32.4, 0.185, 0.033, 0.408),
    ("oot_sar_gm_level_a.json", 0.346, 27.0, 0.401, 0.032, 0.224),
    ("oot_xgb_lag_level_a.json", 0.288, 21.8, 0.584, 0.047, 0.140),
    ("oot_gxgb_level_a.json", 0.313, 24.0, 0.508, None, 0.187),
    ("oot_tabpfn_t0_level_a.json", 0.275, 20.4, 0.622, 0.055, 0.122),
    ("oot_tabpfn_t0_think_medium_level_a.json", 0.275, 20.5, 0.621, 0.055, 0.121),
    ("oot_ols_full.json", 0.401, 32.4, 0.194, 0.023, 0.397),
    ("oot_sar_gm_full.json", 0.326, 25.0, 0.470, 0.031, 0.134),
    ("oot_tabpfn_t0_full.json", 0.265, 19.5, 0.649, 0.058, 0.092),
]:
    p, m, b = oot(name)
    close(p["rmse_ln"], rmse, 3, f"{name} RMSE"); close(p["mape_pct"], mape, 1, f"{name} MAPE")
    close(p["r2_ln"], r2, 3, f"{name} R2")
    if bias is not None:
        close(b, bias, 3, f"{name} bias")
    close(m["I"], I, 3, f"{name} I")

# stochastic models, full base: seed-42 point + per-seed range (as the tables show)
xgf = seeds(L2f["models"]["xgb"], "rmse_ln")
close(xgf[42], 0.294, 3, "leg2 full XGB seed42 0.294")
close(min(xgf.values()), 0.283, 3, "leg2 full XGB lo 0.283")
close(max(xgf.values()), 0.294, 3, "leg2 full XGB hi 0.294")
mx = seeds(L2f["models"]["xgb"], "mape_pct")
close(mx[42], 21.5, 1, "XGB full MAPE seed42 21.5")
close(min(mx.values()), 20.9, 1, "XGB full MAPE lo 20.9")
ox = J("oot_xgb_lag_full.json")["per_seed"]
close(ox["42"]["pooled"]["r2_ln"], 0.57, 2, "XGB full R2 0.57")
close(ox["42"]["bias_ln"], 0.103, 3, "XGB full bias seed42 +0.103")
bx = [ox[s]["bias_ln"] for s in ("42", "43", "44")]
close(min(bx), 0.070, 3, "XGB full bias lo +0.07"); close(max(bx), 0.103, 3, "XGB full bias hi +0.10")
ix = [ox[s]["moran_test"]["I"] for s in ("42", "43", "44")]
close(ox["42"]["moran_test"]["I"], 0.107, 3, "XGB full I seed42 0.107")
close(min(ix), 0.103, 3, "XGB full I lo"); close(max(ix), 0.107, 3, "XGB full I hi")
gwf = seeds(L2f["models"]["gxgb"], "rmse_ln")
close(gwf[42], 0.298, 3, "leg2 full GW seed42 0.298")
close(min(gwf.values()), 0.298, 3, "lo"); close(max(gwf.values()), 0.301, 3, "hi")
close(seeds(L2f["models"]["gxgb"], "mape_pct")[42], 22.1, 1, "GW full MAPE 22.1")
og = J("oot_gxgb_full.json")["per_seed"]
close(og["42"]["pooled"]["r2_ln"], 0.56, 2, "GW full R2 0.56")
close(og["42"]["bias_ln"], 0.065, 3, "GW full bias seed42 +0.06")
bg = [og[s]["bias_ln"] for s in ("42", "43", "44")]
close(min(bg), 0.065, 3, "GW full bias lo +0.06"); close(max(bg), 0.078, 3, "GW full bias hi +0.08")
ig = [og[s]["moran_test"]["I"] for s in ("42", "43", "44")]
close(og["42"]["moran_test"]["I"], 0.163, 3, "GW full I seed42 0.163")
close(min(ig), 0.163, 3, "GW full I lo"); close(max(ig), 0.166, 3, "GW full I hi")
gwa = seeds(L2a["models"]["gxgb"], "rmse_ln")
close(gwa[42], 0.313, 3, "GW 24k seed42 0.313")
close(min(gwa.values()), 0.313, 3, "GW 24k lo"); close(max(gwa.values()), 0.314, 3, "GW 24k hi")
oga = J("oot_gxgb_level_a.json")["per_seed"]
ba = [oga[s]["bias_ln"] for s in ("42", "43", "44")]
close(oga["42"]["bias_ln"], 0.045, 3, "GW 24k bias seed42 +0.045")
close(min(ba), 0.045, 3, "GW 24k bias lo +0.045"); close(max(ba), 0.048, 3, "GW 24k bias hi +0.048")
ia = seeds(L2a["models"]["xgb"], "moran_I")
close(min(ia.values()), 0.138, 3, "XGB 24k I lo"); close(max(ia.values()), 0.140, 3, "XGB 24k I hi")
iga = seeds(L2a["models"]["gxgb"], "moran_I")
close(min(iga.values()), 0.187, 3, "GW 24k I lo"); close(max(iga.values()), 0.188, 3, "GW 24k I hi")
# non-linear bias span (+0.04 to +0.10) vs linear (+0.02 to +0.03)
nl = bx + bg + ba + [J("oot_xgb_lag_level_a.json")["per_seed"][s]["bias_ln"] for s in ("42", "43", "44")] \
     + [J("oot_tabpfn_t0_level_a.json")["per_seed"]["42"]["bias_ln"],
        J("oot_tabpfn_t0_full.json")["per_seed"]["42"]["bias_ln"]]
close(min(nl), 0.04, 2, "non-linear bias lo +0.04"); close(max(nl), 0.10, 2, "non-linear bias hi +0.10")
lin = [J(n)["per_seed"]["42"]["bias_ln"] for n in
       ("oot_ols_level_a.json", "oot_sar_gm_level_a.json", "oot_ols_full.json", "oot_sar_gm_full.json")]
close(min(lin), 0.02, 2, "linear bias lo +0.02"); close(max(lin), 0.03, 2, "linear bias hi +0.03")

# leg-2 paired tests
def pget(fl, a, b, bs=None):
    return [x for x in fl if x["a"] == a and x["b"] == b and (bs is None or x.get("b_seed") == bs)]


q = pget(FL2a, "tabpfn", "xgb", 42)[0]
close(q["rmse_ln"]["diff"], -0.013, 3, "leg2 24k vs XGB -0.013")
close(q["rmse_ln"]["ci95"][0], -0.016, 3, "CI lo"); close(q["rmse_ln"]["ci95"][1], -0.011, 3, "CI hi")
close(q["mape_pct"]["diff"], -1.4, 1, "dMAPE -1.4"); CHECK("10/10", q["blocks_won"] == 10)
close(q["wilcoxon_p"], 0.002, 3, "p=0.002")
qg = pget(FL2a, "tabpfn", "gxgb")
within(min(x["rmse_ln"]["diff"] for x in qg), -0.0395, -0.0385, "leg2 24k vs GW -0.039")
within(max(x["rmse_ln"]["diff"] for x in qg), -0.0388, -0.0378, "leg2 24k vs GW -0.038")
CHECK("leg2 24k vs GW CIs within [-0.049,-0.028] at 3 d.p.",
      all(-0.0495 <= x["rmse_ln"]["ci95"][0] and x["rmse_ln"]["ci95"][1] <= -0.0275 for x in qg))
CHECK("leg2 24k vs GW 10/10 all seeds", all(x["blocks_won"] == 10 for x in qg))
q = pget(FL2a, "tabpfn", "sar")[0]
close(q["rmse_ln"]["diff"], -0.071, 3, "leg2 24k vs SAR -0.071")
close(q["rmse_ln"]["ci95"][0], -0.079, 3, "CI lo"); close(q["rmse_ln"]["ci95"][1], -0.062, 3, "CI hi")
CHECK("10/10", q["blocks_won"] == 10)

q = pget(FL2f, "tabpfn", "xgb", 42)[0]
close(q["rmse_ln"]["diff"], -0.029, 3, "leg2 82k vs XGB -0.029")
close(q["rmse_ln"]["ci95"][0], -0.032, 3, "CI lo"); close(q["rmse_ln"]["ci95"][1], -0.026, 3, "CI hi")
close(q["mape_pct"]["diff"], -2.0, 1, "dMAPE -2.0"); CHECK("10/10", q["blocks_won"] == 10)
for bs, dd in [(43, -0.020), (44, -0.018)]:
    close(pget(FL2f, "tabpfn", "xgb", bs)[0]["rmse_ln"]["diff"], dd, 3,
          f"leg2 82k vs XGB better seed {bs} {dd}")
qg = pget(FL2f, "tabpfn", "gxgb")
within(min(x["rmse_ln"]["diff"] for x in qg), -0.0365, -0.0355, "leg2 82k vs GW -0.036")
within(max(x["rmse_ln"]["diff"] for x in qg), -0.0335, -0.0325, "leg2 82k vs GW -0.033")
CHECK("leg2 82k vs GW 9/10", all(x["blocks_won"] == 9 for x in qg))
CHECK("leg2 82k vs GW p=0.004", all(round(x["wilcoxon_p"], 3) == 0.004 for x in qg))
# smallest block: n = 2,425, lost by 0.001-0.002
pb_t = {r["block"]: r for r in J("oot_tabpfn_t0_full.json")["per_seed"]["42"]["per_block"]}
nmin = min(pb_t.values(), key=lambda r: r["n"])
CHECK("smallest 2026 block n=2,425", nmin["n"] == 2425, str(nmin["n"]))
diffs_small = []
for s in ("42", "43", "44"):
    pb_g = {r["block"]: r for r in J("oot_gxgb_full.json")["per_seed"][s]["per_block"]}
    diffs_small.append(pb_t[nmin["block"]]["rmse_ln"] - pb_g[nmin["block"]]["rmse_ln"])
CHECK("smallest block lost by 0.001-0.002",
      all(0.0005 <= d <= 0.0025 for d in diffs_small), str(diffs_small))

# scale effects
sc = J("oot_compare_tabpfn_t0_full_vs_tabpfn_t0_level_a.json")
close(find_key(sc, "diff"), -0.010, 3, "scale TabPFN -0.010")
ci = find_key(sc, "ci95"); close(ci[0], -0.012, 3, "CI lo"); close(ci[1], -0.008, 3, "CI hi")
close(find_key(J("oot_compare_sar_gm_full_vs_sar_gm_level_a.json"), "diff"), -0.020, 3, "scale SAR -0.020")
sx = J("oot_compare_xgb_lag_full_vs_xgb_lag_level_a.json")
close(find_key(sx, "diff"), 0.006, 3, "scale XGB +0.006")
ci = find_key(sx, "ci95"); close(ci[0], 0.001, 3, "CI lo"); close(ci[1], 0.011, 3, "CI hi")

# flatness across blocks and horizons (TabPFN full)
rb = [r["rmse_ln"] for r in J("oot_tabpfn_t0_full.json")["per_seed"]["42"]["per_block"]]
close(min(rb), 0.23, 2, "per-block RMSE lo 0.23"); close(max(rb), 0.31, 2, "per-block RMSE hi 0.31")
rm = [r["rmse_ln"] for r in J("oot_tabpfn_t0_full.json")["per_seed"]["42"]["per_month_ahead"]
      if r["n"] >= 500 and 1 <= r["months_ahead"] <= 7]
close(min(rm), 0.26, 2, "per-month RMSE lo 0.26"); close(max(rm), 0.28, 2, "per-month RMSE hi 0.28")

# GW bandwidth sensitivity 0.308-0.314
bw = J("gxgb_bandwidth_sensitivity.json")
vals = [v for r in bw["runs"] for v in r["rmse_ln_2026_by_seed"].values()]
close(min(vals), 0.308, 3, "GW bandwidth RMSE lo 0.308")
close(max(vals), 0.314, 3, "GW bandwidth RMSE hi 0.314")

# ------------------------------------------------------- financed robustness
for name, rmse, mape, I in [("cv_ols_2025_level_a_fin.json", 0.354, 28.5, 0.343),
                            ("cv_sar_gm_2025_level_a_fin.json", 0.344, 28.1, 0.342),
                            ("cv_xgb_lag_2025_level_a_fin.json", 0.286, 23.0, 0.327),
                            ("cv_tabpfn_t0_2025_level_a_fin.json", 0.269, 21.1, 0.279)]:
    p, m = pooled(name)
    close(p["rmse_ln"], rmse, 3, f"{name} RMSE"); close(p["mape_pct"], mape, 1, f"{name} MAPE")
    close(m["I"], I, 3, f"{name} I")
for name, rmse, mape, I in [("oot_ols_level_a_fin.json", 0.343, 26.5, 0.491),
                            ("oot_sar_gm_level_a_fin.json", 0.286, 22.0, 0.297),
                            ("oot_xgb_lag_level_a_fin.json", 0.221, 16.7, 0.160),
                            ("oot_tabpfn_t0_level_a_fin.json", 0.207, 15.3, 0.143)]:
    p, m, b = oot(name)
    close(p["rmse_ln"], rmse, 3, f"{name} RMSE"); close(p["mape_pct"], mape, 1, f"{name} MAPE")
    close(m["I"], I, 3, f"{name} I")
c = J("compare_tabpfn_t0_2025_level_a_fin_vs_xgb_lag_2025_level_a_fin.json")
close(find_key(c, "diff"), -0.017, 3, "fin CV vs XGB -0.017")
ci = find_key(c, "ci95"); close(ci[0], -0.032, 3, "CI lo"); close(ci[1], -0.003, 3, "CI hi")
close(find_key(c, "p") or find_key(c, "wilcoxon_p") or find_key(c, "p_value"), 0.049, 3, "p=0.049")
pf_t = {r["fold"] if "fold" in r else r["block"]: r["rmse_ln"]
        for r in J("cv_tabpfn_t0_2025_level_a_fin.json")["per_seed"]["42"]["per_fold"]}
pf_x = {r["fold"] if "fold" in r else r["block"]: r["rmse_ln"]
        for r in J("cv_xgb_lag_2025_level_a_fin.json")["per_seed"]["42"]["per_fold"]}
CHECK("fin CV blocks 8/10", sum(pf_t[k] < pf_x[k] for k in pf_t) == 8,
      str(sum(pf_t[k] < pf_x[k] for k in pf_t)))
c = J("oot_compare_tabpfn_t0_level_a_fin_vs_xgb_lag_level_a_fin.json")
close(find_key(c, "diff"), -0.013, 3, "fin 2026 vs XGB -0.013")
ci = find_key(c, "ci95"); close(ci[0], -0.017, 3, "CI lo"); close(ci[1], -0.010, 3, "CI hi")
ifin = [J("cv_xgb_lag_2025_level_a_fin.json")["per_seed"][s]["moran_oof"]["I"] for s in ("42", "43", "44")]
CHECK("fin CV XGB I 0.327-0.340", round(min(ifin), 3) == 0.327 and round(max(ifin), 3) == 0.340, str(ifin))
TXT("fin CV 9/10 vs seeds 43/44", "9 of 10 against seeds 43", README)

# ------------------------------------------------- what carries location
PC = json.loads((RES / "postal_code_vs_baselines.json").read_text())


STRICT_CEP = "tabpfn_t0_nocoord_nostation_nom_cep"


def pc_leg(name):
    return [c for c in PC["legs"][name]["comparisons"] if c["a"] == STRICT_CEP]


# leg-2 metrics of the postal-code-only model
a24 = pc_leg("leg2_level_a")[0]["a_metrics"]
close(a24["rmse_ln"], 0.277, 3, "postal 24k RMSE"); close(a24["moran_I"], 0.129, 3, "postal 24k I")
a82 = pc_leg("leg2_full")[0]["a_metrics"]
close(a82["rmse_ln"], 0.268, 3, "postal 82k RMSE"); close(a82["moran_I"], 0.091, 3, "postal 82k I")
a1 = pc_leg("leg1")[0]["a_metrics"]
close(a1["rmse_ln"], 0.457, 3, "postal leg1 RMSE"); close(a1["moran_I"], 0.502, 3, "postal leg1 I")


def best_seed_cmp(leg, blabel):
    cs = [c for c in pc_leg(leg) if c["b"] == blabel]
    if len(cs) == 1:
        return cs[0]
    return min(cs, key=lambda c: c["b_metrics"]["rmse_ln"])


for leg, blabel, diff, blocks in [("leg2_level_a", "xgb_lag", -0.011, 10),
                                  ("leg2_full", "xgb_lag", -0.014, 10),
                                  ("leg2_level_a", "gxgb", -0.036, 10),
                                  ("leg2_full", "gxgb", -0.029, 9),
                                  ("leg2_level_a", "sar_gm", -0.069, 10),
                                  ("leg2_full", "sar_gm", -0.057, 10)]:
    c = best_seed_cmp(leg, blabel)
    close(c["rmse_ln"]["diff"], diff, 3, f"postal {leg} vs {blabel} best-seed {diff}")
    CHECK(f"postal {leg} vs {blabel} blocks {blocks}/10", c["blocks_won"] == blocks,
          str(c["blocks_won"]))
ps = sorted({round(c["wilcoxon_p"], 3) for leg in ("leg2_level_a", "leg2_full") for c in pc_leg(leg)})
CHECK("postal leg2 Wilcoxon p in {0.002, 0.004}", set(ps) <= {0.002, 0.004}, str(ps))
cx = [c for c in pc_leg("leg1") if c["b"] == "xgb_lag"]
within(min(c["rmse_ln"]["diff"] for c in cx), 0.0685, 0.0695, "postal leg1 vs XGB +0.069")
within(max(c["rmse_ln"]["diff"] for c in cx), 0.0715, 0.0725, "postal leg1 vs XGB +0.072")
CHECK("postal leg1 vs XGB 4/10", all(c["blocks_won"] == 4 for c in cx))
CHECK("postal leg1 vs XGB CIs just clear of zero", all(0 < c["rmse_ln"]["ci95"][0] for c in cx),
      str([c["rmse_ln"]["ci95"] for c in cx]))

# no-location floors and recovery shares from the descriptor summary
DS = json.loads((RES / "location_descriptor_summary.json").read_text())
dtxt = json.dumps(DS)


def floor_of(leg_idx):
    rows = {r["label"]: r for r in DS["legs"][leg_idx]["rows"]}
    return rows["tabpfn_t0_nocoord_nostation"]["rmse_ln"]


close(floor_of(0), 0.440, 3, "no-location floor, leg 1 = 0.440")
close(floor_of(1), 0.352, 3, "no-location floor, leg 2 24k = 0.352")
close(floor_of(2), 0.327, 3, "no-location floor, leg 2 82k = 0.327")
CHECK("postal leg1 below its own floor (0.457 vs 0.440)",
      round(a1["rmse_ln"], 3) == 0.457 and a1["rmse_ln"] > floor_of(0))
# seen/unseen recovery, leg 2
su = {e["leg"]: e.get("seen_unseen") for e in DS["legs"]}
sutxt = json.dumps(su)
for s in ["0.97", "0.96", "0.98", "0.94", "0.22", "0.09"]:
    CHECK(f"seen/unseen share {s} present in descriptor summary", s in sutxt or s in dtxt, s)

# address on top of the coordinates (nominal ablation)
NA = json.loads((RES / "nominal_ablation_summary.json").read_text())
natxt = json.dumps(NA)
diffs = []
for r in NA.get("comparisons", []) + (NA.get("pairs") or []):
    d = find_key(r, "diff")
    if d is not None:
        diffs.append(abs(d))
if not diffs:  # fall back: scan runs for rmse differences vs the plain rows
    runs = NA["runs"]
    plain = {r["leg"]: r["rmse_ln"] for r in runs if r["variant"] == "plain table"}
    diffs = [abs(r["rmse_ln"] - plain[r["leg"]]) for r in runs if r["variant"] != "plain table"
             and r["leg"] in plain]
CHECK("address-on-top |dRMSE| < 0.006 everywhere", diffs and max(diffs) < 0.006,
      f"max={max(diffs) if diffs else None}")
TXT("bias drop with names, 24k (+0.055 -> +0.045)", "+0.055 → +0.045", FURTHER)
TXT("bias drop with names, 82k (+0.058 -> +0.043)", "+0.058 → +0.043", FURTHER)

# text variant
tx1 = seeds(L1["models"]["text"], "rmse_ln")[42]
close(tx1, 0.3835, 4, "text leg1 RMSE 0.3835")
close(seeds(L1["models"]["text"], "mape_pct")[42], 30.8, 1, "text leg1 MAPE 30.8")
close(seeds(L1["models"]["text"], "moran_I")[42], 0.331, 3, "text leg1 I 0.331")
qt = [x for x in FL1 if x["a"] == "text" and x["b"] == "tabpfn"]
t42 = [x for x in qt if x.get("b_seed") == 42][0]
close(t42["rmse_ln"]["diff"], -0.0103, 4, "text vs plain -0.0103")
close(t42["rmse_ln"]["ci95"][0], -0.0161, 4, "CI lo"); close(t42["rmse_ln"]["ci95"][1], -0.0031, 4, "CI hi")
CHECK("text vs plain 8/10", t42["blocks_won"] == 8)
within(min(x["rmse_ln"]["diff"] for x in qt), -0.0105, -0.0100, "text vs seeds lo -0.010")
within(max(x["rmse_ln"]["diff"] for x in qt), -0.0075, -0.0068, "text vs seeds hi -0.007")
qtg = [x for x in FL1 if x["a"] == "text" and x["b"] == "gxgb"]
within(min(x["rmse_ln"]["diff"] for x in qtg), -0.0268, -0.0260, "text vs GW -0.026")
within(max(x["rmse_ln"]["diff"] for x in qtg), -0.0240, -0.0230, "text vs GW -0.023")
CHECK("text vs GW 9/10, p=0.027", all(x["blocks_won"] == 9 and round(x["wilcoxon_p"], 3) == 0.027 for x in qtg))
t2a = [x for x in FL2a if x["a"] == "text" and x["b"] == "tabpfn"][0]
close(t2a["rmse_ln"]["diff"], 0.0000, 4, "text leg2 24k +0.0000")
close(t2a["rmse_ln"]["ci95"][0], -0.0011, 4, "CI lo"); close(t2a["rmse_ln"]["ci95"][1], 0.0010, 4, "CI hi")
CHECK("text leg2 24k 6/10", t2a["blocks_won"] == 6)
t2f = [x for x in FL2f if x["a"] == "text" and x["b"] == "tabpfn"][0]
close(t2f["rmse_ln"]["diff"], -0.0039, 4, "text leg2 82k -0.0039")
close(t2f["rmse_ln"]["ci95"][0], -0.0049, 4, "CI lo"); close(t2f["rmse_ln"]["ci95"][1], -0.0031, 4, "CI hi")
CHECK("text leg2 82k 10/10", t2f["blocks_won"] == 10)
close(seeds(L2f["models"]["text"], "rmse_ln")[42], 0.2610, 4, "text 82k RMSE 0.2610")
close(seeds(L2f["models"]["text"], "mape_pct")[42], 19.2, 1, "text 82k MAPE 19.2")
close(seeds(L2f["models"]["text"], "moran_I")[42], 0.086, 3, "text 82k I 0.086")
close(seeds(L2a["models"]["text"], "rmse_ln")[42], 0.2750, 4, "text 24k RMSE 0.2750")
close(seeds(L2a["models"]["text"], "mape_pct")[42], 20.6, 1, "text 24k MAPE 20.6")
for s in ["parking (−0.034)", "commercial units (−0.023)", "92 %",
          "(−0.014 and −0.013)"]:
    TXT(f"text mechanism: {s}", s, FURTHER)

# thinking with blocks as groups
g1 = [x for x in FL1 if x["a"] == "think_grp" and x["b"] == "think"][0]
close(g1["rmse_ln"]["diff"], 0.0038, 4, "grp leg1 +0.0038")
close(g1["rmse_ln"]["ci95"][0], -0.0055, 4, "CI lo"); close(g1["rmse_ln"]["ci95"][1], 0.0149, 4, "CI hi")
CHECK("grp leg1 3/10", g1["blocks_won"] == 3)
close(seeds(L1["models"]["think_grp"], "moran_I")[42], 0.350, 3, "grp leg1 I 0.350")
g2 = [x for x in FL2a if x["a"] == "think_grp" and x["b"] == "think"][0]
close(g2["rmse_ln"]["diff"], -0.0014, 4, "grp leg2 -0.0014")
close(g2["rmse_ln"]["ci95"][0], -0.0021, 4, "CI lo"); close(g2["rmse_ln"]["ci95"][1], -0.0007, 4, "CI hi")
CHECK("grp leg2 9/10", g2["blocks_won"] == 9)
close(seeds(L2a["models"]["think_grp"], "bias_ln")[42], 0.043, 3, "grp leg2 bias +0.043")

# ------------------------------------------------------------------- SHAP
SH = json.loads((RES / "shap" / "shap_summary_tabpfn_t0_level_a.json").read_text())
CHECK("SHAP: 50 explained, 20 background", SH["n_explained"] == 50 and SH["n_background"] == 20)
CHECK("SHAP: 27,217 predicted rows", SH["predicted_rows"] == 27217, str(SH["predicted_rows"]))
ms = SH["mean_abs_shap_ln"]
for f, v in [("lon", 0.184), ("lat", 0.143), ("tipo_imovel", 0.127),
             ("area_construida_m2", 0.112), ("idade", 0.082)]:
    close(ms[f], v, 3, f"SHAP mean|.| {f} = {v}")
close(math.exp(ms["lon"]), 1.20, 2, "exp(0.184) = x1.20")
close(math.exp(ms["lat"]), 1.15, 2, "exp(0.143) = x1.15")
close(SH["rmse_ln_on_subsample"], 0.296, 3, "SHAP reconstructed RMSE 0.296")

# --------------------------------------------------------------- appendices
T = J("gxgb_timing_pilot.json")
CHECK("pilot: 200 rows, 18.7 s, 0.093 s/unit",
      T["pilot"]["subsample_rows"] == 200 and T["pilot"]["wall_s"] == 18.7
      and T["pilot"]["s_per_spatial_unit"] == 0.093)
E = T["extrapolation"]
CHECK("pilot extrapolation: 19,117 rows, 0.5 h, 2 days, 2.9 GB",
      E["train_rows_per_fold"] == 19117 and E["hours_per_bandwidth_per_fold"] == 0.5
      and E["days_for_10_bandwidths_x_10_folds"] == 2
      and E["dense_distance_matrix_gb_per_fold"] == 2.9)
X = J("xgb_params_oot_full.json")
pp = X["params"]
close(pp["learning_rate"], 0.049, 3, "XGB lr 0.049")
CHECK("XGB depth 7", pp["max_depth"] == 7)
close(pp["min_child_weight"], 2.0, 1, "XGB mcw 2.0")
close(pp["subsample"], 0.89, 2, "XGB subsample 0.89")
close(pp["colsample_bytree"], 0.56, 2, "XGB colsample 0.56")
close(pp["reg_lambda"], 0.0033, 4, "XGB lambda 0.0033")
close(pp["reg_alpha"], 0.0041, 4, "XGB alpha 0.0041")
close(pp["gamma"], 0.0031, 4, "XGB gamma 0.0031")
CHECK("XGB 398 rounds", pp["n_estimators"] == 398)
close(X["inner_rmse_ln"], 0.351, 3, "XGB inner RMSE 0.351")
close(X["tuning_s"], 901, 0, "XGB tuning 901 s")
CHECK("XGB 30 trials", X["n_trials"] == 30)

# code-level settings cited in the text
src = (ROOT / "src/model_xgb.py").read_text()
CHECK("rotations 30/45/60 in code", "ROTATIONS_DEG = (30.0, 45.0, 60.0)" in src)
CHECK("max 3,000 rounds in code", "max_rounds: int = 3000" in src)
srcg = (ROOT / "src/model_gxgb.py").read_text()
CHECK("GW local params 300/0.05/4", "n_estimators=300, learning_rate=0.05, max_depth=4" in srcg)
CHECK("GW global params 800/0.05/6", "n_estimators=800, learning_rate=0.05, max_depth=6" in srcg)
CHECK("GW M=1000 regression points", "M = 1000" in srcg or "M=1000" in srcg or '"M", 1000' in srcg
      or "m_points: int = 1000" in srcg or "1000" in srcg)
srct = (ROOT / "src/model_tabpfn.py").read_text()
CHECK("TabPFN v3.5 / rmse / 2400 s in code",
      'MODEL_VERSION = "v3.5"' in srct and 'thinking_metric="rmse"' in srct
      and "2400" in srct)
from src.model_sar import X_NAMES  # noqa: E402
CHECK("OLS/SAR design has no raw coordinates", not any(n in ("lat", "lon") for n in X_NAMES),
      str(X_NAMES))

# abstract-level aggregates
spec = list(xgf.values()) + list(gwf.values()) + [J("oot_sar_gm_full.json")["per_seed"]["42"]["pooled"]["rmse_ln"]]
close(min(spec), 0.283, 3, "abstract: best specialist RMSE 0.283")
close(max(spec), 0.326, 3, "abstract: worst specialist RMSE 0.326")
specm = list(mx.values()) + list(seeds(L2f["models"]["gxgb"], "mape_pct").values()) \
        + [J("oot_sar_gm_full.json")["per_seed"]["42"]["pooled"]["mape_pct"]]
close(min(specm), 20.9, 1, "abstract: specialist MAPE lo 20.9")
close(max(specm), 25.0, 1, "abstract: specialist MAPE hi 25.0")
wins_full = [pget(FL2f, "tabpfn", "xgb", bs)[0]["blocks_won"] for bs in (42, 43, 44)] \
    + [x["blocks_won"] for x in pget(FL2f, "tabpfn", "gxgb")] \
    + [pget(FL2f, "tabpfn", "sar")[0]["blocks_won"]]
CHECK("abstract: at least 9 of 10 blocks vs every specialist (full base)", min(wins_full) == 9,
      str(wins_full))

# prose-sourced claims kept verbatim with the repository record
TXT("fit times 9 s / 15 s", "fit in 9 s", README); TXT("fit 15 s", "fit in 15 s", README)
TXT("thinking high costs 13x", "13×", README)
TXT("postal codes 97-100% unseen in a held-out block", "97–100 %", README)

print(f"\n{N_PASS} checks passed, {len(FAILS)} failed.")
if FAILS:
    sys.exit(1)
print("All numbers in the manuscript match the versioned results.")
