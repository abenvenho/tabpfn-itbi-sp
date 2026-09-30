#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
60_app_live_check.py — end-to-end check of the app's live tab with a real token
===============================================================================
Runs ``app/app.py`` headless (Streamlit's ``AppTest``: the same script, the
same widgets, no browser), fills in the form of tab 4 "Appraise a property"
and presses **Estimate** against the Prior Labs API, for each training base
the tab offers. It reports, per case: the estimate, whether the model came
from the cached record of the study or from a fresh fit, and how long it
took. One case replays a 2026 transaction of the test set, so the live
estimate can be read against the one cached in ``results/``.

Usage, from the repository root (needs the app environment: streamlit,
plotly and tabpfn-client, e.g. ``pip install -r app/requirements.txt``):

    export TABPFN_TOKEN="<your API key>"
    python scripts/60_app_live_check.py            # all three bases
    python scripts/60_app_live_check.py --bases level_a

Cost: a handful of one-row predictions, plus one fit per base whose cached
record the server no longer holds for this account. Exit code 0 when every
case returned an estimate, 1 otherwise. Nothing is written to ``results/``.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASES = ["level_a", "full", "level_a_fin"]
REPLAY_SQL_MONTH = 13          # a January-2026 transaction: one month past the training window


def replay_case() -> dict:
    """A 2026 apartment of the test set and the study's cached estimate for it (Level A fit)."""
    test = pd.read_csv(ROOT / "data" / "itbi_sp_2026_test.csv", dtype={"sql": str})
    pred = pd.read_csv(ROOT / "results" / "oot_pred_tabpfn_t0_level_a_seed42.csv", dtype={"sql": str})
    assert (test["sql"].values == pred["sql"].values).all()
    ok = ((test["mes_idx"] == REPLAY_SQL_MONTH) & (test["tipo_imovel"] == "apartment")
          & test["idade"].notna() & test["dist_estacao_m"].notna()
          & test["lat"].between(-23.90, -23.35) & test["lon"].between(-46.85, -46.35)
          & test["area_construida_m2"].between(10, 20000))        # the form's input ranges
    i = int(np.flatnonzero(ok.to_numpy())[0])
    r = test.iloc[i]
    return {"name": f"replay of 2026 transaction SQL {r['sql']}", "base": "level_a",
            "tipo": "apartment", "area_c": float(r["area_construida_m2"]),
            "area_t": float(0.0 if pd.isna(r["area_terreno_m2"]) else r["area_terreno_m2"]),
            "idade": float(r["idade"]), "padrao": int(r["padrao_nivel"]),
            "dist": float(r["dist_estacao_m"]), "lat": float(r["lat"]), "lon": float(r["lon"]),
            "cached_vu": float(np.exp(pred["yhat_ln"].iloc[i])), "declared_vu": float(np.exp(r["ln_vu"]))}


def widget(collection, label_start: str):
    hits = [w for w in collection if str(w.label).startswith(label_start)]
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} widgets labelled {label_start!r}")
    return hits[0]


def run_case(at, case: dict) -> dict:
    widget(at.radio, "Fit TabPFN-3.5 on").set_value(case["base"])
    at.run()
    if "tipo" in case:
        widget(at.selectbox, "Property type").set_value(case["tipo"])
    for key, label in [("area_c", "Built area"), ("area_t", "Lot area"), ("idade", "Age"),
                       ("dist", "Distance to nearest"), ("lat", "Latitude"), ("lon", "Longitude")]:
        if key in case:
            widget(at.number_input, label).set_value(case[key])
    if "padrao" in case:
        widget(at.selectbox, "Finish grade").set_value(case["padrao"])
    t0 = time.time()
    widget(at.button, "Estimate").click()
    at.run()
    out = {"seconds": time.time() - t0, "ok": False}
    if at.exception:
        out["problem"] = "exception: " + " | ".join(str(e.value)[:300] for e in at.exception)
        return out
    api_err = [e.value for e in at.error if "API call failed" in str(e.value)]
    if api_err:
        out["problem"] = api_err[0][:400]
        return out
    unit = [m for m in at.metric if m.label == "Unit value"]
    if not unit:
        out["problem"] = "no estimate on the page after pressing Estimate"
        return out
    out["vu"] = float(re.sub(r"[^\d.]", "", unit[0].value.replace(",", "").split("/")[0]))
    how = [c.value for c in at.caption if str(c.value).startswith("Model: TabPFN-3.5")]
    out["how"] = "cached record" if how and "cached record" in how[0] else "fresh fit"
    out["ok"] = np.isfinite(out["vu"]) and 300 < out["vu"] < 200_000     # R$/m², sanity range
    if not out["ok"]:
        out["problem"] = f"estimate out of the plausible range: {out['vu']}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Live check of tab 4 of the Streamlit app")
    ap.add_argument("--bases", nargs="+", default=BASES, choices=BASES)
    args = ap.parse_args()
    if not os.environ.get("TABPFN_TOKEN"):
        sys.exit("TABPFN_TOKEN is not set (export TABPFN_TOKEN=...)")
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        sys.exit("streamlit is not installed here: pip install -r app/requirements.txt")

    print("Loading the app (47,810 transactions of 2026 and the cached predictions)…", flush=True)
    at = AppTest.from_file(str(ROOT / "app" / "app.py"), default_timeout=1800)
    at.run()
    if at.exception:
        print("The app failed before tab 4:", [str(e.value)[:300] for e in at.exception])
        return 1
    if any("switched off" in str(w.value) for w in at.warning):
        print("Tab 4 is switched off: the app did not see the token.")
        return 1
    if any("not installed" in str(e.value) for e in at.error):
        print("tabpfn-client is not installed in this environment.")
        return 1

    default = "default form (70 m² apartment, age 15, grade 2, Av. Paulista)"
    cases = [{"name": default, "base": b} for b in args.bases]
    if "level_a" in args.bases:
        cases += [{"name": "same, as a house", "base": "level_a", "tipo": "house"},
                  {"name": "same, as commercial", "base": "level_a", "tipo": "commercial"},
                  replay_case()]
    failures, results = 0, []
    for c in cases:
        print(f"- {c['base']:<12} {c['name']} …", flush=True)
        r = run_case(at, c)
        results.append((c, r))
        if r["ok"]:
            line = f"    R$ {r['vu']:,.0f}/m²  ·  {r['how']}  ·  {r['seconds']:.0f}s"
            if "cached_vu" in c:
                line += (f"\n    study's cached estimate for that row: R$ {c['cached_vu']:,.0f}/m² "
                         f"(live/cached = {r['vu'] / c['cached_vu']:.3f}; the form prices at Dec 2025, "
                         f"the study at the month of the deal) · declared: R$ {c['declared_vu']:,.0f}/m²")
            print(line, flush=True)
        else:
            failures += 1
            print(f"    FAILED after {r['seconds']:.0f}s: {r['problem']}", flush=True)

    by_type = {c.get("tipo", "apartment"): r["vu"] for c, r in results
               if c["base"] == "level_a" and r["ok"] and "cached_vu" not in c}
    if len(by_type) == 3 and len({round(v) for v in by_type.values()}) == 1:
        failures += 1
        print("FAILED: the three property types returned the same estimate — the type is not reaching the model.")
    print()
    print("Tab 4 works with this token." if failures == 0 else f"{failures} problem(s) — see above.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
