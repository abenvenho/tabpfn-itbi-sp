#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00_check_api.py — TabPFN API readiness check (runs on the author's machine)
==========================================================================
Confirms, before any credit is spent on real experiments, that:

1. the token in the ``TABPFN_TOKEN`` environment variable is accepted;
2. the account can see the TabPFN-3.5 models;
3. the current usage / credit allowance (``get_api_usage``);
4. what the Level A experiments would cost (``estimate_cost`` sends only
   row/column counts, nothing is uploaded and no quota is consumed);
5. one tiny real round-trip (200 training rows, 20 test rows) with
   ``v3.5`` — the only step that touches the quota, negligibly.

Usage (macOS Terminal, from the repository root):

    pip install --upgrade tabpfn-client
    export TABPFN_TOKEN="<your API key from platform.priorlabs.ai/account/api-keys>"
    python scripts/00_check_api.py            # add --no-roundtrip to skip step 5

The API is reached from the author's own machine because the development
sandbox cannot open connections to api.priorlabs.ai.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

LEVEL_A_TRAIN, LEVEL_A_TEST, N_FEATURES = 21_600, 2_400, 15   # one CV fold


def main() -> None:
    ap = argparse.ArgumentParser(description="TabPFN API readiness check")
    ap.add_argument("--no-roundtrip", action="store_true",
                    help="skip the tiny real fit/predict (no quota touched)")
    args = ap.parse_args()

    if not os.environ.get("TABPFN_TOKEN"):
        sys.exit("TABPFN_TOKEN is not set. Create a key at "
                 "https://platform.priorlabs.ai/account/api-keys and run\n"
                 "  export TABPFN_TOKEN=\"<key>\"")

    import tabpfn_client
    from tabpfn_client import TabPFNRegressor, estimate_cost, get_api_usage

    print(f"tabpfn-client {tabpfn_client.__version__}")
    tabpfn_client.init()                                    # validates the token
    print("1. token accepted")

    models = TabPFNRegressor.list_available_models()
    v35 = [m for m in models if m.startswith("v3.5")]
    print(f"2. TabPFN-3.5 models visible: {v35 or 'NONE (check account/terms)'}")

    print("3. usage / allowance:")
    print("   " + str(get_api_usage()).replace("\n", "\n   "))

    print("4. cost estimates for one Level A fold "
          f"({LEVEL_A_TRAIN:,} train x {LEVEL_A_TEST:,} test x {N_FEATURES} features):")
    Xtr = np.zeros((LEVEL_A_TRAIN, N_FEATURES))
    Xte = np.zeros((LEVEL_A_TEST, N_FEATURES))
    for op, kw in [("predict", {}),
                   ("thinking_fit", {"thinking_effort": "medium"}),
                   ("thinking_fit", {"thinking_effort": "high"})]:
        try:
            r = estimate_cost(Xtr, None if op == "thinking_fit" else Xte,
                              model_version="v3.5", operation=op, **kw)
            print(f"   {op:13s} {kw or ''}: estimated_cost={r.estimated_cost} "
                  f"(pricing {r.pricing_version})")
        except Exception as e:                                   # noqa: BLE001
            print(f"   {op:13s} {kw or ''}: estimate unavailable ({e})")

    if args.no_roundtrip:
        print("5. round-trip skipped")
        return
    rng = np.random.default_rng(42)
    X = rng.normal(size=(220, 5))
    y = X[:, 0] * 2 + X[:, 1] + rng.normal(scale=0.3, size=220)
    reg = TabPFNRegressor.create_default_for_version("v3.5")
    t0 = time.time()
    reg.fit(X[:200], y[:200])
    pred = reg.predict(X[200:])
    rmse = float(np.sqrt(np.mean((pred - y[200:]) ** 2)))
    print(f"5. real round-trip with v3.5 OK: 200 train / 20 test in "
          f"{time.time() - t0:.1f}s, RMSE={rmse:.3f} (noise sd 0.3)")
    print("\nAll good — the experiment scripts can be run.")


if __name__ == "__main__":
    main()
