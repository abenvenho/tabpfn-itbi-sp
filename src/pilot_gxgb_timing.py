#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pilot_gxgb_timing.py — feasibility pilot for geographically weighted XGBoost
===========================================================================
Times ``geoxgboost.optimize_bw`` (Grekousis 2025) on a random subsample of
one training fold and extrapolates the cost to the real fold size. The
package fits one local XGBoost (with a GridSearchCV) per training row and
per bandwidth candidate over a dense n x n distance matrix, so its cost is
O(n) local fits per candidate and O(n^2) memory.

Result recorded in ``results/gxgb_timing_pilot.json`` (200 rows: 0.093 s per
spatial unit -> ~0.5 h per candidate per fold, ~2 days for 10 x 10, 2.9 GB
distance matrix per fold) and the reason ``src/model_xgb.py`` uses a global
XGBoost with fold-internal spatial features instead.

Run:  python -m src.pilot_gxgb_timing [subsample_rows]   (default 300)
Requires: pip install geoxgboost   (not in requirements.txt on purpose)
"""
import time, sys, io, contextlib
import numpy as np, pandas as pd
from geoxgboost import optimize_bw
from src.protocol import load_base, spatial_folds, jittered_coords
from src.model_sar import design_matrix, X_NAMES

df = load_base("data/itbi_sp_2025_level_a.csv")
folds = spatial_folds(df)
b, tr, te = folds[0]
n_sub = int(sys.argv[1]) if len(sys.argv) > 1 else 300
rng = np.random.default_rng(42)
sub = rng.choice(tr, size=n_sub, replace=False)
d = df.iloc[sub].reset_index(drop=True)
X = pd.DataFrame(design_matrix(d), columns=X_NAMES)
y = pd.DataFrame({"ln_vu": d["ln_vu"].to_numpy()})
C = pd.DataFrame(jittered_coords(d), columns=["x", "y"])
params = dict(n_estimators=100, learning_rate=0.1, max_depth=3, reg_alpha=0.0,
              n_jobs=1, random_state=42, verbosity=0)
t0 = time.time()
with contextlib.redirect_stdout(io.StringIO()):
    bw = optimize_bw(X, y, C, params, bw_min=50, bw_max=50, step=1,
                     Kernel="Adaptive", spatial_weights=True, n_splits=3,
                     path_save="/tmp/")
dt = time.time() - t0
per_unit = dt / n_sub
n_fold = len(tr)
print(f"subsample n={n_sub}: {dt:.1f}s for ONE bandwidth -> {per_unit:.3f} s/unit")
print(f"full fold n={n_fold:,}: {per_unit*n_fold/3600:.1f} h per bandwidth candidate")
print(f"grid of 10 bandwidths x 10 folds: {per_unit*n_fold*10*10/3600/24:.0f} days")
print(f"dense distance matrix per fold: {n_fold**2*8/1e9:.1f} GB")
