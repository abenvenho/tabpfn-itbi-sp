# Runbook: the spatial-lag variant (`t0-lag`)

Instructions for executing the `--spatial-lag` variant of the TabPFN-3.5
runs. Written for a Claude cloud session whose environment allows network
access to `api.priorlabs.ai` (the author starts it from the Code tab and
attaches the API-token file); a human with the repository and a token can
follow the same steps in a terminal.

**What this run is.** TabPFN-3.5 on the plain T0 table **plus one column**:
the fold-internal k-NN-8 spatial lag of `ln_vu` — exactly the engineered
spatial feature the `xgb_lag` baseline receives (same construction, same
jitter rule, same k; built by `protocol.knn_spatial_lag`). It makes the
design symmetric with XGB+lag and answers whether the residual spatial
autocorrelation of the main TabPFN runs is an information gap the lag
closes. The **pre-registered expectations** are in the module docstring of
`src/model_tabpfn.py`, section *"Spatial-lag variant"* — read them before
running, report against them after.

## Steps

1. **Token.** The author attaches a file containing the API token. Read it,
   strip whitespace, and export it as `TABPFN_TOKEN` in each shell that runs
   a step. Never print the token, never write it into the repository
   (`.gitignore` already excludes `.env` and `TOKEN*`), never include it in
   a commit, a log file, or a message.

2. **Readiness and cost (no quota spent).**

   ```bash
   bash scripts/80_tabpfn_spatial_lag.sh check
   ```

   Guardrails, in credits as reported by `get_api_usage` / `estimate_cost`:
   the three runs below are expected to land in the same range as the
   September `t0` runs (order of 10⁵–10⁶ credits in total, against a
   monthly allowance of 20 M renewed on the 1st). **Stop and report instead
   of running** if the summed estimate exceeds 3 M credits, or if the
   remaining allowance is less than twice the summed estimate.

3. **The three runs**, in this order (each is resumable: every API response
   is cached under `results/tabpfn_cache/`, so an interrupted step re-run
   skips what is already paid for):

   ```bash
   bash scripts/80_tabpfn_spatial_lag.sh cv        # leg 1: Level A CV, seed 42
   bash scripts/80_tabpfn_spatial_lag.sh oot       # leg 2: 24k -> 2026
   bash scripts/80_tabpfn_spatial_lag.sh oot-full  # leg 2: 82k -> 2026
   bash scripts/80_tabpfn_spatial_lag.sh compare   # paired tests, leg 2 (no API)
   ```

   Labels produced: `tabpfn_t0_lag_2025_level_a` (CV), `tabpfn_t0_lag_level_a`
   and `tabpfn_t0_lag_full` (out-of-time). The `cv` step already runs the
   leg-1 paired comparisons against `tabpfn_t0_2025_level_a` and
   `xgb_lag_2025_level_a`.

4. **Commit the results** on a branch, never on `main` directly:

   ```bash
   git checkout -b t0-lag-results
   git add results/cv_tabpfn_t0_lag_* results/oof_tabpfn_t0_lag_* \
           results/compare_tabpfn_t0_lag_* results/oot_tabpfn_t0_lag_* \
           results/oot_pred_tabpfn_t0_lag_* results/oot_compare_tabpfn_t0_lag_* \
           "results/tabpfn_cache/tabpfn_t0_lag_2025_level_a" \
           "results/tabpfn_cache/oot_tabpfn_t0_lag_level_a" \
           "results/tabpfn_cache/oot_tabpfn_t0_lag_full"
   git commit -m "Results: spatial-lag variant (t0-lag), both legs"
   git push -u origin t0-lag-results
   ```

   Before committing, confirm `git status` shows **no** token file and no
   `.env`. Do not modify source files; if a run fails for a reason that
   seems to need a code change, stop, commit whatever results are complete,
   and report the error instead.

5. **Report**, next to the expectations of the module docstring: pooled
   RMSE_ln / MAPE and Moran's I per leg, and the paired tests (Wilcoxon +
   block bootstrap) against `tabpfn_t0` and `xgb_lag`. Reference values
   (seed 42) for the comparison:

   | run | RMSE_ln | MAPE % | Moran I |
   |---|---|---|---|
   | CV Level A — `tabpfn_t0` | 0.3938 | 32.2 | 0.348 |
   | CV Level A — `xgb_lag` | 0.3846 | 32.3 | 0.309 |
   | 2026, 24k — `tabpfn_t0` | 0.2750 | 20.4 | 0.122 |
   | 2026, 24k — `xgb_lag` | 0.2884 | 21.8 | 0.140 |
   | 2026, 82k — `tabpfn_t0` | 0.2649 | 19.5 | 0.092 |
   | 2026, 82k — `xgb_lag` | 0.2940 | 21.5 | 0.107 |
