#!/usr/bin/env bash
# Robustness check: financed transactions only (7,211 of the 24,000 Level A
# rows; 20,052 of the 47,810 rows of 2026). The under-declaration filter removes
# cash deals declared at the reference value (VVR); financed deals carry a bank
# appraisal and are the cleanest price signal in the base. Both legs of the
# protocol are re-run on this subset with the same folds, tests and metrics.
#
# Usage, from the repository root:
#   bash scripts/40_financed_robustness.sh baselines   # OLS, SAR, XGB+lag (no API; ~1 h for XGB tuning)
#   bash scripts/40_financed_robustness.sh tabpfn      # TabPFN-3.5 zero-shot, both legs (API; TABPFN_TOKEN exported)
#   bash scripts/40_financed_robustness.sh compare     # paired tests on the 2026 leg (no API)
#
# The baselines run in the main environment (requirements.txt); the TabPFN
# step in .venv-tabpfn (requirements-tabpfn.txt). Responses are cached under
# results/tabpfn_cache/<label>/, so the TabPFN step can be resumed.
set -euo pipefail
cd "$(dirname "$0")/.."

case "${1:-}" in
  baselines)
    python -m src.model_sar --estimator ols --financed-only
    python -m src.model_sar --estimator gm --financed-only
    python -m src.model_xgb --financed-only --seeds 42 43 44
    python -m src.out_of_time --model ols --train level_a --financed-only
    python -m src.out_of_time --model sar --train level_a --financed-only
    python -m src.out_of_time --model xgb --train level_a --financed-only --seeds 42 43 44 ;;
  tabpfn)
    if [ ! -d .venv-tabpfn ]; then python3 -m venv .venv-tabpfn; fi
    # shellcheck disable=SC1091
    source .venv-tabpfn/bin/activate
    pip install -q --upgrade pip >/dev/null
    pip install -q -r requirements-tabpfn.txt
    if [ -z "${TABPFN_TOKEN:-}" ]; then
      echo "TABPFN_TOKEN is not set. Run: export TABPFN_TOKEN=\"<key>\"" >&2; exit 1
    fi
    python -m src.model_tabpfn --thinking off --financed-only \
      --compare-with sar_gm_2025_level_a_fin xgb_lag_2025_level_a_fin ols_2025_level_a_fin \
      2>&1 | tee -a results/tabpfn_fin_run.log
    python -m src.out_of_time --model tabpfn --train level_a --thinking off --financed-only \
      2>&1 | tee -a results/tabpfn_fin_run.log ;;
  compare)
    for b in sar_gm_level_a_fin xgb_lag_level_a_fin ols_level_a_fin; do
      python -m src.out_of_time --financed-only --compare tabpfn_t0_level_a_fin $b
    done ;;
  *)
    echo "usage: $0 {baselines|tabpfn|compare}" >&2; exit 2 ;;
esac
