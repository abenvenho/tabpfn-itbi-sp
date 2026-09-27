#!/usr/bin/env bash
# TabPFN-3.5 on Level A — run on a machine with API access (the author's Mac).
#
# Usage, from the repository root:
#   export TABPFN_TOKEN="<your API key>"        # platform.priorlabs.ai/account/api-keys
#   bash scripts/10_tabpfn_level_a.sh check      # token, models, allowance, cost estimates (no quota)
#   bash scripts/10_tabpfn_level_a.sh t0         # zero-shot TabPFN-3.5 (main run, seed 42)
#   bash scripts/10_tabpfn_level_a.sh t0-think   # thinking mode, medium effort (main run)
#   bash scripts/10_tabpfn_level_a.sh t0-seeds   # extra seeds 43 44 for the zero-shot run (CI)
#   bash scripts/10_tabpfn_level_a.sh t0-think-grp  # robustness: block as thinking group_col
#
# Uses its own virtual environment (.venv-tabpfn, from requirements-tabpfn.txt).
# Every API response is cached under results/tabpfn_cache/<label>/, so a step
# can be re-run (or interrupted and resumed) without paying twice.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv-tabpfn ]; then
  python3 -m venv .venv-tabpfn
fi
# shellcheck disable=SC1091
source .venv-tabpfn/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements-tabpfn.txt

if [ -z "${TABPFN_TOKEN:-}" ]; then
  echo "TABPFN_TOKEN is not set. Run: export TABPFN_TOKEN=\"<key>\"" >&2
  exit 1
fi

Q="--quantiles 0.1 0.25 0.5 0.75 0.9"      # 80% and 50% predictive intervals, cached for later
BASE="--compare-with sar_gm_2025_level_a xgb_lag_2025_level_a ols_2025_level_a"

case "${1:-}" in
  check)
    python scripts/00_check_api.py
    python -m src.model_tabpfn --dry-run
    python -m src.model_tabpfn --dry-run --thinking medium
    python -m src.model_tabpfn --dry-run --thinking high ;;
  t0)
    python -m src.model_tabpfn --thinking off $Q $BASE 2>&1 | tee -a results/tabpfn_t0_run.log ;;
  t0-think)
    python -m src.model_tabpfn --thinking medium $Q $BASE 2>&1 | tee -a results/tabpfn_t0_think_run.log ;;
  t0-think-high)
    python -m src.model_tabpfn --thinking high $Q $BASE 2>&1 | tee -a results/tabpfn_t0_think_high_run.log ;;
  t0-seeds)
    python -m src.model_tabpfn --thinking off --seeds 42 43 44 $Q $BASE 2>&1 | tee -a results/tabpfn_t0_run.log ;;
  t0-think-grp)
    python -m src.model_tabpfn --thinking medium --group-col $Q $BASE 2>&1 | tee -a results/tabpfn_t0_think_grp_run.log ;;
  *)
    echo "usage: $0 {check|t0|t0-think|t0-think-high|t0-seeds|t0-think-grp}" >&2; exit 2 ;;
esac
