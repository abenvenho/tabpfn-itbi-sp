#!/usr/bin/env bash
# TabPFN-3.5 + the XGB baseline's k-NN-8 spatial lag as one extra column —
# the symmetric-design variant (see the module docstring of src/model_tabpfn.py,
# section "Spatial-lag variant", for the pre-registered expectations).
#
# Run on a machine with API access (the author's Mac, or a cloud session whose
# environment allows api.priorlabs.ai). Usage, from the repository root:
#   export TABPFN_TOKEN="<your API key>"        # platform.priorlabs.ai/account/api-keys
#   bash scripts/80_tabpfn_spatial_lag.sh check     # token, allowance, cost estimate (no quota)
#   bash scripts/80_tabpfn_spatial_lag.sh cv        # leg 1: Level A spatial-block CV, seed 42
#   bash scripts/80_tabpfn_spatial_lag.sh oot       # leg 2: Level A (24k) -> 2026
#   bash scripts/80_tabpfn_spatial_lag.sh oot-full  # leg 2: full base (82k) -> 2026
#   bash scripts/80_tabpfn_spatial_lag.sh compare   # paired tests, leg 2 (no API)
#
# No predictive quantiles here: the variant's question is point error and the
# Moran's I of the residuals, and skipping the quantile calls halves the
# predict cost. Every API response is cached under results/tabpfn_cache/, so
# any step can be interrupted and resumed without paying twice.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv-tabpfn ]; then
  python3 -m venv .venv-tabpfn
fi
# shellcheck disable=SC1091
source .venv-tabpfn/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements-tabpfn.txt

if [ "${1:-}" != "compare" ] && [ -z "${TABPFN_TOKEN:-}" ]; then
  echo "TABPFN_TOKEN is not set. Run: export TABPFN_TOKEN=\"<key>\"" >&2
  exit 1
fi

BASE="--compare-with tabpfn_t0_2025_level_a xgb_lag_2025_level_a"

case "${1:-}" in
  check)
    python scripts/00_check_api.py --no-roundtrip
    python -m src.model_tabpfn --dry-run --spatial-lag ;;
  cv)
    python -m src.model_tabpfn --spatial-lag $BASE 2>&1 | tee -a results/tabpfn_t0_lag_run.log ;;
  oot)
    python -m src.out_of_time --model tabpfn --train level_a --spatial-lag 2>&1 | tee -a results/oot_tabpfn_lag_run.log ;;
  oot-full)
    python -m src.out_of_time --model tabpfn --train full --spatial-lag --ignore-limits 2>&1 | tee -a results/oot_tabpfn_lag_run.log ;;
  compare)
    for t in level_a full; do
      for b in tabpfn_t0_$t xgb_lag_$t; do
        [ -f results/oot_tabpfn_t0_lag_$t.json ] && [ -f results/oot_$b.json ] && \
          python -m src.out_of_time --compare tabpfn_t0_lag_$t $b
      done
    done ;;
  *)
    echo "usage: $0 {check|cv|oot|oot-full|compare}" >&2; exit 2 ;;
esac
