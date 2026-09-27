#!/usr/bin/env bash
# TabPFN-3.5 out-of-time test: fit on 2025, predict the 47,810 transactions
# of 2026 — run on a machine with API access (the author's Mac).
#
# Usage, from the repository root (TABPFN_TOKEN exported):
#   bash scripts/20_tabpfn_out_of_time.sh t0            # Level A (24k) -> 2026, zero-shot
#   bash scripts/20_tabpfn_out_of_time.sh t0-think      # Level A -> 2026, thinking medium
#   bash scripts/20_tabpfn_out_of_time.sh full          # full 2025 base (82k) -> 2026, zero-shot
#   bash scripts/20_tabpfn_out_of_time.sh full-think    # full base -> 2026, thinking medium
#   bash scripts/20_tabpfn_out_of_time.sh compare       # paired tests vs the baselines (no API)
#
# One fit per run (no folds); predictions in chunks of 5,000 rows, every
# response cached under results/tabpfn_cache/oot_<label>/.
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

Q="--quantiles 0.1 0.25 0.5 0.75 0.9"      # zero-shot fits only

case "${1:-}" in
  t0)
    python -m src.out_of_time --model tabpfn --train level_a --thinking off $Q 2>&1 | tee -a results/oot_tabpfn_run.log ;;
  t0-think)
    python -m src.out_of_time --model tabpfn --train level_a --thinking medium 2>&1 | tee -a results/oot_tabpfn_run.log ;;
  full)
    python -m src.out_of_time --model tabpfn --train full --thinking off --ignore-limits $Q 2>&1 | tee -a results/oot_tabpfn_run.log ;;
  full-think)
    python -m src.out_of_time --model tabpfn --train full --thinking medium --ignore-limits 2>&1 | tee -a results/oot_tabpfn_run.log ;;
  compare)
    for t in level_a full; do
      for a in tabpfn_t0_$t tabpfn_t0_think_medium_$t; do
        for b in sar_gm_$t xgb_lag_$t ols_$t; do
          [ -f results/oot_$a.json ] && [ -f results/oot_$b.json ] && python -m src.out_of_time --compare $a $b
        done
      done
    done ;;
  *)
    echo "usage: $0 {t0|t0-think|full|full-think|compare}" >&2; exit 2 ;;
esac
