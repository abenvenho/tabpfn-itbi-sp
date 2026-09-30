#!/usr/bin/env bash
# ABLATION, apart from the main analysis: TabPFN-3.5 zero-shot on the plain
# table PLUS the nominal location of the ITBI form — district field (bairro)
# and postal code (cep) as raw high-cardinality strings. How much do names
# add to raw coordinates? Design and expectations: src/model_tabpfn.py.
#
# Usage, from the repository root (TABPFN_TOKEN exported):
#   bash scripts/50_nominal_location_ablation.sh probe     # 600-row check that the server takes the columns
#   bash scripts/50_nominal_location_ablation.sh all       # probe + every run below + summary
#   bash scripts/50_nominal_location_ablation.sh cv        # leg 1: leave-one-block-out on Level A (3 variants x 10 folds)
#   bash scripts/50_nominal_location_ablation.sh oot       # leg 2: Level A (24k) -> 2026 (3 variants + 1 categorical)
#   bash scripts/50_nominal_location_ablation.sh oot-full  # leg 2: full 2025 base (82k) -> 2026 (3 variants)
#   bash scripts/50_nominal_location_ablation.sh summary   # tables from results/ (no API)
#
# Second ablation — a name IN PLACE OF the coordinates (lat/lon dropped):
#   bash scripts/50_nominal_location_ablation.sh replace          # every run below + comparison with all models
#   bash scripts/50_nominal_location_ablation.sh replace-summary  # the comparison table only (no API)
# Strict family first (no station distance either — the postal code is the
# only spatial information TabPFN has): its floor, postal code as a string,
# postal code as a number. Then, with the station distance kept: floor,
# postal code, district field, both, postal code as a number. Each in leg 2
# (24k and 82k) and leg 1. The baselines are not re-run.
#
# Variants of the first ablation: +bairro, +cep, +bairro+cep. Same rows, folds and seed as the main
# runs. Every API response is cached under results/tabpfn_cache/<label>/, so a
# step can be interrupted and re-run without paying twice; a failed run does
# not stop the ones after it.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv-tabpfn ]; then
  python3 -m venv .venv-tabpfn
fi
# shellcheck disable=SC1091
source .venv-tabpfn/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements-tabpfn.txt

if [ "${1:-}" != "summary" ] && [ "${1:-}" != "replace-summary" ] && [ -z "${TABPFN_TOKEN:-}" ]; then
  echo "TABPFN_TOKEN is not set. Run: export TABPFN_TOKEN=\"<key>\"" >&2
  exit 1
fi

LOG=results/nominal_ablation_run.log
FAILED=()
run() {           # run <description> <command...>
  local what="$1"; shift
  echo "=== $what" | tee -a "$LOG"
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    FAILED+=("$what")
    echo "!!! FAILED: $what" | tee -a "$LOG"
  fi
}

probe()  { python -m src.nominal_ablation probe 2>&1 | tee -a "$LOG"; }

cv() {
  for v in "bairro" "cep" "bairro cep"; do
    # shellcheck disable=SC2086
    run "leg 1, + $v" python -m src.model_tabpfn --thinking off --nominal $v \
        --compare-with tabpfn_t0_2025_level_a xgb_lag_2025_level_a
  done
}

oot() {
  local Q="--quantiles 0.1 0.25 0.5 0.75 0.9"
  for v in "bairro" "cep" "bairro cep"; do
    # shellcheck disable=SC2086
    run "leg 2 Level A, + $v" python -m src.out_of_time --model tabpfn --train level_a --thinking off --nominal $v $Q
  done
  run "leg 2 Level A, + bairro cep (declared categorical)" \
      python -m src.out_of_time --model tabpfn --train level_a --thinking off --nominal bairro cep --nominal-categorical
}

oot_full() {
  for v in "bairro" "cep" "bairro cep"; do
    # shellcheck disable=SC2086
    run "leg 2 full base, + $v" python -m src.out_of_time --model tabpfn --train full --thinking off --ignore-limits --nominal $v
  done
}

strict() {
  # the postal code as the ONLY spatial information: no lat/lon, no station distance
  for v in "cep_num" "cep" "-"; do
    local nom=""; [ "$v" != "-" ] && nom="--nominal $v"
    # shellcheck disable=SC2086
    run "strict, 2026 from Level A: ${v}" python -m src.out_of_time --model tabpfn --train level_a --thinking off --no-coords --no-station $nom
    # shellcheck disable=SC2086
    run "strict, 2026 from the full base: ${v}" python -m src.out_of_time --model tabpfn --train full --thinking off --ignore-limits --no-coords --no-station $nom
    # shellcheck disable=SC2086
    run "strict, block CV on Level A: ${v}" python -m src.model_tabpfn --thinking off --no-coords --no-station $nom
  done
}

replace() {
  # "-" stands for no name at all: the floor
  for v in "-" "cep" "bairro" "bairro cep" "cep_num"; do
    local nom=""; [ "$v" != "-" ] && nom="--nominal $v"
    # shellcheck disable=SC2086
    run "no lat/lon, 2026 from Level A: ${v}" python -m src.out_of_time --model tabpfn --train level_a --thinking off --no-coords $nom
  done
  for v in "-" "cep" "bairro" "bairro cep" "cep_num"; do
    local nom=""; [ "$v" != "-" ] && nom="--nominal $v"
    # shellcheck disable=SC2086
    run "no lat/lon, 2026 from the full base: ${v}" python -m src.out_of_time --model tabpfn --train full --thinking off --ignore-limits --no-coords $nom
  done
  for v in "-" "cep" "bairro" "bairro cep" "cep_num"; do
    local nom=""; [ "$v" != "-" ] && nom="--nominal $v"
    # shellcheck disable=SC2086
    run "no lat/lon, block CV on Level A: ${v}" python -m src.model_tabpfn --thinking off --no-coords $nom
  done
}

report() {
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo; echo "Runs that failed (see $LOG):"; printf '  - %s\n' "${FAILED[@]}"
  fi
}

case "${1:-}" in
  probe)    probe ;;
  cv)       cv; report ;;
  oot)      oot; report ;;
  oot-full) oot_full; report ;;
  summary)  python -m src.nominal_ablation summary ;;
  replace-summary) python -m src.nominal_ablation descriptors ;;
  replace)
    python -m src.nominal_ablation usage 2>&1 | tee -a "$LOG"
    strict
    python -m src.nominal_ablation descriptors 2>&1 | tee -a "$LOG"     # interim table: strict family
    replace
    python -m src.nominal_ablation usage 2>&1 | tee -a "$LOG"
    python -m src.nominal_ablation descriptors 2>&1 | tee -a "$LOG"
    report ;;
  all)
    if ! probe; then
      echo "Probe failed: nothing else was run." >&2; exit 1
    fi
    oot; oot_full; cv
    python -m src.nominal_ablation summary 2>&1 | tee -a "$LOG"
    report ;;
  *)
    echo "usage: $0 {probe|all|cv|oot|oot-full|summary|replace|replace-summary}" >&2; exit 2 ;;
esac
