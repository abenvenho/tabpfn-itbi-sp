#!/usr/bin/env bash
# Two more TabPFN-3.5 variants — run on a machine with API access (the author's Mac).
#
#   TEXT VARIANT  the unit complement ("AP 201 E 3VGS") and the Referência field
#                 (building / development name) added as free text to the plain
#                 table, coordinates kept. Design and expectations: src/model_tabpfn.py.
#   GROUPED THINKING  thinking mode (medium) with the spatial block as group_col,
#                 so the fit's internal validation never splits a block.
#
# Usage, from the repository root (TABPFN_TOKEN exported):
#   bash scripts/70_text_and_grouped_thinking.sh check      # cost estimates, no quota used
#   bash scripts/70_text_and_grouped_thinking.sh text       # text variant: leg 1, leg 2 (24k), leg 2 (82k)
#   bash scripts/70_text_and_grouped_thinking.sh think-grp  # grouped thinking: leg 1, leg 2 (24k)
#   bash scripts/70_text_and_grouped_thinking.sh all        # both
#   bash scripts/70_text_and_grouped_thinking.sh summary    # tables with every model (no API)
#
# Needs data/itbi_sp_reference_field.csv.gz (python pipeline/05_reference_field.py).
# Every API response is cached under results/tabpfn_cache/<label>/, so a step
# can be interrupted and re-run without paying twice; a failed run does not
# stop the ones after it.
set -uo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv-tabpfn ]; then
  python3 -m venv .venv-tabpfn
fi
# shellcheck disable=SC1091
source .venv-tabpfn/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements-tabpfn.txt

if [ "${1:-}" != "summary" ] && [ -z "${TABPFN_TOKEN:-}" ]; then
  echo "TABPFN_TOKEN is not set. Run: export TABPFN_TOKEN=\"<key>\"" >&2
  exit 1
fi

LOG=results/variants_run.log
FAILED=()
run() {           # run <description> <command...>
  local what="$1"; shift
  echo "=== $what" | tee -a "$LOG"
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    FAILED+=("$what")
    echo "!!! FAILED: $what" | tee -a "$LOG"
  fi
}
TXT="--text complemento referencia"

check() {
  python -m src.model_tabpfn --dry-run --thinking off $TXT
  python -m src.model_tabpfn --dry-run --thinking medium --group-col
}

text() {
  run "text variant, leg 2, Level A (24k) -> 2026" \
    python -m src.out_of_time --model tabpfn --train level_a --thinking off $TXT
  run "text variant, leg 2, full base (82k) -> 2026" \
    python -m src.out_of_time --model tabpfn --train full --thinking off --ignore-limits $TXT
  run "text variant, leg 1, block CV on Level A" \
    python -m src.model_tabpfn --thinking off $TXT
}

think_grp() {
  run "grouped thinking, leg 2, Level A (24k) -> 2026" \
    python -m src.out_of_time --model tabpfn --train level_a --thinking medium --group-col
  run "grouped thinking, leg 1, block CV on Level A" \
    python -m src.model_tabpfn --thinking medium --group-col
}

report() {
  if [ ${#FAILED[@]} -gt 0 ]; then
    echo; echo "Runs that failed (see $LOG):"; printf '  - %s\n' "${FAILED[@]}"
  fi
}

case "${1:-}" in
  check)     check ;;
  text)      text; report ;;
  think-grp) think_grp; report ;;
  all)       text; think_grp; python scripts/72_variants_summary.py 2>&1 | tee -a "$LOG"; report ;;
  summary)   python scripts/72_variants_summary.py ;;
  *)
    echo "usage: $0 {check|text|think-grp|all|summary}" >&2; exit 2 ;;
esac
