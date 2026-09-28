#!/usr/bin/env bash
# Permutation SHAP for the TabPFN-3.5 model fitted on the 2025 Level A base —
# run on a machine with API access (the author's Mac). No refit: the model
# record cached by the out-of-time run is reloaded.
#
# Usage, from the repository root (TABPFN_TOKEN exported):
#   bash scripts/30_tabpfn_shap.sh smoke     # local mock model, no API: pipeline test
#   bash scripts/30_tabpfn_shap.sh check     # samples + server cost estimate, no quota used
#   bash scripts/30_tabpfn_shap.sh run       # explain the 50 sampled 2026 rows (~27k predicted rows)
#
# Every API response is cached under results/shap/cache/<label>/, so an
# interrupted run resumes without paying twice and a repeated run is free.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv-tabpfn ]; then
  python3 -m venv .venv-tabpfn
fi
# shellcheck disable=SC1091
source .venv-tabpfn/bin/activate
pip install -q --upgrade pip >/dev/null
pip install -q -r requirements-tabpfn.txt

case "${1:-}" in
  smoke)
    python -m src.shap_tabpfn --smoke ;;
  check)
    python -m src.shap_tabpfn --dry-run ;;
  run)
    if [ -z "${TABPFN_TOKEN:-}" ]; then
      echo "TABPFN_TOKEN is not set. Run: export TABPFN_TOKEN=\"<key>\"" >&2
      exit 1
    fi
    python -m src.shap_tabpfn 2>&1 | tee -a results/shap/shap_run.log ;;
  *)
    echo "usage: $0 {smoke|check|run}" >&2; exit 2 ;;
esac
