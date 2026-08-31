#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/eval_1000_stratified_v2}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/eval_300_grpo_fixed}"
if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[grpo-validation-manifest] data_root=%s output_dir=%s per_dataset=100 seed=20260831\n' "${DATA_ROOT}" "${OUTPUT_DIR}"
  exit 0
fi
exec "${PYTHON:-python}" -m evaluation.fixed_manifest --data-root "${DATA_ROOT}" --output-dir "${OUTPUT_DIR}"
