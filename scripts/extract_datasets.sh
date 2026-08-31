#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-${REPO_ROOT}/config/extract_train.yml}"
EVAL_CONFIG_PATH="${EVAL_CONFIG_PATH:-${REPO_ROOT}/config/extract_eval.yml}"

printf '[extract-datasets] train_config=%s eval_config=%s\n' \
  "${TRAIN_CONFIG_PATH}" "${EVAL_CONFIG_PATH}"
if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi

"${PYTHON:-python}" -m data_processing.extract_stratified_datasets \
  --train-config "${TRAIN_CONFIG_PATH}" \
  --eval-config "${EVAL_CONFIG_PATH}" \
  "$@"
