#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/retrieval_teacher.yml}"

printf '[build-teacher-retrieval] config=%s\n' "${CONFIG_PATH}"
if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
"${PYTHON:-python}" -m data_processing.retrieval_cli --config "${CONFIG_PATH}" "$@"
