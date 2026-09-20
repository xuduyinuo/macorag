#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SFT_V2_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${SFT_V2_ROOT}/../.." && pwd)"

cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.env" ]]; then
  set -a
  source "${REPO_ROOT}/.env"
  set +a
fi

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
CONFIG_PATH="${CONFIG_PATH:-${SFT_V2_ROOT}/config/teacher_trajectory.yml}"

if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[sft-v2-teacher] config=%s module=sft_v2.generate_trajectories\n' "${CONFIG_PATH}"
  exit 0
fi

exec "${PYTHON:-python}" -m sft_v2.generate_trajectories --config "${CONFIG_PATH}" "$@"

