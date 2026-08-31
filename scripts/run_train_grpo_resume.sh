#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/train_grpo.yml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-${1:-}}"
if [[ -z "${RESUME_CHECKPOINT}" ]]; then
  echo "Usage: RESUME_CHECKPOINT=/path/to/checkpoint bash scripts/run_train_grpo_resume.sh [trainer args...]" >&2
  exit 2
fi
if [[ $# -gt 0 && "$1" == "${RESUME_CHECKPOINT}" ]]; then
  shift
fi
if [[ ! -d "${RESUME_CHECKPOINT}" ]]; then
  echo "GRPO full checkpoint directory not found: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi
if [[ ! -f "${RESUME_CHECKPOINT}/checkpoint_manifest.json" ]]; then
  echo "GRPO full checkpoint manifest not found: ${RESUME_CHECKPOINT}/checkpoint_manifest.json" >&2
  exit 1
fi
if [[ ! -f "${RESUME_CHECKPOINT}/COMPLETE" ]]; then
  echo "GRPO full checkpoint is incomplete: ${RESUME_CHECKPOINT}" >&2
  exit 1
fi

export CONFIG_PATH
RUN_UNTIL_STEP="${RUN_UNTIL_STEP:-1000}"

exec bash "${SCRIPT_DIR}/run_train_grpo.sh" \
  --resume-from-checkpoint "${RESUME_CHECKPOINT}" \
  --run-until-step "${RUN_UNTIL_STEP}" \
  "$@"
