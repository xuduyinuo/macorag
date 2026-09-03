#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/grpo_staged_early_stop.yml}"

cd "${REPO_ROOT}"
exec "${PYTHON:-python}" -m rl_training.staged_early_stop --config "${CONFIG_PATH}" "$@"
