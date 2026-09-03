#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
exec "${PYTHON:-python}" -m rl_training.answer_reward_ablation \
  --config "${CONFIG_PATH:-${REPO_ROOT}/config/grpo_answer_reward_ablation.yml}" "$@"
