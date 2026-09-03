#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# 使用 Qwen 专用配置，同时保留通过环境变量切换配置或 Python 的能力。
export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/train_sft_qwen.yml}"

exec "${SCRIPT_DIR}/run_train_sft.sh" "$@"
