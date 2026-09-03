#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/eval_vllm_server_qwen.yml}"

exec "${SCRIPT_DIR}/eval_vllm_server.sh" "$@"
