#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

: "${ADAPTER_LABEL:?Set ADAPTER_LABEL to sft, rl-step-300, rl-step-500, rl-step-700, or rl-step-1000}"
: "${ADAPTER_PATH:?Set ADAPTER_PATH to the exact SFT adapter or RL checkpoint being served}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a stable evaluation directory}"

for required_file in adapter_config.json prompt_contract.json; do
  if [[ ! -f "${ADAPTER_PATH}/${required_file}" ]]; then
    printf 'Invalid ADAPTER_PATH=%s: missing %s\n' "${ADAPTER_PATH}" "${required_file}" >&2
    exit 2
  fi
done

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/eval_grpo_fixed.yml}"
export CONFIG_PATH

exec "${SCRIPT_DIR}/eval_macorag.sh" \
  --output-dir "${OUTPUT_DIR}" \
  --adapter-label "${ADAPTER_LABEL}" \
  --adapter-identity-path "${ADAPTER_PATH}" \
  --resume \
  "$@"
