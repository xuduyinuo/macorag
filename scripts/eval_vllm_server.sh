#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/eval_macorag.yml}"
previous_arg=""
for arg in "$@"; do
  if [[ "${previous_arg}" == "--config" ]]; then
    CONFIG_PATH="${arg}"
    break
  fi
  previous_arg="${arg}"
done

CONFIGURED_ADAPTER_PATH="$("${PYTHON:-python}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(str(config.get("adapter_path") or "").strip())
PY
)"
ADAPTER_PATH="${ADAPTER_PATH:-${CONFIGURED_ADAPTER_PATH}}"
if [[ -z "${ADAPTER_PATH}" ]]; then
  printf 'Config key adapter_path is required in %s.\n' "${CONFIG_PATH}" >&2
  exit 2
fi
for required_file in adapter_config.json prompt_contract.json; do
  if [[ ! -f "${ADAPTER_PATH}/${required_file}" ]]; then
    printf 'Invalid ADAPTER_PATH=%s: missing %s\n' "${ADAPTER_PATH}" "${required_file}" >&2
    exit 2
  fi
done

if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[eval-vllm] config=%s adapter=%s\n' "${CONFIG_PATH}" "${ADAPTER_PATH}"
  exit 0
fi
"${PYTHON:-python}" -m evaluation.vllm_servers --config "${CONFIG_PATH}" "$@" --adapter-path "${ADAPTER_PATH}"
