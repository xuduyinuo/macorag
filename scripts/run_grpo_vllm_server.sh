#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/train_grpo.yml}"

read -r YAML_SYNC_MODE YAML_MODEL_PATH YAML_HOST YAML_PORT YAML_VLLM_GPU_INDICES YAML_TP YAML_GPU_UTIL YAML_MAX_LEN YAML_MAX_NUM_SEQS YAML_DTYPE YAML_LORA_NAME YAML_LORA_INT_ID YAML_LORA_ADAPTER_PATH YAML_DP < <(
  "${PYTHON:-python}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(
    config.get("vllm_sync_mode", "dense"),
    config.get("model_path", "model/Qwen2.5-3B-Instruct"),
    config.get("vllm_host", "127.0.0.1"),
    int(config.get("vllm_port", 8000)),
    str(config.get("vllm_gpu_indices", "0")),
    int(config.get("vllm_tensor_parallel_size", 1)),
    float(config.get("vllm_gpu_memory_utilization", 0.85)),
    int(config.get("vllm_max_model_len", 4608)),
    int(config.get("vllm_max_num_seqs", 8)),
    config.get("vllm_dtype", "auto"),
    config.get("vllm_lora_name", "macorag_train"),
    int(config.get("vllm_lora_int_id", 1)),
    config.get("vllm_lora_adapter_path") or config.get("sft_adapter_path") or "__MISSING_SFT_ADAPTER_PATH__",
    int(config.get("vllm_data_parallel_size", 1)),
)
PY
)

if [[ "${YAML_SYNC_MODE}" == "lora" ]]; then
  if [[ "${YAML_LORA_ADAPTER_PATH}" == "__MISSING_SFT_ADAPTER_PATH__" ]]; then
    printf 'Config key sft_adapter_path is required in %s.\n' "${CONFIG_PATH}" >&2
    exit 2
  fi
  for required_file in adapter_config.json prompt_contract.json; do
    if [[ ! -f "${YAML_LORA_ADAPTER_PATH}/${required_file}" ]]; then
      printf 'Invalid configured adapter path %s: missing %s\n' "${YAML_LORA_ADAPTER_PATH}" "${required_file}" >&2
      exit 2
    fi
  done
fi

if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[grpo-vllm] config=%s mode=%s dtype=%s gpu=%s gpu_memory_utilization=%s max_model_len=%s max_num_seqs=%s adapter=%s\n' "${CONFIG_PATH}" "${YAML_SYNC_MODE}" "${YAML_DTYPE}" "${YAML_VLLM_GPU_INDICES}" "${YAML_GPU_UTIL}" "${YAML_MAX_LEN}" "${YAML_MAX_NUM_SEQS}" "${YAML_LORA_ADAPTER_PATH}"
  exit 0
fi
if [[ "${YAML_SYNC_MODE}" == "lora" ]]; then
  export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"
  exec "${PYTHON:-python}" -m rl_training.vllm_lora_server \
    --model "${YAML_MODEL_PATH}" \
    --host "${YAML_HOST}" \
    --port "${YAML_PORT}" \
    --tensor-parallel-size "${YAML_TP}" \
    --gpu-memory-utilization "${YAML_GPU_UTIL}" \
    --max-model-len "${YAML_MAX_LEN}" \
    --max-num-seqs "${YAML_MAX_NUM_SEQS}" \
    --dtype "${YAML_DTYPE}" \
    --lora-name "${YAML_LORA_NAME}" \
    --lora-int-id "${YAML_LORA_INT_ID}" \
    --data-parallel-size "${YAML_DP}" \
    --lora-adapter-path "${YAML_LORA_ADAPTER_PATH}" \
    "$@"
fi

export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"

exec trl vllm-serve \
  --model "${YAML_MODEL_PATH}" \
  --host "${YAML_HOST}" \
  --port "${YAML_PORT}" \
  --tensor-parallel-size "${YAML_TP}" \
  --gpu-memory-utilization "${YAML_GPU_UTIL}" \
  --max-model-len "${YAML_MAX_LEN}" \
  --max-num-seqs "${YAML_MAX_NUM_SEQS}" \
  --dtype "${YAML_DTYPE}" \
  "$@"
