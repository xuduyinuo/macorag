#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/train_grpo.yml}"

read -r YAML_MODEL_PATH YAML_HOST YAML_PORT YAML_VLLM_GPU_INDICES YAML_TP YAML_GPU_UTIL YAML_MAX_LEN YAML_DTYPE YAML_LORA_NAME YAML_LORA_INT_ID YAML_LORA_ADAPTER_PATH YAML_DP < <(
  "${PYTHON:-python}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
print(
    config.get("model_path", "model/Qwen2.5-7B-Instruct"),
    config.get("vllm_host", "127.0.0.1"),
    int(config.get("vllm_port", 8000)),
    str(config.get("vllm_gpu_indices", "0")),
    int(config.get("vllm_tensor_parallel_size", 1)),
    float(config.get("vllm_gpu_memory_utilization", 0.75)),
    int(config.get("vllm_max_model_len", 8192)),
    config.get("vllm_dtype", "auto"),
    config.get("vllm_lora_name", "macorag_train"),
    int(config.get("vllm_lora_int_id", 1)),
    config.get("vllm_lora_adapter_path") or config.get("sft_adapter_path", ""),
    int(config.get("vllm_data_parallel_size", 1)),
)
PY
)

export CUDA_VISIBLE_DEVICES="${YAML_VLLM_GPU_INDICES}"

exec "${PYTHON:-python}" -m rl_training.vllm_lora_server \
  --model "${YAML_MODEL_PATH}" \
  --host "${YAML_HOST}" \
  --port "${YAML_PORT}" \
  --tensor-parallel-size "${YAML_TP}" \
  --gpu-memory-utilization "${YAML_GPU_UTIL}" \
  --max-model-len "${YAML_MAX_LEN}" \
  --dtype "${YAML_DTYPE}" \
  --lora-name "${YAML_LORA_NAME}" \
  --lora-int-id "${YAML_LORA_INT_ID}" \
  --data-parallel-size "${YAML_DP}" \
  --lora-adapter-path "${YAML_LORA_ADAPTER_PATH}" \
  "$@"
