#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/train_grpo.yml}"

read -r YAML_GPU_INDICES YAML_NPROC_PER_NODE < <(
  "${PYTHON:-python}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
gpu_indices = str(config.get("gpu_indices") or config.get("gpu_index") or "0").strip()
gpu_list = [item.strip() for item in gpu_indices.split(",") if item.strip()]
nproc = max(1, len(gpu_list))
print(gpu_indices, nproc)
PY
)

export CUDA_VISIBLE_DEVICES="${YAML_GPU_INDICES}"
NPROC_PER_NODE="${YAML_NPROC_PER_NODE}"

torchrun --standalone --nnodes=1 --nproc_per_node=${NPROC_PER_NODE} \
  -m rl_training.train_grpo_macorag --config "${CONFIG_PATH}" "$@"
