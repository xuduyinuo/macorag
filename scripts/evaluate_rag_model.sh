#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/evaluate_rag_model.yml}"

EFFECTIVE_GPU_INDICES="$(
  "${PYTHON:-python}" - "${CONFIG_PATH}" "$@" <<'PY'
import sys

from evaluation.config import parse_args

config_path = sys.argv[1]
args = parse_args(["--config", config_path, *sys.argv[2:]])
gpu_indices = str(getattr(args, "gpu_indices", "") or "").strip()
if gpu_indices:
    print(gpu_indices)
else:
    print(getattr(args, "gpu_index", 0))
PY
)"

export CUDA_VISIBLE_DEVICES="${EFFECTIVE_GPU_INDICES}"

if [[ "${MACORAG_EVAL_DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
  exit 0
fi

"${PYTHON:-python}" -m evaluation.evaluate_rag_model --config "${CONFIG_PATH}" "$@"
