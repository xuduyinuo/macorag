#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/src/sft_v2/config/train_sft.yml}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda/envs/macorag/bin/python}"

GPU_INDICES="$(${PYTHON_BIN} - "${CONFIG_PATH}" <<'PY'
import sys
import yaml
with open(sys.argv[1], encoding="utf-8") as handle:
    print(str((yaml.safe_load(handle) or {}).get("gpu_indices", "0")))
PY
)"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${GPU_INDICES}}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export SFT_V2_RUN_ID="${SFT_V2_RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"

IFS=',' read -r -a GPU_ARRAY <<< "${CUDA_VISIBLE_DEVICES}"
NPROC="${#GPU_ARRAY[@]}"
COMMAND=(
  "${PYTHON_BIN}" -m torch.distributed.run
  --standalone
  --nproc_per_node="${NPROC}"
  -m sft_v2.train_sft
  --config "${CONFIG_PATH}"
  "$@"
)

CHECK_ONLY=0
for argument in "$@"; do
  if [[ "${argument}" == "--check-only" ]]; then
    CHECK_ONLY=1
    break
  fi
done
if [[ "${CHECK_ONLY}" == "1" ]]; then
  COMMAND=(
    "${PYTHON_BIN}" -m sft_v2.train_sft
    --config "${CONFIG_PATH}"
    "$@"
  )
fi

printf 'SFT-v2 command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
if [[ "${SFT_V2_DRY_RUN:-0}" == "1" ]]; then
  exit 0
fi
exec "${COMMAND[@]}"
