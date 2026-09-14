#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  printf 'usage: launch_grpo.bash VARIANT CONFIG [extra trainer args...]\n' >&2
  exit 2
fi

VARIANT="$1"
CONFIG_PATH="$2"
shift 2

ABLATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ABLATION_DIR}/.." && pwd)"
source "${ABLATION_DIR}/process_group.bash"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m ablation.prepare_data

SFT_ADAPTER="$(${PYTHON_BIN} -m ablation.ablation_cli resolve-adapter \
  --pointer "${ABLATION_DIR}/artifacts/sft_shared_adapter.path" \
  --explicit "${SFT_ADAPTER_PATH:-}")"

read -r GPU_INDICES NPROC VLLM_GPU VLLM_HOST VLLM_PORT MODEL_PATH VLLM_TP VLLM_UTIL VLLM_MAX_LEN VLLM_MAX_SEQS VLLM_DTYPE LORA_NAME LORA_ID OUTPUT_ROOT < <("${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path
import yaml
payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
indices = str(payload.get("gpu_indices") or payload.get("gpu_index") or "0")
items = [item.strip() for item in indices.split(",") if item.strip()]
print(
    indices,
    max(1, len(items)),
    str(payload.get("vllm_gpu_indices", "1")),
    str(payload.get("vllm_host", "127.0.0.1")),
    int(payload.get("vllm_port", 8000)),
    str(payload["model_path"]),
    int(payload.get("vllm_tensor_parallel_size", 1)),
    float(payload.get("vllm_gpu_memory_utilization", 0.85)),
    int(payload.get("vllm_max_model_len", 4096)),
    int(payload.get("vllm_max_num_seqs", 8)),
    str(payload.get("vllm_dtype", "auto")),
    str(payload.get("vllm_lora_name", "macorag_train")),
    int(payload.get("vllm_lora_int_id", 1)),
    str(payload["output_root"]),
)
PY
)
export CUDA_VISIBLE_DEVICES="${GPU_INDICES}"

if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[ablation-grpo-vllm] variant=%s model=%s adapter=%s gpu=%s endpoint=http://%s:%s\n' \
    "${VARIANT}" "${MODEL_PATH}" "${SFT_ADAPTER}" "${VLLM_GPU}" "${VLLM_HOST}" "${VLLM_PORT}"
  printf '[ablation-grpo] variant=%s config=%s sft_adapter=%s CUDA_VISIBLE_DEVICES=%s nproc=%s\n' \
    "${VARIANT}" "${CONFIG_PATH}" "${SFT_ADAPTER}" "${CUDA_VISIBLE_DEVICES}" "${NPROC}"
  exit 0
fi

mkdir -p "${OUTPUT_ROOT}"
TRAIN_SERVER_LOG="${OUTPUT_ROOT}/training_vllm.log"
setsid env CUDA_VISIBLE_DEVICES="${VLLM_GPU}" "${PYTHON_BIN}" -m rl_training.vllm_lora_server \
  --model "${MODEL_PATH}" \
  --host "${VLLM_HOST}" \
  --port "${VLLM_PORT}" \
  --tensor-parallel-size "${VLLM_TP}" \
  --gpu-memory-utilization "${VLLM_UTIL}" \
  --max-model-len "${VLLM_MAX_LEN}" \
  --max-num-seqs "${VLLM_MAX_SEQS}" \
  --dtype "${VLLM_DTYPE}" \
  --lora-name "${LORA_NAME}" \
  --lora-int-id "${LORA_ID}" \
  --lora-adapter-path "${SFT_ADAPTER}" >"${TRAIN_SERVER_LOG}" 2>&1 &
TRAIN_SERVER_PID=$!
cleanup_training_server() {
  if [[ -n "${TRAIN_SERVER_PID:-}" ]]; then
    stop_ablation_process_group "${TRAIN_SERVER_PID}" "training-vllm"
    TRAIN_SERVER_PID=""
  fi
}
trap cleanup_training_server EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PYTHON_BIN}" -m ablation.wait_for_vllm \
  --base-url "http://${VLLM_HOST}:${VLLM_PORT}" --model "${MODEL_PATH}" \
  --kind training --pid "${TRAIN_SERVER_PID}" --timeout "${VLLM_START_TIMEOUT:-600}"

"${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${NPROC}" \
  -m ablation.train_grpo_ablation --ablation-variant "${VARIANT}" \
  --config "${CONFIG_PATH}" --sft-adapter-path "${SFT_ADAPTER}" "$@"

# Release the training vLLM GPU before the variant evaluation starts.
cleanup_training_server
trap - EXIT INT TERM
