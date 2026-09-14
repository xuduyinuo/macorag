#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  printf 'usage: evaluate_variant.bash VARIANT ADAPTER_MODE MODEL_OUTPUT_ROOT MAX_ROUNDS\n' >&2
  exit 2
fi

VARIANT="$1"
ADAPTER_MODE="$2"
MODEL_OUTPUT_ROOT="$3"
MAX_ROUNDS="$4"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
source "${HERE}/process_group.bash"
PYTHON_BIN="${PYTHON:-python}"
EVAL_PORT="${EVAL_PORT:-8100}"
VLLM_MODEL="macorag-ablation-${VARIANT//_/-}"
BASE_URL="http://127.0.0.1:${EVAL_PORT}/v1"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m ablation.prepare_evaluation

ADAPTER_PATH=""
REQUESTED_MODEL_PATH="${BASE_MODEL_PATH:-model/Qwen2.5-7B-Instruct}"
EVALUATION_MODEL_PATH="${REQUESTED_MODEL_PATH}"
if [[ "${ADAPTER_MODE}" == "adapter" ]]; then
  if [[ -n "${EVAL_ADAPTER_PATH:-}" ]]; then
    ADAPTER_PATH="$(${PYTHON_BIN} -m ablation.ablation_cli resolve-adapter \
      --pointer "${HERE}/artifacts/unused.path" --explicit "${EVAL_ADAPTER_PATH}")"
  elif [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
    ADAPTER_PATH="${MODEL_OUTPUT_ROOT}/<latest-run>/adapter"
  else
    ADAPTER_PATH="$(${PYTHON_BIN} -m ablation.ablation_cli latest-adapter --root "${MODEL_OUTPUT_ROOT}")"
  fi
  if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" != "1" ]]; then
    EVALUATION_MODEL_PATH="$(${PYTHON_BIN} -m ablation.ablation_cli adapter-model-identity \
      --adapter "${ADAPTER_PATH}" --expected "${REQUESTED_MODEL_PATH}")"
  fi
fi

if [[ "${ADAPTER_MODE}" == "adapter" && "${MACORAG_LAUNCH_DRY_RUN:-0}" != "1" ]]; then
  EVAL_OUTPUT_DIR="$(dirname "${ADAPTER_PATH}")/evaluation/eval_500_each"
else
  EVAL_OUTPUT_DIR="${MODEL_OUTPUT_ROOT}/evaluation/eval_500_each"
fi
SERVER_LOG="${EVAL_OUTPUT_DIR}/vllm_server.log"

if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" != "1" && "${FORCE_EVALUATION:-0}" != "1" ]] && \
  "${PYTHON_BIN}" -m ablation.ablation_cli evaluation-complete \
    --output "${EVAL_OUTPUT_DIR}" \
    --manifest "${HERE}/data/eval_500_each/manifest.jsonl"; then
  exit 0
fi

if [[ -n "${EVAL_GPU_INDEX:-}" ]]; then
  SELECTED_EVAL_GPU="${EVAL_GPU_INDEX}"
else
  SELECTED_EVAL_GPU="$(${PYTHON_BIN} -m ablation.select_gpu \
    --min-free-mib "${EVAL_MIN_FREE_MIB:-18000}")"
fi
printf '[ablation-eval] selected_gpu=%s selection=%s\n' \
  "${SELECTED_EVAL_GPU}" "$([[ -n "${EVAL_GPU_INDEX:-}" ]] && printf explicit || printf auto)"

SERVER_ARGS=(
  --config "${HERE}/configs/eval_vllm_server.yml"
  --model-path "${EVALUATION_MODEL_PATH}"
  --adapter-path "${ADAPTER_PATH}"
  --vllm-model "${VLLM_MODEL}"
  --gpu-indices "${SELECTED_EVAL_GPU}"
  --vllm-base-urls "${BASE_URL}"
)
EVAL_ARGS=(
  --config "${HERE}/configs/eval_500_each.yml"
  --output-dir "${EVAL_OUTPUT_DIR}"
  --adapter-label "${VARIANT}"
  --model-path "${EVALUATION_MODEL_PATH}"
  --adapter-path "${ADAPTER_PATH}"
  --adapter-identity-path "${ADAPTER_PATH}"
  --max-rounds "${MAX_ROUNDS}"
  --gpu-indices "${SELECTED_EVAL_GPU}"
  --vllm-base-urls "${BASE_URL}"
  --vllm-model "${VLLM_MODEL}"
)

if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[ablation-eval] variant=%s samples=2wiki:500,hotpotqa:500,musique:500 output=%s\n' "${VARIANT}" "${EVAL_OUTPUT_DIR}"
  MACORAG_VLLM_DRY_RUN=1 "${PYTHON_BIN}" -m evaluation.vllm_servers "${SERVER_ARGS[@]}"
  MACORAG_LAUNCH_DRY_RUN=1 CONFIG_PATH="${HERE}/configs/eval_500_each.yml" \
    bash "${REPO_ROOT}/scripts/eval_macorag.sh" "${EVAL_ARGS[@]:2}"
  exit 0
fi

mkdir -p "${EVAL_OUTPUT_DIR}"
setsid "${PYTHON_BIN}" -m evaluation.vllm_servers "${SERVER_ARGS[@]}" >"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    stop_ablation_process_group "${SERVER_PID}" "evaluation-vllm-${VARIANT}"
    SERVER_PID=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"${PYTHON_BIN}" -m ablation.wait_for_vllm \
  --base-url "${BASE_URL}" --model "${VLLM_MODEL}" --kind openai --pid "${SERVER_PID}" \
  --timeout "${VLLM_START_TIMEOUT:-600}"
"${PYTHON_BIN}" -m evaluation.evaluate_rag_model "${EVAL_ARGS[@]}"

cleanup
trap - EXIT INT TERM
