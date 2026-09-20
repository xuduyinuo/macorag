#!/usr/bin/env bash
set -euo pipefail

INVOCATION_DIR="$(pwd -P)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON:-/data/conda/envs/macorag/bin/python}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/train_mappo.yml}"
TRAIN_GPU_INDEX="${TRAIN_GPU_INDEX:-0}"
START_TIMEOUT_SECONDS="${VLLM_START_TIMEOUT_SECONDS:-600}"
STOP_TIMEOUT_SECONDS="${VLLM_STOP_TIMEOUT_SECONDS:-30}"
LAUNCH_DRY_RUN="${MAPPO_LAUNCH_DRY_RUN:-0}"
TRAIN_ARGS=("$@")
RESUME_CHECKPOINT=""

for ((ARG_INDEX=0; ARG_INDEX<${#TRAIN_ARGS[@]}; ARG_INDEX++)); do
  ARG_VALUE="${TRAIN_ARGS[ARG_INDEX]}"
  case "${ARG_VALUE}" in
    --config)
      if ((ARG_INDEX + 1 >= ${#TRAIN_ARGS[@]})); then
        printf 'Missing value after --config.\n' >&2
        exit 2
      fi
      CONFIG_PATH="${TRAIN_ARGS[ARG_INDEX + 1]}"
      ;;
    --config=*)
      CONFIG_PATH="${ARG_VALUE#--config=}"
      ;;
    --resume-from-checkpoint)
      if ((ARG_INDEX + 1 >= ${#TRAIN_ARGS[@]})); then
        printf 'Missing value after --resume-from-checkpoint.\n' >&2
        exit 2
      fi
      RESUME_CHECKPOINT="${TRAIN_ARGS[ARG_INDEX + 1]}"
      ;;
    --resume-from-checkpoint=*)
      RESUME_CHECKPOINT="${ARG_VALUE#--resume-from-checkpoint=}"
      ;;
  esac
done

if [[ "${CONFIG_PATH}" != /* ]]; then
  CONFIG_PATH="${INVOCATION_DIR}/${CONFIG_PATH}"
fi
if [[ ! -f "${CONFIG_PATH}" ]]; then
  printf 'MAPPO config not found: %s\n' "${CONFIG_PATH}" >&2
  exit 2
fi

# Switch away from src/rl_v2 before invoking Python.
# A package module named types.py would shadow Python's standard-library
# types module via sys.path[0]; keep all Python calls rooted at the repository.
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mapfile -t VLLM_SETTINGS < <(
  "${PYTHON_BIN}" -c '
import sys
from pathlib import Path
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
values = (
    str(bool(cfg.get("use_vllm_generation", True))).lower(),
    str(cfg.get("vllm_host", "127.0.0.1")),
    str(int(cfg.get("vllm_port", 8003))),
    str(cfg.get("vllm_gpu_indices", "1")),
    str(cfg.get("vllm_served_model_name", "rl_v2_base")),
    str(cfg.get("vllm_lora_name", "rl_v2_policy")),
    str(cfg.get("retrieval_device", "cuda:1")),
)
print("\n".join(values))
' "${CONFIG_PATH}"
)

if ((${#VLLM_SETTINGS[@]} != 7)); then
  printf 'Unable to parse vLLM settings from %s.\n' "${CONFIG_PATH}" >&2
  exit 2
fi
USE_VLLM="${VLLM_SETTINGS[0]}"
VLLM_HOST="${VLLM_SETTINGS[1]}"
VLLM_PORT="${VLLM_SETTINGS[2]}"
VLLM_GPU_INDICES="${VLLM_SETTINGS[3]}"
VLLM_BASE_MODEL="${VLLM_SETTINGS[4]}"
VLLM_LORA_NAME="${VLLM_SETTINGS[5]}"
RETRIEVAL_DEVICE="${VLLM_SETTINGS[6]}"
TRAIN_VISIBLE_GPU_INDICES="${TRAIN_VISIBLE_GPU_INDICES:-${TRAIN_GPU_INDEX},${VLLM_GPU_INDICES}}"
if [[ "${RETRIEVAL_DEVICE}" != "cuda:1" ]]; then
  printf 'Evaluation-aligned RL-v2 retrieval requires retrieval_device=cuda:1; got %s.\n' \
    "${RETRIEVAL_DEVICE}" >&2
  exit 2
fi

if [[ "${USE_VLLM}" != "true" ]]; then
  printf 'Config has use_vllm_generation=false; this launcher requires vLLM.\n' >&2
  exit 2
fi
IFS=',' read -r -a VLLM_GPU_ARRAY <<< "${VLLM_GPU_INDICES}"
for GPU_INDEX in "${VLLM_GPU_ARRAY[@]}"; do
  if [[ "${GPU_INDEX//[[:space:]]/}" == "${TRAIN_GPU_INDEX}" ]]; then
    printf 'Training GPU %s overlaps vLLM GPUs %s.\n' "${TRAIN_GPU_INDEX}" "${VLLM_GPU_INDICES}" >&2
    exit 2
  fi
done

VLLM_ADAPTER_ARGS=()
if [[ -n "${VLLM_ADAPTER_PATH:-}" ]]; then
  VLLM_ADAPTER_ARGS=(--adapter-path "${VLLM_ADAPTER_PATH}")
elif [[ -n "${RESUME_CHECKPOINT}" ]]; then
  VLLM_ADAPTER_ARGS=(--adapter-path "${RESUME_CHECKPOINT%/}/actor")
fi

if [[ "${LAUNCH_DRY_RUN}" == "1" ]]; then
  "${PYTHON_BIN}" -m src.rl_v2.launch_vllm \
    --config "${CONFIG_PATH}" "${VLLM_ADAPTER_ARGS[@]}" --dry-run
  printf '[rl-v2-trainer] CUDA_VISIBLE_DEVICES=%s %s -m src.rl_v2.train_mappo --config %s' \
    "${TRAIN_VISIBLE_GPU_INDICES}" "${PYTHON_BIN}" "${CONFIG_PATH}"
  if ((${#TRAIN_ARGS[@]})); then
    printf ' %q' "${TRAIN_ARGS[@]}"
  fi
  printf '\n'
  exit 0
fi

PORT_IN_USE="$("${PYTHON_BIN}" -c '
import socket, sys
sock = socket.socket()
sock.settimeout(1.0)
try:
    used = sock.connect_ex((sys.argv[1], int(sys.argv[2]))) == 0
finally:
    sock.close()
print("1" if used else "0")
' "${VLLM_HOST}" "${VLLM_PORT}")"
if [[ "${PORT_IN_USE}" == "1" ]]; then
  EXISTING_MODELS="$(curl --noproxy '*' --silent --max-time 2 \
    "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" || true)"
  printf 'vLLM port is already occupied; refusing to reuse or stop it: host=%s port=%s response=%s\n' \
    "${VLLM_HOST}" "${VLLM_PORT}" "${EXISTING_MODELS:-<non-HTTP service>}" >&2
  exit 2
fi

REMOVE_VLLM_LOG=0
if [[ -n "${MAPPO_VLLM_LOG:-}" ]]; then
  VLLM_LOG="${MAPPO_VLLM_LOG}"
else
  VLLM_LOG="$(mktemp /tmp/mappo_vllm_XXXXXXXX.log)"
  REMOVE_VLLM_LOG=1
fi
VLLM_PID=""

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    printf '[mappo] stopping owned vLLM process group %s\n' "${VLLM_PID}"
    kill -TERM -- "-${VLLM_PID}" 2>/dev/null || true
    local waited=0
    while kill -0 "${VLLM_PID}" 2>/dev/null && ((waited < STOP_TIMEOUT_SECONDS)); do
      sleep 1
      waited=$((waited + 1))
    done
    if kill -0 "${VLLM_PID}" 2>/dev/null; then
      kill -KILL -- "-${VLLM_PID}" 2>/dev/null || true
    fi
    wait "${VLLM_PID}" 2>/dev/null || true
  fi
  if [[ "${REMOVE_VLLM_LOG}" == "1" ]]; then
    rm -f -- "${VLLM_LOG}"
  fi
  exit "${status}"
}
trap cleanup EXIT INT TERM

printf '[mappo] starting vLLM on physical GPU(s) %s; service output is hidden\n' \
  "${VLLM_GPU_INDICES}"
setsid "${PYTHON_BIN}" -m src.rl_v2.launch_vllm \
  --config "${CONFIG_PATH}" "${VLLM_ADAPTER_ARGS[@]}" \
  >"${VLLM_LOG}" 2>&1 &
VLLM_PID=$!

READY=0
for ((WAITED=0; WAITED<START_TIMEOUT_SECONDS; WAITED+=2)); do
  if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
    printf 'vLLM exited before readiness. Last log lines:\n' >&2
    tail -n 80 "${VLLM_LOG}" >&2 || true
    exit 1
  fi
  MODELS_PAYLOAD="$(curl --noproxy '*' --silent --max-time 2 \
    "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" || true)"
  if [[ -n "${MODELS_PAYLOAD}" ]] && "${PYTHON_BIN}" -c '
import json, sys
try:
    payload = json.loads(sys.argv[1])
except Exception:
    raise SystemExit(1)
visible = {str(item.get("id")) for item in payload.get("data", []) if isinstance(item, dict)}
raise SystemExit(0 if {sys.argv[2], sys.argv[3]} <= visible else 1)
' "${MODELS_PAYLOAD}" "${VLLM_BASE_MODEL}" "${VLLM_LORA_NAME}"; then
    READY=1
    break
  fi
  sleep 2
done
if [[ "${READY}" != "1" ]]; then
  printf 'Timed out after %ss waiting for vLLM models %s and %s. Last log lines:\n' \
    "${START_TIMEOUT_SECONDS}" "${VLLM_BASE_MODEL}" "${VLLM_LORA_NAME}" >&2
  tail -n 80 "${VLLM_LOG}" >&2 || true
  exit 1
fi

printf '[mappo] vLLM ready at http://%s:%s; learner uses physical GPU %s; retrieval encoder shares vLLM GPU(s) %s\n' \
  "${VLLM_HOST}" "${VLLM_PORT}" "${TRAIN_GPU_INDEX}" "${VLLM_GPU_INDICES}"
CUDA_VISIBLE_DEVICES="${TRAIN_VISIBLE_GPU_INDICES}" "${PYTHON_BIN}" \
  -m src.rl_v2.train_mappo --config "${CONFIG_PATH}" "${TRAIN_ARGS[@]}"
