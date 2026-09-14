#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"
if [[ "${EVAL_ONLY:-0}" != "1" ]]; then
  "${PYTHON_BIN}" -m ablation.prepare_data
  "${PYTHON_BIN}" -m ablation.ablation_cli register-base \
    --model "${BASE_MODEL_PATH:-${REPO_ROOT}/model/Qwen2.5-7B-Instruct}" \
    --output "${HERE}/outputs/wo_sft_grpo" \
    --data-manifest "${HERE}/data/manifest.json"
fi
if [[ "${SKIP_EVALUATION:-0}" != "1" ]]; then
  bash "${HERE}/evaluate_variant.bash" wo_sft_grpo base "${HERE}/outputs/wo_sft_grpo" 4
fi
