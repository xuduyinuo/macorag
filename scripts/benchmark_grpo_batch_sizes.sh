#!/usr/bin/env bash
set -uo pipefail

CONFIG="${CONFIG:-config/train_grpo.yml}"
SAMPLE_COUNT="${SAMPLE_COUNT:-20}"
BATCH_SIZES="${BATCH_SIZES:-1 2 4}"
TRAIN_GPU="${TRAIN_GPU:-1}"
REFERENCE_BATCH_SIZE="${REFERENCE_BATCH_SIZE:-4}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-outputs/grpo_batchsize_benchmark/$(date +%Y-%m-%d_%H-%M-%S)}"
DRY_RUN="${DRY_RUN:-0}"
PYTHON_BIN="${PYTHON_BIN:-/data/conda/envs/macorag/bin/python}"

for batch_size in ${BATCH_SIZES}; do
    candidate_dir="${BENCHMARK_ROOT}/bs${batch_size}"
    runs_root="${candidate_dir}/runs"
    command=(
        env -u CUDA_VISIBLE_DEVICES PYTHONPATH=src
        "${PYTHON_BIN}" -m rl_training.train_grpo_macorag
        --config "${CONFIG}"
        --max-total-samples "${SAMPLE_COUNT}"
        --per-device-train-batch-size "${batch_size}"
        --reference-per-device-batch-size "${REFERENCE_BATCH_SIZE}"
        --gpu-indices "${TRAIN_GPU}"
        --output-root "${runs_root}"
        --save-steps 0
        --disable-tqdm
        --log-all-group-rollouts
    )

    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
        continue
    fi

    mkdir -p "${candidate_dir}"
    : > "${candidate_dir}/gpu_memory_mib.txt"
    printf 'Running action batch size %s; logs: %s\n' "${batch_size}" "${candidate_dir}/train.log"
    "${command[@]}" > "${candidate_dir}/train.log" 2>&1 &
    train_pid=$!

    (
        while kill -0 "${train_pid}" 2>/dev/null; do
            nvidia-smi --query-gpu=memory.used --id="${TRAIN_GPU}" \
                --format=csv,noheader,nounits 2>/dev/null \
                | tr -d ' ' >> "${candidate_dir}/gpu_memory_mib.txt"
            sleep 1
        done
    ) &
    monitor_pid=$!

    wait "${train_pid}"
    exit_code=$?
    wait "${monitor_pid}" 2>/dev/null || true

    if [[ ${exit_code} -eq 0 ]]; then
        status="success"
        failure_reason=""
    elif rg -qi 'CUDA out of memory|torch\.OutOfMemoryError' "${candidate_dir}/train.log"; then
        status="oom"
        failure_reason="CUDA out of memory"
    else
        status="failed"
        failure_reason="training process exited non-zero"
    fi

    STATUS="${status}" EXIT_CODE="${exit_code}" FAILURE_REASON="${failure_reason}" \
        REFERENCE_BATCH_SIZE_VALUE="${REFERENCE_BATCH_SIZE}" CANDIDATE_DIR="${candidate_dir}" \
        SAMPLE_COUNT_VALUE="${SAMPLE_COUNT}" \
        "${PYTHON_BIN}" -c \
        'import json, os, pathlib; p=pathlib.Path(os.environ["CANDIDATE_DIR"])/"status.json"; p.write_text(json.dumps({"status": os.environ["STATUS"], "exit_code": int(os.environ["EXIT_CODE"]), "failure_reason": os.environ["FAILURE_REASON"], "reference_batch_size": int(os.environ["REFERENCE_BATCH_SIZE_VALUE"]), "sample_count": int(os.environ["SAMPLE_COUNT_VALUE"])}, indent=2)+"\n", encoding="utf-8")'

    printf 'Batch size %s finished with status=%s exit_code=%s\n' "${batch_size}" "${status}" "${exit_code}"
    if [[ "${status}" == "oom" ]]; then
        printf 'Stopping larger candidates after OOM at batch size %s.\n' "${batch_size}"
        break
    fi
done

if [[ "${DRY_RUN}" != "1" ]]; then
    "${PYTHON_BIN}" scripts/summarize_grpo_batch_benchmark.py "${BENCHMARK_ROOT}"
fi
