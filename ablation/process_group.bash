#!/usr/bin/env bash

# Stop a service and every child process that inherited its process group.
# The caller must launch the service with `setsid` before using this helper.
stop_ablation_process_group() {
  local leader_pid="$1"
  local label="${2:-service}"
  local timeout_seconds="${VLLM_STOP_TIMEOUT_SECONDS:-30}"
  local process_group=""
  local deadline=0

  if [[ ! "${leader_pid}" =~ ^[0-9]+$ ]] || (( leader_pid <= 1 )); then
    printf '[ablation-cleanup] refusing invalid %s pid=%s\n' "${label}" "${leader_pid}" >&2
    return 1
  fi

  process_group="$(ps -o pgid= -p "${leader_pid}" 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "${process_group}" ]]; then
    wait "${leader_pid}" 2>/dev/null || true
    return 0
  fi
  if [[ "${process_group}" != "${leader_pid}" ]]; then
    printf '[ablation-cleanup] refusing non-isolated %s pid=%s pgid=%s\n' \
      "${label}" "${leader_pid}" "${process_group}" >&2
    kill -TERM "${leader_pid}" 2>/dev/null || true
    wait "${leader_pid}" 2>/dev/null || true
    return 1
  fi

  printf '[ablation-cleanup] stopping %s pid=%s pgid=%s\n' \
    "${label}" "${leader_pid}" "${process_group}"
  kill -TERM -- "-${process_group}" 2>/dev/null || true
  deadline=$((SECONDS + timeout_seconds))
  while kill -0 -- "-${process_group}" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 1
  done
  if kill -0 -- "-${process_group}" 2>/dev/null; then
    printf '[ablation-cleanup] forcing %s shutdown after %ss\n' \
      "${label}" "${timeout_seconds}" >&2
    kill -KILL -- "-${process_group}" 2>/dev/null || true
  fi
  wait "${leader_pid}" 2>/dev/null || true
  printf '[ablation-cleanup] stopped %s\n' "${label}"
}
