#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/generate_teacher_sft.yml}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${MACORAG_LAUNCH_DRY_RUN:-0}" == "1" ]]; then
  printf '[validate-pipeline] teacher_config=%s config_root=%s\n' \
    "${CONFIG_PATH}" "${REPO_ROOT}/config"
  exit 0
fi

"${PYTHON:-python}" - "${REPO_ROOT}" "${CONFIG_PATH}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

from prompt_config import load_prompt_contract


root = Path(sys.argv[1])
teacher_path = Path(sys.argv[2])
if not teacher_path.is_absolute():
    teacher_path = root / teacher_path


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"Missing pipeline config: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Pipeline config must be a mapping: {path}")
    return payload


contract = load_prompt_contract(root / "config" / "prompts.yml")
configs = {
    "generate_teacher_sft.yml": load(teacher_path),
    "train_sft.yml": load(root / "config" / "train_sft.yml"),
    "train_grpo.yml": load(root / "config" / "train_grpo.yml"),
    "eval_macorag.yml": load(root / "config" / "eval_macorag.yml"),
}
for name, config in configs.items():
    if int(config.get("max_rounds", -1)) != 4:
        raise SystemExit(f"{name}: max_rounds must be 4")
    if int(config.get("retrieval_top_k", 5)) != 5:
        raise SystemExit(f"{name}: retrieval_top_k must be 5")
    if config.get("prompt_config_path") != "config/prompts.yml":
        raise SystemExit(f"{name}: prompt_config_path mismatch")

teacher = configs["generate_teacher_sft.yml"]
if teacher.get("retrieval_backend") != "e5_faiss":
    raise SystemExit("teacher retrieval backend mismatch")
if teacher.get("embedding_model") != "intfloat/e5-base-v2":
    raise SystemExit("teacher embedding model mismatch")
if configs["train_sft.yml"].get("data_root") != teacher.get("output_dir"):
    raise SystemExit("teacher output and SFT data roots do not match")

extract_train = load(root / "config" / "extract_train.yml")
extract_eval = load(root / "config" / "extract_eval.yml")
retrieval_train = load(root / "config" / "retrieval_train.yml")
retrieval_eval = load(root / "config" / "retrieval_eval.yml")
retrieval_teacher = load(root / "config" / "retrieval_teacher.yml")
grpo = configs["train_grpo.yml"]
evaluation = configs["eval_macorag.yml"]

contracts = (
    ("train extraction/runtime", extract_train.get("output_root"), grpo.get("rl_data_root")),
    ("train retrieval data", retrieval_train.get("data_root"), grpo.get("rl_data_root")),
    ("train retrieval index", retrieval_train.get("retrieval_root"), grpo.get("retrieval_root")),
    ("eval extraction/runtime", extract_eval.get("output_root"), evaluation.get("data_root")),
    ("eval retrieval data", retrieval_eval.get("data_root"), evaluation.get("data_root")),
    ("eval retrieval index", retrieval_eval.get("retrieval_root"), evaluation.get("retrieval_root")),
    ("teacher retrieval data", retrieval_teacher.get("data_root"), teacher.get("source_root")),
    ("teacher retrieval index", retrieval_teacher.get("retrieval_root"), teacher.get("retrieval_root")),
)
for name, producer, consumer in contracts:
    if producer != consumer:
        raise SystemExit(f"{name} mismatch: {producer!r} != {consumer!r}")

print(
    json.dumps(
        {
            "status": "ok",
            "prompt_contract_version": contract.version,
            "prompt_contract_fingerprint": contract.fingerprint,
        },
        indent=2,
    )
)
PY
