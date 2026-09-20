from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptContract:
    version: str
    system_prompts: dict[str, str]
    instructions: dict[str, str]
    fingerprint: str
    source_path: Path


def load_prompt_contract(path: str | Path) -> PromptContract:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load the SFT-v2 prompt contract") from exc
    source_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    systems = payload.get("system_prompts") or {}
    instructions = payload.get("instructions") or {}
    required_systems = {"query_retriever", "evidence_updater", "answer_generator"}
    required_instructions = {"query_retriever", "evidence_updater", "answer_normal", "answer_final"}
    if not required_systems.issubset(systems) or not required_instructions.issubset(instructions):
        raise ValueError("Prompt contract is missing one or more role prompts")
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return PromptContract(
        version=str(payload.get("prompt_contract_version") or ""),
        system_prompts={str(key): str(value).strip() for key, value in systems.items()},
        instructions={str(key): str(value).strip() for key, value in instructions.items()},
        fingerprint=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        source_path=source_path,
    )


def _tag(name: str, payload: Any) -> str:
    return f"<{name}>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</{name}>"


def query_messages(contract: PromptContract, *, question: str, state: dict[str, Any]) -> list[dict[str, str]]:
    user = "\n".join(
        [
            contract.instructions["query_retriever"],
            _tag("question", question),
            _tag("state", state),
        ]
    )
    return [
        {"role": "system", "content": contract.system_prompts["query_retriever"]},
        {"role": "user", "content": user},
    ]


def evidence_messages(
    contract: PromptContract,
    *,
    question: str,
    state: dict[str, Any],
    observation: dict[str, Any],
) -> list[dict[str, str]]:
    user = "\n".join(
        [
            contract.instructions["evidence_updater"],
            _tag("question", question),
            _tag("state", state),
            _tag("observation", observation),
        ]
    )
    return [
        {"role": "system", "content": contract.system_prompts["evidence_updater"]},
        {"role": "user", "content": user},
    ]


def answer_messages(
    contract: PromptContract,
    *,
    question: str,
    state: dict[str, Any],
    round_index: int,
    max_rounds: int,
) -> list[dict[str, str]]:
    final_round = round_index + 1 >= max_rounds
    instruction = contract.instructions["answer_final" if final_round else "answer_normal"]
    user = "\n".join(
        [
            instruction,
            f"Round: {round_index + 1}/{max_rounds}",
            _tag("question", question),
            _tag("state", state),
        ]
    )
    return [
        {"role": "system", "content": contract.system_prompts["answer_generator"]},
        {"role": "user", "content": user},
    ]

