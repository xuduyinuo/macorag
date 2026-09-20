from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


ROLE_KEYS = ("query_retriever", "evidence_updater", "answer_generator")


@dataclass(frozen=True)
class PolicyPromptContract:
    source_path: Path
    version: str
    fingerprint: str
    tags: dict[str, str]
    system_prompts: dict[str, str]
    few_shots: dict[str, list[dict[str, Any]]]

    def system_prompt(self, role: str) -> str:
        return self.system_prompts[role]


def load_policy_prompt_contract(path: str | Path) -> PolicyPromptContract:
    source = Path(path).resolve()
    raw = source.read_bytes()
    payload = yaml.safe_load(raw) or {}
    version = str(payload.get("prompt_contract_version") or "").strip()
    tags = dict(payload.get("tags") or {})
    systems = dict(payload.get("system_prompts") or {})
    few_shots = dict(payload.get("few_shots") or {})
    if not version:
        raise ValueError(f"Missing prompt_contract_version in {source}")
    for role in ROLE_KEYS:
        if not str(tags.get(role) or "").strip():
            raise ValueError(f"Missing tag for {role} in {source}")
        if not str(systems.get(role) or "").strip():
            raise ValueError(f"Missing system prompt for {role} in {source}")
        examples = few_shots.get(role)
        if not isinstance(examples, list) or len(examples) < 3:
            raise ValueError(f"At least three few-shot examples are required for {role}")
        for index, example in enumerate(examples):
            if not isinstance(example, dict) or not example.get("user") or not example.get("assistant"):
                raise ValueError(f"Invalid {role} few-shot example at index {index}")
            if "rationale" in str(example["assistant"]).lower():
                raise ValueError(f"rationale is forbidden in {role} few-shot targets")
    return PolicyPromptContract(
        source_path=source,
        version=version,
        fingerprint=hashlib.sha256(raw).hexdigest(),
        tags={key: str(value) for key, value in tags.items()},
        system_prompts={key: str(value).strip() for key, value in systems.items()},
        few_shots={key: list(value) for key, value in few_shots.items()},
    )


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def tagged_target(contract: PolicyPromptContract, role: str, payload: dict[str, Any]) -> str:
    tag = contract.tags[role]
    return f"<{tag}>{compact_json(payload)}</{tag}>"


def few_shot_messages(
    contract: PolicyPromptContract,
    role: str,
    *,
    final_round: bool = False,
) -> list[dict[str, str]]:
    examples = contract.few_shots[role]
    if role == "answer_generator":
        # Normal rounds see both abstention and successful-answer boundaries.
        # Final rounds additionally see the forced supported-answer example.
        examples = [
            item for item in examples
            if final_round or not bool(item.get("final_round", False))
        ]
    messages: list[dict[str, str]] = []
    for item in examples:
        messages.append({"role": "user", "content": str(item["user"]).strip()})
        messages.append({"role": "assistant", "content": str(item["assistant"]).strip()})
    return messages


def policy_messages(
    contract: PolicyPromptContract,
    role: str,
    user_prompt: str,
    *,
    final_round: bool = False,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": contract.system_prompt(role)},
        *few_shot_messages(contract, role, final_round=final_round),
        {"role": "user", "content": user_prompt.strip()},
    ]
