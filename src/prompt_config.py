from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping


DEFAULT_PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts.yml"
EXPECTED_PROMPT_CONTRACT_VERSION = "macorag-rag-v2"
PROMPT_ROLES = ("query_retriever", "evidence_updater", "answer_generator")


@dataclass(frozen=True)
class PromptContract:
    version: str
    system_prompts: Mapping[str, str]
    instructions: Mapping[str, Any]
    fingerprint: str
    source_path: Path
    payload: Mapping[str, Any]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to load prompt config.") from exc
    if not path.exists():
        raise SystemExit(f"Prompt config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid prompt config format at {path}: expected a mapping.")
    return payload


def _canonical_fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_prompt_contract(path: str | Path | None = None) -> PromptContract:
    prompt_path = (Path(path) if path is not None else DEFAULT_PROMPT_CONFIG_PATH).resolve()
    payload = _load_yaml(prompt_path)
    version = str(payload.get("prompt_contract_version") or "").strip()
    if version != EXPECTED_PROMPT_CONTRACT_VERSION:
        raise SystemExit(
            f"Unsupported prompt contract at {prompt_path}: expected "
            f"{EXPECTED_PROMPT_CONTRACT_VERSION!r}, got {version!r}."
        )
    raw_system_prompts = payload.get("system_prompts")
    if not isinstance(raw_system_prompts, dict):
        raw_system_prompts = {}
    system_prompts = {role: str(raw_system_prompts.get(role) or "").strip() for role in PROMPT_ROLES}
    missing = [role for role, value in system_prompts.items() if not value]
    if missing:
        raise SystemExit(f"Prompt contract missing system prompts at {prompt_path}: {', '.join(missing)}")
    instructions = payload.get("instructions")
    if not isinstance(instructions, dict):
        raise SystemExit(f"Prompt contract missing instructions mapping: {prompt_path}")
    semantic_payload = {
        "prompt_contract_version": version,
        "system_prompts": system_prompts,
        "instructions": instructions,
    }
    return PromptContract(
        version=version,
        system_prompts=MappingProxyType(system_prompts),
        instructions=MappingProxyType(instructions),
        fingerprint=_canonical_fingerprint(semantic_payload),
        source_path=prompt_path,
        payload=MappingProxyType(semantic_payload),
    )


def system_prompt_for(role: str, contract: PromptContract | None = None) -> str:
    prompt_contract = contract or load_prompt_contract()
    normalized_role = str(getattr(role, "value", role))
    if normalized_role not in prompt_contract.system_prompts:
        raise ValueError(f"Unknown prompt role: {normalized_role}")
    return prompt_contract.system_prompts[normalized_role]


def load_system_prompt(path: str | Path | None = None) -> str:
    """Compatibility wrapper for legacy single-system-prompt callers."""
    return system_prompt_for("answer_generator", load_prompt_contract(path))


DEFAULT_PROMPT_CONTRACT = load_prompt_contract()
DEFAULT_SYSTEM_PROMPT = system_prompt_for("answer_generator", DEFAULT_PROMPT_CONTRACT)
