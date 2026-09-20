from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .mappo_types import AgentRole, RAGState


ROLE_KEYS = ("query_retriever", "evidence_updater", "answer_generator")
OUTPUT_CONTRACT_MARKER = "macorag-policy-v3"


@dataclass(frozen=True)
class PolicyPromptContract:
    source_path: Path
    version: str
    fingerprint: str
    tags: dict[str, str]
    system_prompts: dict[str, str]
    few_shots: dict[str, list[dict[str, Any]]]


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
        if not isinstance(few_shots.get(role), list) or len(few_shots[role]) < 3:
            raise ValueError(f"At least three few-shot examples are required for {role}")
    return PolicyPromptContract(
        source_path=source,
        version=version,
        fingerprint=hashlib.sha256(raw).hexdigest(),
        tags={key: str(value) for key, value in tags.items()},
        system_prompts={key: str(value).strip() for key, value in systems.items()},
        few_shots={key: list(value) for key, value in few_shots.items()},
    )


PROMPT_CONTRACT = load_policy_prompt_contract(Path(__file__).with_name("policy_prompts.yml"))
SYSTEM_PROMPTS = {
    AgentRole.QUERY: PROMPT_CONTRACT.system_prompts[AgentRole.QUERY.value],
    AgentRole.EVIDENCE: PROMPT_CONTRACT.system_prompts[AgentRole.EVIDENCE.value],
    AgentRole.ANSWER: PROMPT_CONTRACT.system_prompts[AgentRole.ANSWER.value],
}


def validate_prompt_contract(
    configured_path: str | Path, adapter_path: str | Path,
    expected_version: str,
) -> PolicyPromptContract:
    contract = load_policy_prompt_contract(configured_path)
    if contract.version != expected_version:
        raise ValueError(
            f"RL prompt version mismatch: {contract.version!r} != {expected_version!r}"
        )
    if contract.fingerprint != PROMPT_CONTRACT.fingerprint:
        raise ValueError("Configured RL prompt file differs from bundled policy_prompts.yml")
    root = Path(adapter_path)
    candidates = (root / "prompt_contract.json", root / "actor" / "prompt_contract.json")
    manifest_path = next((path for path in candidates if path.is_file()), None)
    if manifest_path is None:
        raise FileNotFoundError(
            "SFT adapter is missing prompt_contract.json: "
            + ", ".join(str(path) for path in candidates)
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("prompt_contract_version") != contract.version:
        raise ValueError("SFT adapter and RL prompt contract versions differ")
    if manifest.get("prompt_contract_fingerprint") != contract.fingerprint:
        raise ValueError("SFT adapter and RL prompt contract fingerprints differ")
    return contract


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def role_messages(role: AgentRole, user_prompt: str, *, final_round: bool) -> list[dict[str, str]]:
    examples = PROMPT_CONTRACT.few_shots[role.value]
    if role is AgentRole.ANSWER:
        examples = [item for item in examples if final_round or not bool(item.get("final_round", False))]
    messages = [{"role": "system", "content": SYSTEM_PROMPTS[role]}]
    for item in examples:
        messages.extend((
            {"role": "user", "content": str(item["user"]).strip()},
            {"role": "assistant", "content": str(item["assistant"]).strip()},
        ))
    messages.append({"role": "user", "content": user_prompt.strip()})
    return messages


def _clip(value: Any, chars: int) -> str:
    # Match run_macorag_eval.py: strip boundaries and take a plain character
    # prefix without inserting a training-only truncation marker.
    return str(value or "").strip()[:chars]


def _selected_evidence(state: RAGState, limit: int, text_chars: int) -> list[dict[str, str]]:
    items = state.evidence
    if len(items) > limit:
        head = (limit + 1) // 2
        items = [*items[:head], *items[-(limit - head):]]
    return [
        {"title": str(item.get("title") or ""), "text": _clip(item.get("text"), text_chars)}
        for item in items
    ]


def build_prompt(
    role: AgentRole, *, question: str, state: RAGState,
    observation: dict[str, Any] | None = None, final_round: bool = False,
    max_rounds: int = 4, max_evidence_items: int = 6,
    max_history_items: int = 3, evidence_text_chars: int = 160,
    observation_text_chars: int = 200, **_: Any,
) -> str:
    evidence = _selected_evidence(state, max_evidence_items, evidence_text_chars)
    if role is AgentRole.QUERY:
        history = [
            {"query": str(item.get("query") or "")}
            for item in state.retrieval_history[-max_history_items:]
        ]
        visible = {
            "current_sub_goal": state.sub_goal or None,
            "selected_evidence": evidence,
            "retrieval_history": history,
        }
        return (
            f"Question: {question}\nRound: {state.round_index + 1} of {max_rounds}\n"
            f"Current state:\n{compact_json(visible)}"
        )
    if role is AgentRole.EVIDENCE:
        passages = []
        for item in list((observation or {}).get("passages", []))[:max_evidence_items]:
            passages.append({
                "passage_id": f"P{int(item['passage_id'])}",
                "title": str(item.get("title") or ""),
                "text": _clip(item.get("text"), observation_text_chars),
            })
        return (
            f"Question: {question}\nCurrent sub-goal: {state.sub_goal}\n"
            f"Accumulated selected evidence: {compact_json(evidence)}\n"
            f"Latest observation:\n{compact_json(passages)}"
        )
    marker = " (final round)" if final_round else ""
    return (
        f"Question: {question}\nRound: {state.round_index + 1} of {max_rounds}{marker}\n"
        f"Accumulated selected evidence:\n{compact_json(evidence)}"
    )


def parse_action(text: str, role: AgentRole, *, final_round: bool = False) -> dict[str, Any]:
    tag = PROMPT_CONTRACT.tags[role.value]
    matches = re.findall(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", text, re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one required tag: {tag}")
    try:
        value = json.loads(matches[0])
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in tag {tag}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Tag {tag} must contain a JSON object")
    if role is AgentRole.QUERY:
        if set(value) != {"sub_goal", "query"}:
            raise ValueError("query action must contain exactly sub_goal and query")
        if any(not isinstance(value[key], str) or not value[key].strip() for key in value):
            raise ValueError("query action fields must be non-empty strings")
    elif role is AgentRole.EVIDENCE:
        if set(value) != {"selected_passage_ids"}:
            raise ValueError("evidence action must contain exactly selected_passage_ids")
        ids = value["selected_passage_ids"]
        if not isinstance(ids, list) or any(not isinstance(item, str) or not re.fullmatch(r"P\d+", item) for item in ids):
            raise ValueError("selected_passage_ids must be a list of P-prefixed pointers")
        normalized = [int(item[1:]) for item in ids]
        if len(normalized) != len(set(normalized)):
            raise ValueError("selected_passage_ids must not contain duplicates")
        value["selected_passage_ids"] = normalized
    else:
        if set(value) != {"can_answer", "answer"} or not isinstance(value.get("can_answer"), bool):
            raise ValueError("answer action must contain exactly can_answer and answer")
        answer = value.get("answer")
        if value["can_answer"] and (not isinstance(answer, str) or not answer.strip()):
            raise ValueError("answer must be non-empty when can_answer=true")
        if not value["can_answer"] and answer is not None:
            raise ValueError("answer must be null when can_answer=false")
        if final_round and not value["can_answer"]:
            raise ValueError("final round requires can_answer=true")
    return value


def central_state(
    state: RAGState, *, role: AgentRole,
    observation: dict[str, Any] | None = None, max_rounds: int,
) -> dict[str, Any]:
    return {
        "role": role.value, "question": state.question, "sub_goal": state.sub_goal,
        "evidence": state.evidence, "retrieval_history": state.retrieval_history,
        "observation": observation or {"passages": []},
        "round_index": state.round_index, "max_rounds": max_rounds,
    }


def protocol_metrics(episodes: list[Any], *, max_parse_failure_rate: float,
                     max_missing_answer_tag_rate: float,
                     min_final_compliance_rate: float) -> dict[str, Any]:
    total = len(episodes)
    if not total:
        return {"count": 0, "parse_failure_rate": 0.0, "missing_answer_tag_rate": 0.0,
                "final_compliance_rate": 0.0, "checkpoint_eligible": False}
    parse_failures = sum(bool(item.parse_errors) for item in episodes)
    missing = sum(any("required tag: answer" in error for error in item.parse_errors) for item in episodes)
    compliant = sum(not item.parse_errors and bool(str(item.final_answer or "").strip()) for item in episodes)
    parse_rate, missing_rate, final_rate = parse_failures / total, missing / total, compliant / total
    return {
        "count": total, "parse_failure_rate": parse_rate,
        "missing_answer_tag_rate": missing_rate, "final_compliance_rate": final_rate,
        "checkpoint_eligible": parse_rate <= max_parse_failure_rate
        and missing_rate <= max_missing_answer_tag_rate
        and final_rate >= min_final_compliance_rate,
    }


@dataclass
class ProtocolWindowMonitor:
    window_size: int
    max_parse_failure_rate: float
    bad_windows_to_warn: int = 2
    pending: list[Any] = field(default_factory=list)
    consecutive_bad_windows: int = 0
    completed_windows: int = 0

    def add(self, episodes: list[Any]) -> dict[str, Any]:
        self.pending.extend(episodes)
        should_warn = False
        rate = sum(bool(item.parse_errors) for item in episodes) / max(1, len(episodes))
        while len(self.pending) >= self.window_size:
            window, self.pending = self.pending[:self.window_size], self.pending[self.window_size:]
            rate = sum(bool(item.parse_errors) for item in window) / self.window_size
            self.completed_windows += 1
            if rate > self.max_parse_failure_rate:
                self.consecutive_bad_windows += 1
                should_warn = self.consecutive_bad_windows == self.bad_windows_to_warn
            else:
                self.consecutive_bad_windows = 0
        return {"window_parse_failure_rate": rate, "completed_windows": self.completed_windows,
                "consecutive_bad_windows": self.consecutive_bad_windows, "should_warn": should_warn}
