from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .mappo_types import AgentRole, RAGState


SYSTEM_PROMPTS = {
    AgentRole.QUERY: "You are the query-retriever role in a cooperative multi-agent RAG system. Output exactly one <query-retriever> tag containing valid JSON. Do not output analysis, Markdown, or any other tag.",
    AgentRole.EVIDENCE: "You are the evidence-updater role in a cooperative multi-agent RAG system. Output exactly one <update-evidence> tag containing valid JSON. Do not output analysis, Markdown, or any other tag.",
    AgentRole.ANSWER: "You are the answer-generator role in a cooperative multi-agent RAG system. Output exactly one <answer> tag containing valid JSON with can_answer, answer, and rationale. Do not output analysis, Markdown, or any other tag.",
}

OUTPUT_CONTRACT_MARKER = "<output-contract>"
_TRUNCATION_MARKER = " ...[truncated]"


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _clip_text(value: Any, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    if max_chars <= len(_TRUNCATION_MARKER):
        return _TRUNCATION_MARKER[:max_chars]
    keep = max(0, max_chars - len(_TRUNCATION_MARKER))
    return text[:keep].rstrip() + _TRUNCATION_MARKER


def _spread_items(items: list[Any], limit: int) -> list[Any]:
    """Keep both early and recent multi-hop context when a list is compacted."""
    if len(items) <= limit:
        return items
    first = (limit + 1) // 2
    recent = limit - first
    return items[:first] if recent == 0 else [*items[:first], *items[-recent:]]


def _compact_passage(item: Any, text_chars: int) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {"text": _clip_text(item, text_chars)}
    compact: dict[str, Any] = {}
    for key in ("passage_id", "title"):
        if key in item:
            compact[key] = item[key]
    compact["text"] = _clip_text(item.get("text", ""), text_chars)
    return compact


def _compact_state(
    state: RAGState,
    *,
    role: AgentRole,
    max_evidence_items: int,
    max_history_items: int,
    evidence_text_chars: int,
) -> dict[str, Any]:
    evidence = _spread_items(list(state.evidence), max_evidence_items)
    history = _spread_items(list(state.retrieval_history), max_history_items)
    # Query/evidence agents mainly need coverage and IDs. The evidence updater
    # spends its text budget on the fresh observation; the answer agent gets
    # the largest accumulated-evidence allowance for synthesis.
    role_text_chars = {
        AgentRole.QUERY: min(evidence_text_chars, 120),
        AgentRole.EVIDENCE: min(evidence_text_chars, 80),
        AgentRole.ANSWER: evidence_text_chars,
    }[role]
    compact_history = []
    for item in history:
        if not isinstance(item, dict):
            compact_history.append({"query": _clip_text(item, 120)})
            continue
        compact_history.append({
            "query": _clip_text(item.get("query", ""), 120),
            "sub_goal": _clip_text(item.get("sub_goal", ""), 120),
            "passage_ids": list(item.get("passage_ids", []))[:max_evidence_items],
        })
    return {
        "current_sub_goal": _clip_text(state.sub_goal, 160) or None,
        "evidence": [_compact_passage(item, role_text_chars) for item in evidence],
        "retrieval_history": compact_history,
        "evidence_count": len(state.evidence),
        "retrieval_count": len(state.retrieval_history),
        "round_index": state.round_index,
    }


def _compact_observation(
    observation: dict[str, Any] | None,
    *,
    max_items: int,
    text_chars: int,
) -> dict[str, Any]:
    value = observation or {"passages": []}
    passages = list(value.get("passages", []))
    return {
        "query": _clip_text(value.get("query", ""), 160),
        "passages": [
            _compact_passage(item, text_chars)
            for item in passages[:max_items]
        ],
        "passage_count": len(passages),
    }


def build_prompt(
    role: AgentRole,
    *,
    question: str,
    state: RAGState,
    observation: dict[str, Any] | None = None,
    final_round: bool = False,
    max_evidence_items: int = 6,
    max_history_items: int = 3,
    evidence_text_chars: int = 160,
    observation_text_chars: int = 200,
) -> str:
    state_json = _dump(_compact_state(
        state,
        role=role,
        max_evidence_items=max_evidence_items,
        max_history_items=max_history_items,
        evidence_text_chars=evidence_text_chars,
    ))
    if role is AgentRole.QUERY:
        return (
            "Plan the next non-repeated knowledge-base query.\n"
            f"Question: {question}\n<state>{state_json}</state>\n"
            f"{OUTPUT_CONTRACT_MARKER}\n"
            'Return exactly one tag: <query-retriever>{"sub_goal":"...","query":"..."}</query-retriever>\n'
            "Do not return analysis, Markdown, or any other text.\n"
            "</output-contract>"
        )
    if role is AgentRole.EVIDENCE:
        compact_observation = _compact_observation(
            observation,
            max_items=max_evidence_items,
            text_chars=observation_text_chars,
        )
        return (
            "Select only useful passage IDs from the latest observation.\n"
            f"Question: {question}\n<state>{state_json}</state>\n"
            f"<observation>{_dump(compact_observation)}</observation>\n"
            f"{OUTPUT_CONTRACT_MARKER}\n"
            'Return exactly one tag: <update-evidence>{"selected_passage_ids":[],"rationale":"..."}</update-evidence>\n'
            "Do not return analysis, Markdown, or any other text.\n"
            "</output-contract>"
        )
    rule = (
        "This is the final round. Set can_answer=true and return a concise non-empty answer. If evidence is insufficient, make the best supported guess and start rationale with fallback_guess:."
        if final_round else
        "Set can_answer=true only when selected evidence is sufficient; otherwise use false and null."
    )
    if final_round:
        examples = (
            "Final-round example (a refusal is not allowed):\n"
            '<answer>{"can_answer":true,"answer":"best supported answer",'
            '"rationale":"fallback_guess: strongest available evidence"}</answer>'
        )
    else:
        examples = (
            "Example when evidence is insufficient:\n"
            '<answer>{"can_answer":false,"answer":null,'
            '"rationale":"more evidence is needed"}</answer>\n'
            "Example when evidence is sufficient:\n"
            '<answer>{"can_answer":true,"answer":"London",'
            '"rationale":"selected evidence directly supports London"}</answer>'
        )
    return (
        f"Answer using only accumulated evidence. {rule}\n"
        f"Question: {question}\n<state>{state_json}</state>\n"
        f"{OUTPUT_CONTRACT_MARKER}\n"
        f"{examples}\n"
        "Now return exactly one <answer> tagged JSON object and nothing else.\n"
        "</output-contract>"
    )


def protocol_metrics(episodes: list[Any], *, max_parse_failure_rate: float,
                     max_missing_answer_tag_rate: float,
                     min_final_compliance_rate: float) -> dict[str, Any]:
    total = len(episodes)
    if total == 0:
        return {"count": 0, "parse_failure_rate": 0.0, "missing_answer_tag_rate": 0.0,
                "final_compliance_rate": 0.0, "checkpoint_eligible": False}
    parse_failures = sum(bool(item.parse_errors) for item in episodes)
    missing_answer = sum(
        any("Missing required tag: answer" in error for error in item.parse_errors)
        for item in episodes
    )
    final_compliant = sum(
        not item.parse_errors and isinstance(item.final_answer, str) and bool(item.final_answer.strip())
        for item in episodes
    )
    parse_rate = parse_failures / total
    missing_rate = missing_answer / total
    final_rate = final_compliant / total
    return {
        "count": total,
        "parse_failure_rate": parse_rate,
        "missing_answer_tag_rate": missing_rate,
        "final_compliance_rate": final_rate,
        "checkpoint_eligible": (
            parse_rate <= max_parse_failure_rate
            and missing_rate <= max_missing_answer_tag_rate
            and final_rate >= min_final_compliance_rate
        ),
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
        latest_rate = sum(bool(item.parse_errors) for item in episodes) / max(1, len(episodes))
        while len(self.pending) >= self.window_size:
            window = self.pending[:self.window_size]
            del self.pending[:self.window_size]
            latest_rate = sum(bool(item.parse_errors) for item in window) / self.window_size
            self.completed_windows += 1
            if latest_rate > self.max_parse_failure_rate:
                self.consecutive_bad_windows += 1
                if self.consecutive_bad_windows == self.bad_windows_to_warn:
                    should_warn = True
            else:
                self.consecutive_bad_windows = 0
        return {
            "window_parse_failure_rate": latest_rate,
            "completed_windows": self.completed_windows,
            "consecutive_bad_windows": self.consecutive_bad_windows,
            "should_warn": should_warn,
        }


def parse_action(text: str, role: AgentRole, *, final_round: bool = False) -> dict[str, Any]:
    tag = {
        AgentRole.QUERY: "query-retriever",
        AgentRole.EVIDENCE: "update-evidence",
        AgentRole.ANSWER: "answer",
    }[role]
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.DOTALL)
    if match is None:
        raise ValueError(f"Missing required tag: {tag}")
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in tag {tag}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Tag {tag} must contain a JSON object")
    if role is AgentRole.QUERY:
        for key in ("sub_goal", "query"):
            if not isinstance(value.get(key), str) or not value[key].strip():
                raise ValueError(f"query_retriever.{key} must be a non-empty string")
    elif role is AgentRole.EVIDENCE:
        ids = value.get("selected_passage_ids")
        if not isinstance(ids, list) or any(isinstance(x, bool) or not isinstance(x, int) for x in ids):
            raise ValueError("selected_passage_ids must be a list of integers")
    else:
        if not isinstance(value.get("can_answer"), bool):
            raise ValueError("answer.can_answer must be a JSON boolean")
        answer = value.get("answer")
        if value["can_answer"] and (not isinstance(answer, str) or not answer.strip()):
            raise ValueError("answer.answer must be non-empty when can_answer=true")
        if final_round and not value["can_answer"]:
            raise ValueError("final round requires can_answer=true")
    return value


def central_state(
    state: RAGState,
    *,
    role: AgentRole,
    observation: dict[str, Any] | None = None,
    max_rounds: int,
) -> dict[str, Any]:
    return {
        "role": role.value,
        "question": state.question,
        "sub_goal": state.sub_goal,
        "evidence": state.evidence,
        "retrieval_history": state.retrieval_history,
        "observation": observation or {"passages": []},
        "round_index": state.round_index,
        "max_rounds": max_rounds,
    }
