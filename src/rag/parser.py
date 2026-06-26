from __future__ import annotations

import json
import re
from typing import Any

from .schema import AgentRole, ParsedAction


def _tag_payload(text: str, tag: str) -> dict[str, Any]:
    pattern = re.compile(rf"<{re.escape(tag)}>\s*(.*?)\s*</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"Missing required tag: {tag}")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in tag {tag}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Tag {tag} must contain a JSON object.")
    return payload


def parse_action_text(text: str, role: AgentRole | str) -> ParsedAction:
    role = AgentRole(role)
    if role == AgentRole.QUERY_RETRIEVER:
        query_retriever = _tag_payload(text, "query-retriever")
        if not query_retriever.get("sub_goal"):
            raise ValueError("Missing required field: query_retriever.sub_goal")
        if "query" not in query_retriever:
            raise ValueError("Missing required field: query_retriever.query")
        return ParsedAction(role=role, query_retriever=query_retriever)

    if role == AgentRole.EVIDENCE_UPDATER:
        update_evidence = _tag_payload(text, "update-evidence")
        if "selected_passage_ids" not in update_evidence:
            raise ValueError("Missing required field: update_evidence.selected_passage_ids")
        return ParsedAction(role=role, update_evidence=update_evidence)

    if role != AgentRole.ANSWER_GENERATOR:
        raise ValueError(f"Unsupported role: {role}")
    answer = _tag_payload(text, "answer")
    if "can_answer" not in answer:
        raise ValueError("Missing required field: answer.can_answer")
    if "answer" not in answer:
        raise ValueError("Missing required field: answer.answer")
    return ParsedAction(role=role, answer=answer)
