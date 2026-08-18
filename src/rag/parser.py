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


def parse_can_answer(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise ValueError("answer.can_answer must be a JSON boolean.")


def _require_string(
    payload: dict[str, Any],
    key: str,
    *,
    field: str,
    allow_empty: bool,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise ValueError(f"{field} must be a{' non-empty' if not allow_empty else ''} string.")
    return value


def _validate_optional_string(payload: dict[str, Any], key: str, *, field: str) -> None:
    if key in payload and not isinstance(payload[key], str):
        raise ValueError(f"{field} must be a string when provided.")


def parse_action_text(text: str, role: AgentRole | str) -> ParsedAction:
    role = AgentRole(role)
    if role == AgentRole.QUERY_RETRIEVER:
        query_retriever = _tag_payload(text, "query-retriever")
        if "sub_goal" not in query_retriever:
            raise ValueError("Missing required field: query_retriever.sub_goal")
        if "query" not in query_retriever:
            raise ValueError("Missing required field: query_retriever.query")
        _require_string(
            query_retriever,
            "sub_goal",
            field="query_retriever.sub_goal",
            allow_empty=False,
        )
        _require_string(
            query_retriever,
            "query",
            field="query_retriever.query",
            allow_empty=True,
        )
        return ParsedAction(role=role, query_retriever=query_retriever)

    if role == AgentRole.EVIDENCE_UPDATER:
        update_evidence = _tag_payload(text, "update-evidence")
        if "selected_passage_ids" not in update_evidence:
            raise ValueError("Missing required field: update_evidence.selected_passage_ids")
        selected_ids = update_evidence["selected_passage_ids"]
        if not isinstance(selected_ids, list) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in selected_ids
        ):
            raise ValueError(
                "update_evidence.selected_passage_ids must be a list of integer passage IDs."
            )
        _validate_optional_string(
            update_evidence,
            "rationale",
            field="update_evidence.rationale",
        )
        return ParsedAction(role=role, update_evidence=update_evidence)

    if role != AgentRole.ANSWER_GENERATOR:
        raise ValueError(f"Unsupported role: {role}")
    answer = _tag_payload(text, "answer")
    if "can_answer" not in answer:
        raise ValueError("Missing required field: answer.can_answer")
    if "answer" not in answer:
        raise ValueError("Missing required field: answer.answer")
    answer["can_answer"] = parse_can_answer(answer["can_answer"])
    _require_string(answer, "answer", field="answer.answer", allow_empty=True)
    _validate_optional_string(answer, "rationale", field="answer.rationale")
    return ParsedAction(role=role, answer=answer)
