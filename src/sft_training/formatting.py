from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from rag.prompt_budget import PromptBudgetError


@dataclass(frozen=True)
class DecisionPromptCompaction:
    text: str
    original_tokens: int
    final_tokens: int
    removed_retrieval_history: int = 0
    removed_evidence: int = 0
    truncated_passage_texts: int = 0

    @property
    def was_truncated(self) -> bool:
        return self.final_tokens < self.original_tokens


def _read_tag(text: str, tag: str) -> tuple[re.Match[str] | None, dict[str, Any]]:
    match = re.search(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", text, re.DOTALL)
    if match is None:
        return None, {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None, {}
    return (match, payload) if isinstance(payload, dict) else (None, {})


def _replace_tag(text: str, match: re.Match[str], tag: str, payload: dict[str, Any]) -> str:
    value = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return text[: match.start()] + f"<{tag}>{value}</{tag}>" + text[match.end() :]


def compact_decision_prompt(
    prompt: str,
    *,
    token_count: Callable[[str], int],
    max_tokens: int,
) -> DecisionPromptCompaction:
    """Fit an inference prompt while preserving question and evidence candidate IDs/order.

    Old history/evidence are removed first. Candidate objects are never removed or
    renumbered; only their text is shortened, so a teacher index keeps its meaning.
    """

    if max_tokens <= 0:
        raise PromptBudgetError("max_tokens must be positive")
    original = int(token_count(prompt))
    if original <= max_tokens:
        return DecisionPromptCompaction(prompt, original, original)

    text = prompt
    removed_history = 0
    removed_evidence = 0
    while int(token_count(text)) > max_tokens:
        match, state = _read_tag(text, "state")
        history = state.get("retrieval_history")
        evidence = state.get("evidence")
        if match is not None and isinstance(history, list) and history:
            history.pop(0)
            removed_history += 1
            text = _replace_tag(text, match, "state", state)
            continue
        if match is not None and isinstance(evidence, list) and evidence:
            evidence.pop(0)
            removed_evidence += 1
            text = _replace_tag(text, match, "state", state)
            continue
        break

    truncated_passages = 0
    if int(token_count(text)) > max_tokens:
        match, observation = _read_tag(text, "observation")
        passages = observation.get("passages")
        if match is not None and isinstance(passages, list) and passages:
            original_texts = [
                str(item.get("text") or "") if isinstance(item, dict) else str(item)
                for item in passages
            ]
            low, high = 0, max((len(value) for value in original_texts), default=0)
            best: str | None = None
            best_limit = -1
            while low <= high:
                limit = (low + high) // 2
                candidate_observation = dict(observation)
                candidate_passages: list[Any] = []
                for item, value in zip(passages, original_texts):
                    if isinstance(item, dict):
                        compacted = dict(item)
                        compacted["text"] = value if len(value) <= limit else value[:limit] + "…"
                    else:
                        compacted = value if len(value) <= limit else value[:limit] + "…"
                    candidate_passages.append(compacted)
                candidate_observation["passages"] = candidate_passages
                candidate = _replace_tag(text, match, "observation", candidate_observation)
                if int(token_count(candidate)) <= max_tokens:
                    best, best_limit = candidate, limit
                    low = limit + 1
                else:
                    high = limit - 1
            if best is not None:
                text = best
                truncated_passages = sum(len(value) > best_limit for value in original_texts)

    final = int(token_count(text))
    if final > max_tokens:
        raise PromptBudgetError(
            "fixed prompt/question/candidate metadata exceeds max_tokens after safe compaction"
        )
    return DecisionPromptCompaction(
        text=text,
        original_tokens=original,
        final_tokens=final,
        removed_retrieval_history=removed_history,
        removed_evidence=removed_evidence,
        truncated_passage_texts=truncated_passages,
    )
