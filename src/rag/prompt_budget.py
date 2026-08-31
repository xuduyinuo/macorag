from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable


class PromptBudgetError(RuntimeError):
    pass


@dataclass(frozen=True)
class CompactedPrompt:
    text: str
    original_tokens: int
    final_tokens: int
    removed_retrieval_history: int = 0
    removed_evidence: int = 0
    removed_observation_passages: int = 0


def _tag_payload(text: str, tag: str) -> tuple[re.Match[str] | None, dict]:
    escaped_tag = re.escape(tag)
    pattern = re.compile(
        rf"(?=(<{escaped_tag}>(.*?)</{escaped_tag}>))",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(text):
        try:
            payload = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return match, payload
    return None, {}


def _replace_tag(text: str, match: re.Match[str], tag: str, payload: dict) -> str:
    replacement = f"<{tag}>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</{tag}>"
    return text[: match.start(1)] + replacement + text[match.end(1) :]


def compact_tagged_json_prompt(
    prompt: str,
    *,
    token_count: Callable[[str], int],
    max_tokens: int,
) -> CompactedPrompt:
    if max_tokens <= 0:
        raise PromptBudgetError("max_tokens must be positive")
    original_tokens = int(token_count(prompt))
    if original_tokens <= max_tokens:
        return CompactedPrompt(prompt, original_tokens, original_tokens)

    text = prompt
    removed_history = 0
    removed_evidence = 0
    removed_observation = 0

    while int(token_count(text)) > max_tokens:
        state_match, state = _tag_payload(text, "state")
        if state_match is not None and isinstance(state.get("retrieval_history"), list) and state["retrieval_history"]:
            state["retrieval_history"].pop(0)
            removed_history += 1
            text = _replace_tag(text, state_match, "state", state)
            continue
        if state_match is not None and isinstance(state.get("evidence"), list) and state["evidence"]:
            state["evidence"].pop(0)
            removed_evidence += 1
            text = _replace_tag(text, state_match, "state", state)
            continue
        observation_match, observation = _tag_payload(text, "observation")
        passages = observation.get("passages")
        if observation_match is not None and isinstance(passages, list) and passages:
            passages.pop()
            removed_observation += 1
            text = _replace_tag(text, observation_match, "observation", observation)
            continue
        raise PromptBudgetError(
            "fixed prompt content exceeds max_tokens after compacting retrieval history, evidence, and observation"
        )

    return CompactedPrompt(
        text=text,
        original_tokens=original_tokens,
        final_tokens=int(token_count(text)),
        removed_retrieval_history=removed_history,
        removed_evidence=removed_evidence,
        removed_observation_passages=removed_observation,
    )
