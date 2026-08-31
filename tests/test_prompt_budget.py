from __future__ import annotations

import pytest

from rag.prompt_budget import PromptBudgetError, compact_tagged_json_prompt


def test_prompt_compaction_preserves_protocol_edges_and_newest_evidence() -> None:
    prompt = (
        "Task: answer from accumulated evidence.\n"
        '<state>{"evidence":[{"text":"old"},{"text":"new"}],'
        '"retrieval_history":[{"query":"q1"},{"query":"q2"}]}</state>\n'
        'Return exactly: <answer>{"can_answer":false,"answer":null,"rationale":"need more"}</answer>'
    )

    result = compact_tagged_json_prompt(prompt, token_count=len, max_tokens=len(prompt) - 30)

    assert result.text.startswith("Task: answer")
    assert result.text.endswith("</answer>")
    assert '"text":"new"' in result.text
    assert '"text":"old"' not in result.text
    assert result.removed_retrieval_history == 2


def test_prompt_compaction_fails_when_fixed_protocol_exceeds_budget() -> None:
    prompt = 'Task\n<state>{"evidence":[],"retrieval_history":[]}</state>\nReturn <answer>{}</answer>'

    with pytest.raises(PromptBudgetError, match="fixed prompt content"):
        compact_tagged_json_prompt(prompt, token_count=len, max_tokens=10)


def test_prompt_compaction_skips_prose_tag_references() -> None:
    prompt = (
        "Use <state> and <observation> below.\n"
        '<state>{"evidence":[{"text":"old"}],'
        '"retrieval_history":[{"query":"q"}]}</state>\n'
        '<observation>{"passages":[{"text":"passage"}]}</observation>\n'
        "Return exactly."
    )
    expected = (
        "Use <state> and <observation> below.\n"
        '<state>{"evidence":[],"retrieval_history":[]}</state>\n'
        '<observation>{"passages":[]}</observation>\n'
        "Return exactly."
    )

    result = compact_tagged_json_prompt(
        prompt,
        token_count=len,
        max_tokens=len(expected),
    )

    assert result.text == expected
    assert result.removed_retrieval_history == 1
    assert result.removed_evidence == 1
    assert result.removed_observation_passages == 1
