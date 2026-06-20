import pytest

from macorag.teacher_protocol import ProtocolError, parse_teacher_message


def test_parse_teacher_message_extracts_json_actions():
    message = """
<plan>{"thought": "find supporting documents", "steps": ["retrieve", "answer"]}</plan>
<retrieval>{"query": "Who founded Example Co?", "top_k": 5}</retrieval>
<update-evidence>{"doc_ids": ["d1"], "notes": "Alice founded Example Co."}</update-evidence>
<answer>{"text": "Alice", "confidence": 0.9}</answer>
"""

    parsed = parse_teacher_message(message)

    assert parsed["plan"] == {
        "thought": "find supporting documents",
        "steps": ["retrieve", "answer"],
    }
    assert parsed["retrieval"] == {
        "query": "Who founded Example Co?",
        "top_k": 5,
    }
    assert parsed["update-evidence"] == {
        "doc_ids": ["d1"],
        "notes": "Alice founded Example Co.",
    }
    assert parsed["answer"] == {
        "text": "Alice",
        "confidence": 0.9,
    }


def test_parse_teacher_message_requires_closed_update_evidence_tag():
    message = """
<plan>{}</plan>
<retrieval>{}</retrieval>
<update-evidence>{}</answer>
<answer>{}</answer>
"""

    with pytest.raises(ProtocolError):
        parse_teacher_message(message)
