from __future__ import annotations

import json
import re
from typing import Any


class ProtocolError(ValueError):
    pass


TAG_NAMES = ("plan", "retrieval", "update-evidence", "answer")


def _extract_tag(text: str, tag: str) -> dict[str, Any]:
    pattern = rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>"
    matches = re.findall(pattern, text, flags=re.DOTALL)
    if not matches:
        raise ProtocolError(f"missing closed <{tag}> tag")
    if len(matches) > 1:
        raise ProtocolError(f"duplicate <{tag}> tags")

    raw_payload = matches[0].strip()
    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"<{tag}> payload must be valid JSON") from exc

    if not isinstance(payload, dict):
        raise ProtocolError(f"<{tag}> payload must be a JSON object")

    return payload


def parse_teacher_message(text: str) -> dict[str, dict[str, Any]]:
    return {tag: _extract_tag(text, tag) for tag in TAG_NAMES}
