from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .trajectory_parser import (
    validate_answer_decision,
    validate_query_decision,
)


def evaluate_structured_targets(records: Iterable[Any]) -> dict[str, Any]:
    """Lightweight validity report; this is a sanity metric, never early stopping."""

    import json
    import re

    totals: Counter[str] = Counter()
    valid: Counter[str] = Counter()
    for record in records:
        role = str(record.agent_role)
        totals[role] += 1
        match = re.fullmatch(r"<[^>]+>(.*)</[^>]+>", record.target_text, re.DOTALL)
        try:
            payload = json.loads(match.group(1)) if match else None
        except json.JSONDecodeError:
            payload = None
        error = "invalid target"
        if role == "query_retriever":
            error = validate_query_decision(payload)
        elif role == "evidence_updater":
            # Range validity is checked against the original observation during parsing.
            error = None if isinstance(payload, dict) and isinstance(payload.get("selected_passage_ids"), list) else error
        elif role == "answer_generator":
            error = validate_answer_decision(payload)
        if error is None:
            valid[role] += 1
    return {
        role: {
            "total": totals[role],
            "valid": valid[role],
            "valid_rate": valid[role] / max(1, totals[role]),
        }
        for role in sorted(totals)
    }
