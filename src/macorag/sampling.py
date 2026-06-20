from __future__ import annotations

from collections import defaultdict
from typing import Any
import random


def _bucket_key(example: dict[str, Any]) -> str:
    question_type = example.get("question_type")
    if question_type:
        return str(question_type)

    hop_count = example.get("hop_count")
    if hop_count is not None:
        return f"{hop_count}hop"

    return "unknown"


def sample_examples(
    examples: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        buckets[_bucket_key(example)].append(example)

    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    bucket_names = sorted(buckets)
    while len(selected) < target_count and bucket_names:
        progressed = False
        for name in list(bucket_names):
            bucket = buckets[name]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) == target_count:
                    break
            else:
                bucket_names.remove(name)
        if not progressed:
            break

    return selected
