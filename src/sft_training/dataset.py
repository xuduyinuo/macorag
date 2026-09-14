from __future__ import annotations

from array import array
from dataclasses import asdict, dataclass
import hashlib
import struct
import sys
import math
from typing import Any

from .data import TrajectoryRecord
from rag.prompt_budget import PromptBudgetError
from prompt_config import DEFAULT_SYSTEM_PROMPT, system_prompt_for
from .collator import MacoRAGSFTCollator
from .formatting import compact_decision_prompt


@dataclass(frozen=True)
class TokenizationEvent:
    sample_id: str
    trajectory_id: str
    agent_type: str
    round_id: int
    input_tokens: int
    target_tokens: int
    sequence_tokens: int
    original_prompt_tokens: int
    truncated: bool
    removed_retrieval_history: int = 0
    removed_evidence: int = 0
    truncated_passage_texts: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize_tokenization(events: list[TokenizationEvent]) -> dict[str, Any]:
    def percentile(values: list[int], quantile: float) -> int:
        if not values:
            return 0
        ordered = sorted(values)
        return ordered[min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))]

    sequence = [item.sequence_tokens for item in events]
    inputs = [item.input_tokens for item in events]
    targets = [item.target_tokens for item in events]
    return {
        "number_of_decision_samples": len(events),
        "average_input_length": sum(inputs) / max(1, len(inputs)),
        "average_target_length": sum(targets) / max(1, len(targets)),
        "p50_sequence_length": percentile(sequence, 0.50),
        "p90_sequence_length": percentile(sequence, 0.90),
        "p95_sequence_length": percentile(sequence, 0.95),
        "truncated_sample_count": sum(item.truncated for item in events),
        "truncated_sample_rate": sum(item.truncated for item in events) / max(1, len(events)),
    }


def debug_tokenized_sample(record: TrajectoryRecord, feature: dict[str, Any], tokenizer: Any) -> str:
    labels = feature["labels"]
    pieces = []
    for token_id, label in zip(feature["input_ids"], labels):
        token = tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
        pieces.append(f"{'✓' if label != -100 else 'X'} {token!r}")
    return "\n".join(
        [
            f"Trajectory ID: {record.trajectory_id}",
            f"Round ID: {record.round_index}",
            f"Agent Type: {record.agent_type}",
            f"Role Instruction: {record.role_instruction}",
            "Input Context:",
            record.prompt_text,
            "Teacher Target:",
            record.target_text,
            "Tokenized Sequence / Loss Mask (X=masked, ✓=target):",
            *pieces,
        ]
    )


def create_target_token_mask(prompt_length: int, target_length: int, *, include_eos: bool = False) -> list[int]:
    """Return 1 only for actual teacher decision content tokens."""

    if prompt_length < 0 or target_length <= 0:
        raise ValueError("prompt_length must be non-negative and target_length must be positive")
    return [0] * prompt_length + [1] * target_length + ([1] if include_eos else [0])


def build_sft_labels(input_ids: list[int], decision_token_mask: list[int]) -> list[int]:
    if len(input_ids) != len(decision_token_mask):
        raise ValueError("input_ids and decision_token_mask must have equal length")
    labels = [token if mask else -100 for token, mask in zip(input_ids, decision_token_mask)]
    if not any(value != -100 for value in labels):
        raise ValueError("No teacher decision tokens remain after tokenization")
    return labels


def _dataset_fingerprint(dataset: Any) -> str:
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", len(dataset)))
    for index in range(len(dataset)):
        row = dataset[index]
        for key in ("input_ids", "labels"):
            values = array("i", (int(value) for value in row[key]))
            if sys.byteorder != "little":
                values.byteswap()
            digest.update(key.encode("ascii"))
            digest.update(struct.pack("<Q", len(values)))
            digest.update(values.tobytes())
    return digest.hexdigest()


def _tokenize_records(
    records: list[TrajectoryRecord],
    tokenizer: Any,
    max_length: int,
    system_prompt: str,
    skipped_records: list[dict[str, Any]] | None = None,
    tokenization_events: list[TokenizationEvent] | None = None,
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    input_id_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for row in records:
        active_system_prompt = (
            row.role_instruction or system_prompt_for(row.agent_role)
            if system_prompt == DEFAULT_SYSTEM_PROMPT and row.agent_role
            else system_prompt
        )
        target_tokens = tokenizer(row.target_text, add_special_tokens=False)["input_ids"]
        if not isinstance(target_tokens, list) or not target_tokens:
            if skipped_records is not None:
                skipped_records.append(
                    {
                        "qid": row.qid,
                        "dataset": row.dataset,
                        "action_type": row.action_type,
                        "token_length": len(target_tokens or []),
                        "max_length": max_length,
                    }
                )
            continue
        prompt_budget = max_length - len(target_tokens) - 1

        def encode_prompt(text: str) -> list[int]:
            prompt_messages = [
                {"role": "system", "content": active_system_prompt},
                {"role": "user", "content": text},
            ]
            return list(tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True))

        try:
            compacted = compact_decision_prompt(
                row.prompt_text,
                token_count=lambda text: len(encode_prompt(text)),
                max_tokens=prompt_budget,
            )
            prompt_tokens = encode_prompt(compacted.text)
        except PromptBudgetError:
            prompt_tokens = encode_prompt(row.prompt_text)
            compacted = None
        if not isinstance(prompt_tokens, list):
            continue
        input_ids = list(prompt_tokens) + list(target_tokens) + [tokenizer.eos_token_id]
        decision_mask = create_target_token_mask(len(prompt_tokens), len(target_tokens))
        labels = build_sft_labels(input_ids, decision_mask)
        if len(input_ids) > max_length:
            if skipped_records is not None:
                skipped_records.append(
                    {
                        "qid": row.qid,
                        "dataset": row.dataset,
                        "action_type": row.action_type,
                        "token_length": len(input_ids),
                        "max_length": max_length,
                    }
                )
            continue
        attention_mask = [1] * len(input_ids)
        input_id_rows.append(input_ids)
        label_rows.append(labels)
        attention_rows.append(attention_mask)
        if tokenization_events is not None:
            tokenization_events.append(
                TokenizationEvent(
                    sample_id=row.sample_id,
                    trajectory_id=row.trajectory_id,
                    agent_type=row.agent_type,
                    round_id=row.round_index,
                    input_tokens=len(prompt_tokens),
                    target_tokens=len(target_tokens),
                    sequence_tokens=len(input_ids),
                    original_prompt_tokens=(
                        compacted.original_tokens if compacted is not None else len(prompt_tokens)
                    ),
                    truncated=bool(compacted is not None and compacted.was_truncated),
                    removed_retrieval_history=(
                        compacted.removed_retrieval_history if compacted is not None else 0
                    ),
                    removed_evidence=(compacted.removed_evidence if compacted is not None else 0),
                    truncated_passage_texts=(
                        compacted.truncated_passage_texts if compacted is not None else 0
                    ),
                )
            )
    return input_id_rows, attention_rows, label_rows


def _build_dataset(
    tokenizer: Any,
    records: list[TrajectoryRecord],
    max_length: int,
    system_prompt: str,
    skipped_records: list[dict[str, Any]] | None = None,
    tokenization_events: list[TokenizationEvent] | None = None,
):
    from torch.utils.data import Dataset

    owned_events: list[TokenizationEvent] = []
    active_events = tokenization_events if tokenization_events is not None else owned_events
    event_start = len(active_events)
    input_ids, attention_masks, labels = _tokenize_records(
        records,
        tokenizer,
        max_length,
        system_prompt,
        skipped_records=skipped_records,
        tokenization_events=active_events,
    )
    retained_events = active_events[event_start:]

    class TrajectoryDataset(Dataset):
        def __len__(self) -> int:
            return len(input_ids)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "input_ids": input_ids[index],
                "attention_mask": attention_masks[index],
                "labels": labels[index],
                "agent_type_id": {"query": 0, "evidence": 1, "answer": 2}.get(
                    retained_events[index].agent_type,
                    -1,
                ),
            }

    return TrajectoryDataset()


def _pad_batch(features: list[dict[str, Any]], pad_token_id: int, max_length: int | None = None) -> dict[str, Any]:
    return MacoRAGSFTCollator(pad_token_id=pad_token_id, max_length=max_length)(features)
