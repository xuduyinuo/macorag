from __future__ import annotations

from array import array
import hashlib
import struct
import sys
from typing import Any

from .data import TrajectoryRecord
from rag.prompt_budget import PromptBudgetError, compact_tagged_json_prompt
from prompt_config import DEFAULT_SYSTEM_PROMPT, system_prompt_for


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
) -> tuple[list[list[int]], list[list[int]], list[list[int]]]:
    input_id_rows: list[list[int]] = []
    attention_rows: list[list[int]] = []
    label_rows: list[list[int]] = []
    for row in records:
        active_system_prompt = (
            system_prompt_for(row.agent_role)
            if system_prompt == DEFAULT_SYSTEM_PROMPT and row.agent_role
            else system_prompt
        )
        target_tokens = tokenizer(row.target_text, add_special_tokens=False)["input_ids"]
        prompt_budget = max_length - len(target_tokens) - 1

        def encode_prompt(text: str) -> list[int]:
            prompt_messages = [
                {"role": "system", "content": active_system_prompt},
                {"role": "user", "content": text},
            ]
            return list(tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True))

        try:
            compacted = compact_tagged_json_prompt(
                row.prompt_text,
                token_count=lambda text: len(encode_prompt(text)),
                max_tokens=prompt_budget,
            )
            prompt_tokens = encode_prompt(compacted.text)
        except PromptBudgetError:
            prompt_tokens = encode_prompt(row.prompt_text)
        if not isinstance(prompt_tokens, list) or not isinstance(target_tokens, list):
            continue
        input_ids = list(prompt_tokens) + list(target_tokens) + [tokenizer.eos_token_id]
        labels = [-100] * len(prompt_tokens) + list(target_tokens) + [tokenizer.eos_token_id]
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
    return input_id_rows, attention_rows, label_rows


def _build_dataset(
    tokenizer: Any,
    records: list[TrajectoryRecord],
    max_length: int,
    system_prompt: str,
    skipped_records: list[dict[str, Any]] | None = None,
):
    from torch.utils.data import Dataset

    input_ids, attention_masks, labels = _tokenize_records(
        records,
        tokenizer,
        max_length,
        system_prompt,
        skipped_records=skipped_records,
    )

    class TrajectoryDataset(Dataset):
        def __len__(self) -> int:
            return len(input_ids)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                "input_ids": input_ids[index],
                "attention_mask": attention_masks[index],
                "labels": labels[index],
            }

    return TrajectoryDataset()


def _pad_batch(features: list[dict[str, Any]], pad_token_id: int, max_length: int | None = None) -> dict[str, Any]:
    import torch

    max_len = max(len(item["input_ids"]) for item in features)
    if max_length is not None and max_len > max_length:
        raise RuntimeError(f"Batch token length {max_len} exceeds configured max_length {max_length}.")
    padded_input = []
    padded_attention = []
    padded_labels = []
    for feature in features:
        input_ids = feature["input_ids"]
        attention = feature["attention_mask"]
        label = feature["labels"]
        pad_len = max_len - len(input_ids)
        padded_input.append([pad_token_id] * pad_len + input_ids)
        padded_attention.append([0] * pad_len + attention)
        padded_labels.append([-100] * pad_len + label)

    return {
        "input_ids": torch.tensor(padded_input, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }
