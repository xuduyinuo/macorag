from __future__ import annotations

from typing import Any

from .data import TrajectoryRecord


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
        prompt_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row.prompt_text},
        ]
        prompt_tokens = tokenizer.apply_chat_template(prompt_messages, add_generation_prompt=True, tokenize=True)
        target_tokens = tokenizer(row.target_text, add_special_tokens=False)["input_ids"]
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
        padded_input.append(input_ids + [pad_token_id] * pad_len)
        padded_attention.append(attention + [0] * pad_len)
        padded_labels.append(label + [-100] * pad_len)

    return {
        "input_ids": torch.tensor(padded_input, dtype=torch.long),
        "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
        "labels": torch.tensor(padded_labels, dtype=torch.long),
    }
