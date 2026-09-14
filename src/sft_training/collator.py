from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MacoRAGSFTCollator:
    """Pad decision samples while keeping every non-target/padding label at -100."""

    pad_token_id: int
    max_length: int | None = None

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate an empty SFT batch")
        import torch

        max_len = max(len(item["input_ids"]) for item in features)
        if self.max_length is not None and max_len > self.max_length:
            raise RuntimeError(
                f"Batch token length {max_len} exceeds configured max_length {self.max_length}."
            )
        input_rows: list[list[int]] = []
        attention_rows: list[list[int]] = []
        label_rows: list[list[int]] = []
        for feature in features:
            input_ids = list(feature["input_ids"])
            attention = list(feature["attention_mask"])
            labels = list(feature["labels"])
            if not (len(input_ids) == len(attention) == len(labels)):
                raise ValueError("input_ids, attention_mask, and labels must have equal length")
            if not any(label != -100 for label in labels):
                raise ValueError("SFT sample has no teacher decision tokens")
            pad = max_len - len(input_ids)
            input_rows.append([self.pad_token_id] * pad + input_ids)
            attention_rows.append([0] * pad + attention)
            label_rows.append([-100] * pad + labels)
        batch = {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(attention_rows, dtype=torch.long),
            "labels": torch.tensor(label_rows, dtype=torch.long),
        }
        if all("agent_type_id" in feature for feature in features):
            batch["agent_type_id"] = torch.tensor(
                [int(feature["agent_type_id"]) for feature in features],
                dtype=torch.long,
            )
        assert bool((batch["labels"][batch["attention_mask"] == 0] == -100).all())
        assert int((batch["labels"] != -100).sum()) > 0
        return batch
