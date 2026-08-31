from __future__ import annotations

from typing import Any


def _mean_per_example_target_loss(logits: Any, labels: Any) -> Any:
    import torch.nn.functional as functional

    if logits.shape[1] != labels.shape[1]:
        logits = logits[:, -labels.shape[1] :, :]
    token_losses = functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape(labels.shape)
    valid = labels.ne(-100)
    valid_counts = valid.sum(dim=1)
    per_example = (token_losses * valid).sum(dim=1) / valid_counts.clamp_min(1)
    return per_example[valid_counts > 0].mean()


def _make_train_sampler(dataset: Any, world_size: int | None = None, process_rank: int | None = None) -> Any:
    from torch.utils.data import RandomSampler
    return RandomSampler(dataset)


def _make_ordered_sampler(dataset: Any, world_size: int | None = None, process_rank: int | None = None) -> Any:
    from torch.utils.data import SequentialSampler
    return SequentialSampler(dataset)


def _make_length_grouped_eval_sampler(
    dataset: Any,
    world_size: int | None = None,
    process_rank: int | None = None,
) -> Any:
    from torch.utils.data import Sampler

    class LengthGroupedEvalSampler(Sampler[int]):
        def __init__(self) -> None:
            self.indices = sorted(
                range(len(dataset)),
                key=lambda index: (len(dataset[index]["input_ids"]), index),
            )
        def __iter__(self):
            return iter(self.indices)

        def __len__(self) -> int:
            return len(self.indices)

    return LengthGroupedEvalSampler()


def _make_target_only_trainer_cls(trainer_cls: Any, *, train_shuffle: bool = True) -> Any:
    class OrderedTargetOnlyTrainer(trainer_cls):
        def _get_train_sampler(self, train_dataset: Any | None = None) -> Any:
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            if dataset is None:
                return None
            if train_shuffle:
                return _make_train_sampler(dataset)
            return _make_ordered_sampler(dataset)

        def _get_eval_sampler(self, eval_dataset: Any) -> Any:
            return _make_length_grouped_eval_sampler(eval_dataset)

        def training_step(
            self,
            model: Any,
            inputs: dict[str, Any],
            num_items_in_batch: Any = None,
        ) -> Any:
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                token_owner = getattr(model, "module", model)
                token_count = attention_mask.sum()
                previous = getattr(token_owner, "_macorag_train_token_count", None)
                token_owner._macorag_train_token_count = token_count if previous is None else previous + token_count
            return super().training_step(
                model,
                inputs,
                num_items_in_batch=num_items_in_batch,
            )

        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            labels = inputs.get("labels")
            if labels is None:
                return super().compute_loss(
                    model,
                    inputs,
                    return_outputs=return_outputs,
                    num_items_in_batch=num_items_in_batch,
                )

            valid_target_counts = labels.ne(-100).sum(dim=1)
            logits_to_keep = int(valid_target_counts.max().item()) + 1
            logits_to_keep = max(1, min(logits_to_keep, labels.shape[1]))

            import torch.nn.functional as functional

            shift_labels = functional.pad(labels, (0, 1), value=-100)[..., 1:].contiguous()
            shift_labels = shift_labels[:, -logits_to_keep:].contiguous()

            model_inputs = dict(inputs)
            model_inputs["logits_to_keep"] = logits_to_keep
            model_inputs["shift_labels"] = shift_labels
            if num_items_in_batch is not None:
                model_inputs["num_items_in_batch"] = num_items_in_batch

            outputs = model(**model_inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
            if not bool(getattr(model, "training", True)):
                logits = outputs["logits"] if isinstance(outputs, dict) else outputs.logits
                loss = _mean_per_example_target_loss(logits, shift_labels)
            return (loss, outputs) if return_outputs else loss

    return OrderedTargetOnlyTrainer
