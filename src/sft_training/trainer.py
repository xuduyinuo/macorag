from __future__ import annotations

from typing import Any


def _distributed_context() -> tuple[int, int]:
    try:
        import torch
    except ModuleNotFoundError:
        return 1, 0
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    return 1, 0


def _make_ordered_sampler(dataset: Any, world_size: int | None = None, process_rank: int | None = None) -> Any:
    from torch.utils.data import SequentialSampler
    from torch.utils.data.distributed import DistributedSampler

    if world_size is None or process_rank is None:
        world_size, process_rank = _distributed_context()
    if world_size > 1:
        return DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=process_rank,
            shuffle=False,
            drop_last=False,
        )
    return SequentialSampler(dataset)


def _make_target_only_trainer_cls(trainer_cls: Any) -> Any:
    class OrderedTargetOnlyTrainer(trainer_cls):
        def _get_train_sampler(self, train_dataset: Any | None = None) -> Any:
            dataset = train_dataset if train_dataset is not None else self.train_dataset
            if dataset is None:
                return None
            return _make_ordered_sampler(dataset)

        def _get_eval_sampler(self, eval_dataset: Any) -> Any:
            return _make_ordered_sampler(eval_dataset)

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
            return (loss, outputs) if return_outputs else loss

    return OrderedTargetOnlyTrainer
