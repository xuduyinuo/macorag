from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any


def make_run_dir(output_root: str | Path, timestamp: str | None = None) -> Path:
    """按训练阶段统一规范生成运行目录：保存根目录/时间戳。"""
    run_timestamp = timestamp or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(output_root) / run_timestamp


def _is_main_process(args: Any) -> bool:
    return int(getattr(args, "process_index", 0)) == 0


def _sample_progress_payload(state: Any, samples_per_epoch: int, total_epochs: float) -> dict[str, Any]:
    epoch_value = float(state.epoch or 0.0)
    completed_epochs = int(math.floor(epoch_value))
    epoch_fraction = epoch_value - completed_epochs
    total_epoch_count = max(1, int(math.ceil(total_epochs)))
    if epoch_value > 0.0 and abs(epoch_fraction) < 1e-9:
        current_epoch = min(completed_epochs, total_epoch_count)
        sample_in_epoch = samples_per_epoch
    elif epoch_value >= total_epochs:
        current_epoch = max(1, int(math.ceil(total_epochs)))
        sample_in_epoch = samples_per_epoch
    else:
        current_epoch = min(completed_epochs + 1, total_epoch_count)
        sample_in_epoch = int(math.floor(epoch_fraction * samples_per_epoch))
        sample_in_epoch = max(0, min(samples_per_epoch, sample_in_epoch))
    total_target = int(math.ceil(total_epochs * samples_per_epoch))
    seen_total = int(math.floor(epoch_value * samples_per_epoch))
    seen_total = max(0, min(total_target, seen_total))
    return {
        "train_sample_epoch": current_epoch,
        "train_sample_in_epoch": sample_in_epoch,
        "train_samples_per_epoch": samples_per_epoch,
        "train_sample_seen_total": seen_total,
        "train_sample_target_total": total_target,
    }


def _make_jsonl_logging_callback(
    log_path: Path,
    trainer_callback_cls: Any,
    samples_per_epoch: int,
    total_epochs: float,
) -> Any:
    class JsonlLoggingCallback(trainer_callback_cls):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.last_logged_sample_seen_total = 0

        def on_log(self, args: Any, state: Any, control: Any, logs: dict[str, Any] | None = None, **kwargs: Any) -> None:
            if not _is_main_process(args):
                return
            if not logs:
                return
            if "loss" not in logs:
                return
            progress = _sample_progress_payload(state, samples_per_epoch, total_epochs)
            sample_seen_total = int(progress["train_sample_seen_total"])
            if sample_seen_total <= self.last_logged_sample_seen_total:
                return
            with self.path.open("a", encoding="utf-8") as file:
                for sample_seen in range(self.last_logged_sample_seen_total + 1, sample_seen_total + 1):
                    epoch = ((sample_seen - 1) // samples_per_epoch) + 1
                    sample = ((sample_seen - 1) % samples_per_epoch) + 1
                    payload = {
                        "epoch": epoch,
                        "sample": sample,
                        "sample_total": progress["train_samples_per_epoch"],
                        "loss": logs.get("loss"),
                        "grad_norm": logs.get("grad_norm"),
                        "learning_rate": logs.get("learning_rate"),
                    }
                    file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.last_logged_sample_seen_total = sample_seen_total

    return JsonlLoggingCallback(log_path)


def _make_eval_metrics_callback(log_path: Path, trainer_callback_cls: Any) -> Any:
    class EvalMetricsCallback(trainer_callback_cls):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            if not _is_main_process(args):
                return
            if not metrics:
                return
            payload = {
                "step": int(getattr(state, "global_step", 0) or 0),
                "epoch": getattr(state, "epoch", None),
            }
            for key in sorted(metrics):
                if key.startswith("eval_"):
                    payload[key] = metrics[key]
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    return EvalMetricsCallback(log_path)


def _make_sample_progress_callback(
    trainer_callback_cls: Any,
    samples_per_epoch: int,
    total_epochs: float,
) -> Any:
    class SampleProgressCallback(trainer_callback_cls):
        def __init__(self) -> None:
            self.progress_bar: Any = None
            self.current_epoch: int | None = None
            self.last_sample_in_epoch = 0

        def _close_bar(self) -> None:
            if self.progress_bar is not None:
                self.progress_bar.close()
                self.progress_bar = None

        def _ensure_bar(self, epoch_number: int) -> None:
            if self.current_epoch == epoch_number and self.progress_bar is not None:
                return
            self._close_bar()
            from tqdm.auto import tqdm

            total_epoch_count = max(1, int(math.ceil(total_epochs)))
            self.current_epoch = epoch_number
            self.last_sample_in_epoch = 0
            self.progress_bar = tqdm(
                total=samples_per_epoch,
                desc=f"epoch {epoch_number}/{total_epoch_count} train samples",
                unit="sample",
                dynamic_ncols=True,
                leave=True,
            )

        def on_train_begin(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if _is_main_process(args):
                self._ensure_bar(1)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not _is_main_process(args):
                return
            progress = _sample_progress_payload(state, samples_per_epoch, total_epochs)
            epoch_number = int(progress["train_sample_epoch"])
            sample_in_epoch = int(progress["train_sample_in_epoch"])
            self._ensure_bar(epoch_number)
            delta = sample_in_epoch - self.last_sample_in_epoch
            if delta > 0 and self.progress_bar is not None:
                self.progress_bar.update(delta)
                self.last_sample_in_epoch = sample_in_epoch

        def on_epoch_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if not _is_main_process(args):
                return
            progress = _sample_progress_payload(state, samples_per_epoch, total_epochs)
            sample_in_epoch = int(progress["train_sample_in_epoch"])
            if self.progress_bar is not None and sample_in_epoch > self.last_sample_in_epoch:
                self.progress_bar.update(sample_in_epoch - self.last_sample_in_epoch)
                self.last_sample_in_epoch = sample_in_epoch
            self._close_bar()

        def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            if _is_main_process(args):
                self._close_bar()

    return SampleProgressCallback()
