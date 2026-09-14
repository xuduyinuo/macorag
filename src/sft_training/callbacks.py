from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _read_repairable_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                break
            raise SystemExit(f"Invalid JSONL record in {path} at line {index + 1}: {exc}") from exc
        if not isinstance(payload, dict):
            raise SystemExit(f"Invalid JSONL object in {path} at line {index + 1}")
        rows.append(payload)
    return rows


def _rewrite_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp_path, path)


def _prepare_resume_logs(output_dir: Path, checkpoint: Path, *, samples_per_epoch: int) -> int:
    state_path = checkpoint / "trainer_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    checkpoint_step = int(state.get("global_step", 0) or 0)
    checkpoint_epoch = float(state.get("epoch", 0.0) or 0.0)
    checkpoint_sample = int(math.floor(checkpoint_epoch * samples_per_epoch))
    paths = {
        "train": output_dir / "train_metrics.jsonl",
        "eval": output_dir / "eval_metrics.jsonl",
        "phase": output_dir / "phase_metrics.jsonl",
    }
    existing = {name: _read_repairable_jsonl(path) for name, path in paths.items()}
    previous_segments = [
        int(row.get("resume_segment", 0) or 0)
        for rows in existing.values()
        for row in rows
    ]
    resume_segment = max(previous_segments, default=0) + 1

    def keep_train(row: dict[str, Any]) -> bool:
        if row.get("event") == "resume":
            return int(row.get("step", 0) or 0) <= checkpoint_step
        if "global_step" in row:
            return int(row.get("global_step", 0) or 0) <= checkpoint_step
        epoch = int(row.get("epoch", 1) or 1)
        sample = int(row.get("sample", 0) or 0)
        absolute_sample = (epoch - 1) * samples_per_epoch + sample
        return absolute_sample <= checkpoint_sample

    filtered = {
        "train": [row for row in existing["train"] if keep_train(row)],
        "eval": [row for row in existing["eval"] if int(row.get("step", 0) or 0) <= checkpoint_step],
        "phase": [row for row in existing["phase"] if int(row.get("step", 0) or 0) <= checkpoint_step],
    }
    marker = {
        "event": "resume",
        "resume_segment": resume_segment,
        "checkpoint": str(checkpoint),
        "step": checkpoint_step,
        "epoch": checkpoint_epoch,
    }
    for name, path in paths.items():
        _rewrite_jsonl_atomic(path, [*filtered[name], marker])
    return resume_segment


def _distributed_token_sum(value: int, device: Any = None) -> int:
    import torch

    total = torch.tensor(int(value), dtype=torch.long, device=device)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(total)
    return int(total.item())


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
    resume_segment: int = 0,
) -> Any:
    class JsonlLoggingCallback(trainer_callback_cls):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.last_logged_sample_seen_total = self._read_existing_progress()

        def _read_existing_progress(self) -> int:
            if not self.path.is_file():
                return 0
            try:
                lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line]
                if not lines:
                    return 0
                for line in reversed(lines):
                    payload = json.loads(line)
                    if "sample" not in payload:
                        continue
                    epoch = int(payload.get("epoch", 1))
                    sample = int(payload.get("sample", 0))
                    return max(0, (epoch - 1) * samples_per_epoch + sample)
                return 0
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                return 0

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
                        "event": "metric",
                        "resume_segment": int(resume_segment),
                        "global_step": int(getattr(state, "global_step", 0) or 0),
                        "epoch": epoch,
                        "sample": sample,
                        "sample_total": progress["train_samples_per_epoch"],
                        "loss": logs.get("loss"),
                        "grad_norm": logs.get("grad_norm"),
                        "learning_rate": logs.get("learning_rate"),
                    }
                    payload.update(
                        {
                            key: value
                            for key, value in logs.items()
                            if str(key).startswith("train/")
                        }
                    )
                    file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            self.last_logged_sample_seen_total = sample_seen_total

    return JsonlLoggingCallback(log_path)


def _make_phase_metrics_callback(
    log_path: Path,
    trainer_callback_cls: Any,
    *,
    eval_token_count: int,
    resume_segment: int = 0,
    clock: Any = time.monotonic,
) -> Any:
    class PhaseMetricsCallback(trainer_callback_cls):
        def __init__(self, path: Path) -> None:
            self.path = path
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.last_train_time: float | None = None
            self.last_train_tokens = 0
            self.pending_save_start: float | None = None

        @staticmethod
        def _model_token_count(model: Any) -> int:
            value = getattr(model, "_macorag_train_token_count", 0)
            if hasattr(value, "item"):
                value = value.item()
            return int(value)

        @staticmethod
        def _model_counter_device(model: Any) -> Any:
            value = getattr(model, "_macorag_train_token_count", None)
            return getattr(value, "device", None)

        def _write(self, payload: dict[str, Any]) -> None:
            payload = {
                "event": "metric",
                "resume_segment": int(resume_segment),
                **payload,
            }
            with self.path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(payload, ensure_ascii=False) + "\n")

        def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            if not _is_main_process(args):
                return
            self.last_train_time = float(clock())
            self.last_train_tokens = self._model_token_count(model)

        def on_log(
            self,
            args: Any,
            state: Any,
            control: Any,
            logs: dict[str, Any] | None = None,
            model: Any = None,
            **kwargs: Any,
        ) -> None:
            if not logs or "loss" not in logs:
                return
            self._flush_train_metrics(args, state, model)

        def _flush_train_metrics(self, args: Any, state: Any, model: Any) -> None:
            now = float(clock())
            token_total = self._model_token_count(model)
            runtime = max(0.0, now - (self.last_train_time if self.last_train_time is not None else now))
            local_token_count = max(0, token_total - self.last_train_tokens)
            token_count = _distributed_token_sum(
                local_token_count,
                device=self._model_counter_device(model),
            )
            if _is_main_process(args) and token_count > 0:
                self._write(
                    {
                        "phase": "train",
                        "step": int(getattr(state, "global_step", 0) or 0),
                        "epoch": getattr(state, "epoch", None),
                        "runtime": runtime,
                        "token_count": token_count,
                        "tokens_per_second": token_count / runtime if runtime > 0.0 else None,
                        "throughput_scope": "global_non_padding",
                    }
                )
            self.last_train_time = now
            self.last_train_tokens = token_total

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: dict[str, Any] | None = None,
            model: Any = None,
            **kwargs: Any,
        ) -> None:
            if not metrics:
                return
            runtime = float(metrics.get("eval_runtime", 0.0) or 0.0)
            if _is_main_process(args):
                self._write(
                    {
                        "phase": "eval",
                        "step": int(getattr(state, "global_step", 0) or 0),
                        "epoch": getattr(state, "epoch", None),
                        "runtime": runtime,
                        "token_count": int(eval_token_count),
                        "tokens_per_second": eval_token_count / runtime if runtime > 0.0 else None,
                        "throughput_scope": "logical_dataset_non_padding",
                    }
                )
            self.last_train_time = float(clock())
            self.last_train_tokens = self._model_token_count(model)
            self.pending_save_start = self.last_train_time

        def _mark_save_start(self, control: Any) -> None:
            if bool(getattr(control, "should_save", False)):
                self.pending_save_start = float(clock())

        def _flush_before_boundary(self, args: Any, state: Any, control: Any, model: Any) -> None:
            boundary = bool(getattr(control, "should_evaluate", False) or getattr(control, "should_save", False))
            if boundary and not bool(getattr(control, "should_log", False)):
                self._flush_train_metrics(args, state, model)

        def on_step_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            self._flush_before_boundary(args, state, control, model)
            if _is_main_process(args):
                self._mark_save_start(control)

        def on_epoch_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            self._flush_before_boundary(args, state, control, model)
            if _is_main_process(args):
                self._mark_save_start(control)

        def on_save(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            if not _is_main_process(args):
                return
            now = float(clock())
            runtime = max(0.0, now - (self.pending_save_start or now))
            self._write(
                {
                    "phase": "save",
                    "step": int(getattr(state, "global_step", 0) or 0),
                    "epoch": getattr(state, "epoch", None),
                    "runtime": runtime,
                }
            )
            self.pending_save_start = None
            self.last_train_time = now
            self.last_train_tokens = self._model_token_count(model)

        def on_train_end(self, args: Any, state: Any, control: Any, model: Any = None, **kwargs: Any) -> None:
            self._flush_train_metrics(args, state, model)

    return PhaseMetricsCallback(log_path)


def _make_eval_metrics_callback(
    log_path: Path,
    trainer_callback_cls: Any,
    *,
    resume_segment: int = 0,
) -> Any:
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
                "event": "metric",
                "resume_segment": int(resume_segment),
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
