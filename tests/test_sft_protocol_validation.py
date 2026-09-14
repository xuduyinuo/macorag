from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sft_training.protocol_validation as protocol_validation
import sft_training.train_sft_lora_macorag as sft_train
from sft_training.config import parse_args
from sft_training.data import TrainingSample, TrajectoryRecord


def _sample(dataset: str, index: int) -> TrainingSample:
    qid = f"{dataset}-{index:03d}"
    return TrainingSample(
        qid=qid,
        dataset=dataset,
        records=[
            TrajectoryRecord(
                qid=qid,
                dataset=dataset,
                question=f"question {qid}",
                action_type="query",
                prompt_text="teacher prompt must not be used",
                target_text="teacher target must not be used",
            )
        ],
    )


def test_fixed_protocol_subset_is_balanced_deterministic_and_held_out() -> None:
    samples = [
        *[_sample("2wiki", index) for index in range(50)],
        *[_sample("hotpotqa", index) for index in range(50)],
        *[_sample("musique", index) for index in range(50)],
    ]
    selected = protocol_validation.select_fixed_protocol_samples(samples, size=100, seed=7)
    repeated = protocol_validation.select_fixed_protocol_samples(samples, size=100, seed=7)

    counts = {}
    for sample in selected:
        counts[sample.dataset] = counts.get(sample.dataset, 0) + 1
    assert counts == {"2wiki": 34, "hotpotqa": 33, "musique": 33}
    assert [(item.dataset, item.qid) for item in selected] == [
        (item.dataset, item.qid) for item in repeated
    ]
    manifest = protocol_validation.protocol_sample_manifest(selected)
    assert set(manifest[0]) == {"qid", "dataset", "question"}
    assert "teacher target" not in json.dumps(manifest)


def test_protocol_selection_score_hard_gates_then_uses_eval_loss() -> None:
    failed = {
        "parse_failure_rate": 0.011,
        "missing_answer_tag_rate": 0.0,
        "final_compliance_rate": 1.0,
    }
    passed = {
        "parse_failure_rate": 0.01,
        "missing_answer_tag_rate": 0.002,
        "final_compliance_rate": 0.99,
    }
    assert protocol_validation.protocol_selection_score(
        eligible=True, eval_loss=2.0, metrics=passed
    ) > protocol_validation.protocol_selection_score(
        eligible=False, eval_loss=0.01, metrics=failed
    )
    assert protocol_validation.protocol_selection_score(
        eligible=True, eval_loss=0.2, metrics=passed
    ) > protocol_validation.protocol_selection_score(
        eligible=True, eval_loss=0.3, metrics=passed
    )


def test_qwen_config_enables_fixed_free_generation_selection() -> None:
    args = parse_args(["--config", "config/train_sft_qwen.yml"])
    assert args.validation_split is True
    assert args.eval_strategy == "steps"
    assert args.protocol_validation_enabled is True
    assert args.protocol_validation_size == 100
    assert args.protocol_smoke_size == 20
    assert args.protocol_smoke_steps == 400
    assert args.protocol_candidate_count == 3
    assert args.gpu_indices == "0,1"
    assert args.per_device_train_batch_size == 1
    assert args.gradient_accumulation_steps == 4
    assert (
        args.per_device_train_batch_size * args.gradient_accumulation_steps * 2
    ) == 8
    assert args.metric_for_best_model == "eval_loss"
    assert args.greater_is_better is False
    assert args.protocol_max_parse_failure_rate == pytest.approx(0.01)
    assert args.protocol_max_missing_answer_tag_rate == pytest.approx(0.002)
    assert args.protocol_min_final_compliance_rate == pytest.approx(0.99)


def test_two_stage_protocol_selection_defers_best_model_loading(tmp_path: Path) -> None:
    class DummyTrainingArguments:
        def __init__(self, *, eval_strategy=None, **kwargs):
            self.kwargs = {"eval_strategy": eval_strategy, **kwargs}

    args = parse_args(["--config", "config/train_sft_qwen.yml"])
    train_args = sft_train._training_arguments(args, tmp_path, True, DummyTrainingArguments)
    assert train_args.kwargs["load_best_model_at_end"] is False
    assert train_args.kwargs["metric_for_best_model"] == "eval_loss"
    assert train_args.kwargs["greater_is_better"] is False


def test_select_top_eval_checkpoints_uses_lowest_losses_with_saved_directories(
    tmp_path: Path,
) -> None:
    rows = [
        {"step": 200, "epoch": 0.1, "eval_loss": 0.30},
        {"step": 400, "epoch": 0.2, "eval_loss": 0.10},
        {"step": 600, "epoch": 0.3, "eval_loss": 0.20},
        {"step": 800, "epoch": 0.4, "eval_loss": 0.05},
    ]
    (tmp_path / "eval_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    for step in (200, 400, 600):
        (tmp_path / f"checkpoint-{step}").mkdir()

    selected = protocol_validation.select_top_eval_checkpoints(tmp_path, count=2)

    assert [item["step"] for item in selected] == [400, 600]
    manifest = json.loads(
        (tmp_path / "protocol_validation/candidate_checkpoints.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["actual_count"] == 2


def test_prune_non_candidates_only_removes_numeric_trainer_checkpoints(tmp_path: Path) -> None:
    keep = tmp_path / "checkpoint-200"
    remove = tmp_path / "checkpoint-400"
    unrelated = tmp_path / "checkpoint-not-a-step"
    for path in (keep, remove, unrelated):
        path.mkdir()
        (path / "marker").write_text("x", encoding="utf-8")

    removed = protocol_validation.prune_non_candidate_checkpoints(
        tmp_path,
        [{"checkpoint_path": str(keep)}],
    )

    assert removed == [str(remove)]
    assert keep.is_dir()
    assert not remove.exists()
    assert unrelated.is_dir()


def test_protocol_callback_injects_metrics_and_records_eligible_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class DummyCallback:
        pass

    class FakePolicy:
        def __init__(self, **kwargs):
            assert kwargs["score_completions"] is False
            assert kwargs["generation_use_cache"] is True

        def reset_trace(self):
            return None

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def run(self, *, question, dataset):
            return SimpleNamespace(
                final_answer="answer",
                trajectory=[
                    {
                        "force_final_answer": True,
                        "raw_responses": {
                            "answer_generator": '<answer>{"can_answer":true,"answer":"answer"}</answer>'
                        },
                        "answer": {"can_answer": True, "answer": "answer"},
                    }
                ],
                parse_errors=[],
                state=SimpleNamespace(retrieval_count=1),
            )

    monkeypatch.setattr(protocol_validation, "HFSharedPolicy", FakePolicy)
    monkeypatch.setattr(protocol_validation, "RAGLoopExecutor", FakeExecutor)
    callback = protocol_validation.make_protocol_validation_callback(
        DummyCallback,
        output_dir=tmp_path,
        samples=[_sample("2wiki", 1), _sample("hotpotqa", 1)],
        tokenizer=object(),
        prompt_contract=object(),
        max_rounds=4,
        max_prompt_length=4096,
        max_completion_length=256,
        temperature=0.0,
        top_p=1.0,
        retrieval_top_k=5,
        retrieval_backend="e5_faiss",
        retrieval_root="unused",
        retrieval_embedding_model="unused",
        retrieval_device="cpu",
        retrieval_max_length=512,
        retrieval_batch_size=32,
        max_parse_failure_rate=0.01,
        max_missing_answer_tag_rate=0.002,
        min_final_compliance_rate=0.99,
        retrieval_env_factory=lambda **kwargs: object(),
    )
    metrics = {"eval_loss": 0.25}
    callback.on_evaluate(
        SimpleNamespace(process_index=0),
        SimpleNamespace(global_step=200, epoch=0.5),
        SimpleNamespace(),
        metrics,
        model=object(),
    )

    assert metrics["eval_protocol_checkpoint_eligible"] == 1.0
    assert metrics["eval_protocol_final_compliance_rate"] == 1.0
    assert metrics["eval_protocol_selection_score"] == pytest.approx(999999.75)
    assert callback.best_eligible["checkpoint_path"] == str(tmp_path / "checkpoint-200")
    rows = [
        json.loads(line)
        for line in (tmp_path / "protocol_validation/checkpoint-200/predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 2


def test_protocol_callback_shards_and_merges_distributed_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    class DummyCallback:
        pass

    generation_model = object()
    seen_models = []

    class FakePolicy:
        def __init__(self, **kwargs):
            seen_models.append(kwargs["model"])

        def reset_trace(self):
            return None

    class FakeExecutor:
        def __init__(self, **kwargs):
            pass

        def run(self, *, question, dataset):
            return SimpleNamespace(
                final_answer="answer",
                trajectory=[
                    {
                        "force_final_answer": True,
                        "raw_responses": {
                            "answer_generator": '<answer>{"can_answer":true,"answer":"answer"}</answer>'
                        },
                        "answer": {"can_answer": True, "answer": "answer"},
                    }
                ],
                parse_errors=[],
                state=SimpleNamespace(retrieval_count=1),
            )

    monkeypatch.setattr(protocol_validation, "HFSharedPolicy", FakePolicy)
    monkeypatch.setattr(protocol_validation, "RAGLoopExecutor", FakeExecutor)
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)

    def all_gather_object(gathered, local_payload):
        other_prediction = dict(local_payload["predictions"][0])
        other_prediction.update(
            {
                "qid": "hotpotqa-001",
                "dataset": "hotpotqa",
                "question": "question hotpotqa-001",
            }
        )
        gathered[0] = local_payload
        gathered[1] = {
            "predictions": [other_prediction],
            "runtime": local_payload["runtime"],
        }

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    callback = protocol_validation.make_protocol_validation_callback(
        DummyCallback,
        output_dir=tmp_path,
        samples=[_sample("2wiki", 1), _sample("hotpotqa", 1)],
        tokenizer=object(),
        prompt_contract=object(),
        max_rounds=4,
        max_prompt_length=4096,
        max_completion_length=256,
        temperature=0.0,
        top_p=1.0,
        retrieval_top_k=5,
        retrieval_backend="e5_faiss",
        retrieval_root="unused",
        retrieval_embedding_model="unused",
        retrieval_device="cpu",
        retrieval_max_length=512,
        retrieval_batch_size=32,
        max_parse_failure_rate=0.01,
        max_missing_answer_tag_rate=0.002,
        min_final_compliance_rate=0.99,
        retrieval_env_factory=lambda **kwargs: object(),
    )
    callback.on_evaluate(
        SimpleNamespace(process_index=0),
        SimpleNamespace(global_step=400, epoch=0.5),
        SimpleNamespace(),
        {"eval_loss": 0.25},
        model=SimpleNamespace(module=generation_model),
    )

    assert seen_models == [generation_model]
    rows = [
        json.loads(line)
        for line in (tmp_path / "protocol_validation/checkpoint-400/predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [(row["dataset"], row["qid"]) for row in rows] == [
        ("2wiki", "2wiki-001"),
        ("hotpotqa", "hotpotqa-001"),
    ]


def test_no_eligible_checkpoint_refuses_final_export(tmp_path: Path) -> None:
    callback = SimpleNamespace(best_eligible=None)
    trainer = SimpleNamespace(state=SimpleNamespace(best_model_checkpoint=None))
    with pytest.raises(SystemExit, match="final adapter was not exported"):
        protocol_validation.require_eligible_best_checkpoint(callback, trainer)
