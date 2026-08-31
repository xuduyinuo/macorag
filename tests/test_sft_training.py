from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sft_training.train_sft_lora_macorag import (
    TrajectoryRecord,
    _make_eval_metrics_callback,
    _make_jsonl_logging_callback,
    _make_phase_metrics_callback,
    _model_kwargs,
    _prepare_resume_logs,
    _resolve_resume_checkpoint,
    _resume_uses_random_sampler,
    _run_trainer,
    _synchronize_resume_logs,
    _training_arguments,
    _validate_acceleration_runtime,
    _validate_resume_compatibility,
    _validate_resume_runtime_files,
    _write_json_atomic,
    make_run_dir,
    _tokenize_records,
    parse_args,
    trajectory_to_sft_records,
)
from prompt_config import load_system_prompt
from rag import (
    AnswerPromptContext,
    RAGState,
    advance_rag_state,
    build_answer_generator_prompt,
    build_evidence_updater_prompt,
    build_query_retriever_prompt,
)
from sft_training.data import (
    TrainingSample,
    flatten_training_samples,
    split_training_samples,
    validate_teacher_dataset_contract,
)
from sft_training.dataset import _dataset_fingerprint, _pad_batch
from sft_training.callbacks import _distributed_token_sum
from sft_training.trainer import (
    _make_length_grouped_eval_sampler,
    _make_target_only_trainer_cls,
    _make_train_sampler,
    _mean_per_example_target_loss,
)


FORBIDDEN_PROMPT_TERMS = (
    "multi-agent",
    "多智能体",
    "agent_role",
    "planner_retriever",
    "evidence_answerer",
    "你是",
)


def test_trajectory_to_sft_records_splits_query_and_evidence_update_actions() -> None:
    records = trajectory_to_sft_records(
        {
            "qid": "qid-1",
            "dataset": "2wiki",
            "question": "Are both lakes in the same country?",
            "trajectory": [
            {
                "state": {"evidence": [{"text": "state leak"}]},
                "plan": {"sub_goal": "find entity", "sub_query": "entity query"},
                "retrieval": {"query": "entity query", "top_k": 5},
                "observation": {
                    "passages": [
                        {"passage_id": 0, "title": "T", "text": "observation leak", "score": 0.9}
                    ]
                },
                "update_evidence": {
                    "selected_passage_ids": [0],
                    "rationale": "Selected the relevant passage.",
                    "evidence": [
                        {
                            "passage_id": 0,
                            "title": "T",
                            "text": "evidence leak",
                            "score": 0.9,
                            "source_query": "entity query",
                        }
                    ],
                },
                "answer": {"can_answer": False, "answer": None},
            }
            ],
        }
    )

    assert len(records) == 3

    query_record = records[0]
    assert query_record.action_type == "query_retriever"
    assert query_record.agent_role == "query_retriever"
    assert "Are both lakes in the same country?" in query_record.prompt_text
    assert query_record.prompt_text.startswith("Task: plan the next knowledge-base query.")
    assert "You are a retrieval-augmented reasoning assistant" not in query_record.prompt_text
    assert "<state>" in query_record.prompt_text
    assert "state leak" in query_record.prompt_text
    assert "<observation>" not in query_record.prompt_text
    assert not any(term in query_record.prompt_text for term in FORBIDDEN_PROMPT_TERMS)
    assert "<query-retriever>" in query_record.target_text
    assert "<plan>" not in query_record.target_text
    assert "<retrieval>" not in query_record.target_text
    assert "<update-evidence>" not in query_record.target_text
    assert "<answer>" not in query_record.target_text
    assert '"sub_goal": "find entity"' in query_record.target_text
    assert '"query": "entity query"' in query_record.target_text
    state_before = RAGState(
        question="Are both lakes in the same country?",
        evidence=[{"text": "state leak"}],
    )
    assert query_record.prompt_text == build_query_retriever_prompt(
        question=state_before.question,
        state=state_before,
    )

    update_record = records[1]
    assert update_record.action_type == "evidence_update"
    assert update_record.agent_role == "evidence_updater"
    assert "<state>" in update_record.prompt_text
    assert "<retrieval>" not in update_record.prompt_text
    assert "<observation>" in update_record.prompt_text
    assert "observation leak" in update_record.prompt_text
    assert update_record.prompt_text.startswith("Task: select evidence from the latest observation.")
    assert "You are a retrieval-augmented reasoning assistant" not in update_record.prompt_text
    assert not any(term in update_record.prompt_text for term in FORBIDDEN_PROMPT_TERMS)
    assert "<update-evidence>" in update_record.target_text
    assert "<answer>" not in update_record.target_text
    assert "<plan>" not in update_record.target_text
    assert "<retrieval>" not in update_record.target_text

    assert "selected_passage_ids" in update_record.target_text
    assert "rationale" in update_record.target_text
    assert "evidence leak" not in update_record.target_text
    assert '"evidence"' not in update_record.target_text
    assert '"score"' not in update_record.target_text
    assert '"source_query"' not in update_record.target_text
    updater_state = RAGState(
        question=state_before.question,
        current_sub_goal="find entity",
        evidence=[{"text": "state leak"}],
    )
    observation = {
        "passages": [{"passage_id": 0, "title": "T", "text": "observation leak", "score": 0.9}]
    }
    assert update_record.prompt_text == build_evidence_updater_prompt(
        question=state_before.question,
        state=updater_state,
        observation=observation,
    )

    answer_record = records[2]
    assert answer_record.action_type == "answer"
    assert answer_record.agent_role == "answer_generator"
    assert "<state>" in answer_record.prompt_text
    assert "<observation>" not in answer_record.prompt_text
    assert "<update-evidence>" not in answer_record.prompt_text
    assert "observation leak" in answer_record.prompt_text
    assert answer_record.prompt_text.startswith("Task: answer from accumulated evidence.")
    assert "You are a retrieval-augmented reasoning assistant" not in answer_record.prompt_text
    assert "<answer>" in answer_record.target_text
    assert "<plan>" not in answer_record.target_text
    assert "<retrieval>" not in answer_record.target_text
    assert "<update-evidence>" not in answer_record.target_text
    answer_state = advance_rag_state(
        state_before,
        query_action={"sub_goal": "find entity", "query": "entity query"},
        observation=observation,
        update_action={"selected_passage_ids": [0]},
    )
    assert answer_record.prompt_text == build_answer_generator_prompt(
        question=state_before.question,
        state=answer_state,
        context=AnswerPromptContext(round_index=0, max_rounds=4),
    )
    assert answer_record.round_index == 0
    assert answer_record.max_rounds == 4


def test_parse_args_loads_yaml_config(tmp_path) -> None:
    config = tmp_path / "train.yml"
    config.write_text(
        "\n".join(
            [
                'model_path: "model/from-yaml"',
                'data_root: "data/from-yaml"',
                'output_root: "outputs/from-yaml"',
                "max_length: 1234",
                "max_samples: 5",
                "lora_r: 8",
                "lora_alpha: 16",
                "lora_dropout: 0.1",
                "target_modules:",
                "  - q_proj",
                "  - v_proj",
                "per_device_train_batch_size: 2",
                "gradient_accumulation_steps: 4",
                "num_train_epochs: 1.5",
                "learning_rate: 0.0003",
                "validation_split: true",
                "eval_split_ratio: 0.1",
                "early_stopping_patience: 3",
                "early_stopping_threshold: 0.01",
                'metric_for_best_model: "eval_loss"',
                "bf16: true",
                "load_4bit: true",
                "disable_tqdm: false",
                'gpu_indices: "0,1"',
                'resume_from_checkpoint: "outputs/sft/checkpoint-20"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.model_path == "model/from-yaml"
    assert args.data_root == "data/from-yaml"
    assert args.output_root == "outputs/from-yaml"
    assert args.max_length == 1234
    assert args.max_samples == 5
    assert args.lora_r == 8
    assert args.lora_alpha == 16
    assert args.lora_dropout == 0.1
    assert args.target_modules == ["q_proj", "v_proj"]
    assert args.per_device_train_batch_size == 2
    assert args.gradient_accumulation_steps == 4
    assert args.num_train_epochs == 1.5
    assert args.learning_rate == 0.0003
    assert args.validation_split is True
    assert args.eval_split_ratio == 0.1
    assert args.early_stopping_patience == 3
    assert args.early_stopping_threshold == 0.01
    assert args.metric_for_best_model == "eval_loss"
    assert args.bf16 is True
    assert args.load_4bit is True
    assert args.disable_tqdm is False
    assert args.gpu_indices == "0,1"
    assert args.resume_from_checkpoint == "outputs/sft/checkpoint-20"


def test_teacher_dataset_contract_rejects_prompt_mismatch(tmp_path: Path) -> None:
    from prompt_config import load_prompt_contract

    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "prompt_contract_version": "macorag-rag-v2",
                "prompt_contract_fingerprint": "wrong",
                "max_rounds": 4,
                "retrieval_top_k": 5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="prompt contract fingerprint mismatch"):
        validate_teacher_dataset_contract(
            tmp_path,
            expected_contract=load_prompt_contract(),
            max_rounds=4,
            retrieval_top_k=5,
        )


def test_sft_default_system_prompt_comes_from_shared_prompt_file() -> None:
    args = parse_args(["--config", "config/train_sft.yml", "--max-samples", "1"])

    assert args.system_prompt == load_system_prompt()


def test_parse_args_rejects_removed_output_and_log_keys(tmp_path) -> None:
    config = tmp_path / "train.yml"
    config.write_text(
        "\n".join(
            [
                'output_dir: "outputs/from-yaml"',
                'log_jsonl_path: "outputs/from-yaml/train_metrics.jsonl"',
            ]
        ),
        encoding="utf-8",
    )

    try:
        parse_args(["--config", str(config)])
    except SystemExit as exc:
        message = str(exc)
        assert "output_dir" in message
        assert "log_jsonl_path" in message
    else:
        raise AssertionError("expected removed SFT output/log keys to fail")


def test_parse_args_cli_overrides_yaml_config(tmp_path) -> None:
    config = tmp_path / "train.yml"
    config.write_text(
        "\n".join(
            [
                'output_root: "outputs/from-yaml"',
                "max_length: 1024",
                "max_samples: 5",
                "validation_split: false",
                "target_modules:",
                "  - q_proj",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--output-root",
            "outputs/from-cli",
            "--max-length",
            "2048",
            "--max-samples",
            "2",
            "--validation-split",
            "--target-modules",
            "k_proj",
            "o_proj",
        ]
    )

    assert args.output_root == "outputs/from-cli"
    assert args.max_length == 2048
    assert args.max_samples == 2
    assert args.validation_split is True
    assert args.target_modules == ["k_proj", "o_proj"]


def test_tokenize_records_masks_prompt_and_trains_only_target() -> None:
    class DummyTokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            assert add_generation_prompt is True
            assert tokenize is True
            assert "visible prompt" in messages[-1]["content"]
            return [1, 2, 3]

        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            assert text == "<plan>{}</plan>"
            return {"input_ids": [4, 5]}

    input_ids, attention_masks, labels = _tokenize_records(
        [
            TrajectoryRecord(
                qid="qid",
                question="question",
                dataset="dataset",
                action_type="query",
                prompt_text="visible prompt",
                target_text="<plan>{}</plan>",
            )
        ],
        DummyTokenizer(),
        max_length=32,
        system_prompt="system",
    )

    assert input_ids == [[1, 2, 3, 4, 5, 99]]
    assert attention_masks == [[1, 1, 1, 1, 1, 1]]
    assert labels == [[-100, -100, -100, 4, 5, 99]]


def test_tokenize_records_skips_records_over_max_length() -> None:
    class DummyTokenizer:
        eos_token_id = 99

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            return [1, 2, 3, 4]

        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [5, 6, 7]}

    skipped_records = []
    input_ids, attention_masks, labels = _tokenize_records(
        [
            TrajectoryRecord(
                qid="qid-long",
                question="question",
                dataset="dataset",
                action_type="query",
                prompt_text="visible prompt",
                target_text="<plan>{}</plan>",
            )
        ],
        DummyTokenizer(),
        max_length=6,
        system_prompt="system",
        skipped_records=skipped_records,
    )

    assert input_ids == []
    assert attention_masks == []
    assert labels == []
    assert skipped_records == [
        {
            "qid": "qid-long",
            "dataset": "dataset",
            "action_type": "query",
            "token_length": 8,
            "max_length": 6,
        }
    ]


def test_dataset_fingerprint_changes_with_content_and_order() -> None:
    class TinyDataset:
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    row_a = {"input_ids": [1, 2], "labels": [-100, 2]}
    row_b = {"input_ids": [1, 3], "labels": [-100, 3]}

    fingerprint = _dataset_fingerprint(TinyDataset([row_a, row_b]))

    assert fingerprint != _dataset_fingerprint(TinyDataset([row_b, row_a]))
    assert fingerprint != _dataset_fingerprint(TinyDataset([row_a, row_a]))


def test_jsonl_logging_callback_writes_one_line_per_trained_sample(tmp_path) -> None:
    class DummyCallback:
        pass

    log_path = tmp_path / "train_metrics.jsonl"
    callback = _make_jsonl_logging_callback(
        log_path,
        DummyCallback,
        samples_per_epoch=10,
        total_epochs=1.0,
        resume_segment=2,
    )

    callback.on_log(
        None,
        SimpleNamespace(epoch=0.25, global_step=1),
        None,
        {"loss": 1.2, "grad_norm": 3.4, "learning_rate": 0.0001},
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"event": "metric", "resume_segment": 2, "global_step": 1, "epoch": 1, "sample": 1, "sample_total": 10, "loss": 1.2, "grad_norm": 3.4, "learning_rate": 0.0001},
        {"event": "metric", "resume_segment": 2, "global_step": 1, "epoch": 1, "sample": 2, "sample_total": 10, "loss": 1.2, "grad_norm": 3.4, "learning_rate": 0.0001},
    ]


def test_jsonl_logging_callback_continues_existing_sample_progress(tmp_path) -> None:
    class DummyCallback:
        pass

    log_path = tmp_path / "train_metrics.jsonl"
    log_path.write_text(
        json.dumps({"epoch": 1, "sample": 5, "sample_total": 10, "loss": 1.0}) + "\n",
        encoding="utf-8",
    )
    callback = _make_jsonl_logging_callback(
        log_path,
        DummyCallback,
        samples_per_epoch=10,
        total_epochs=1.0,
    )

    callback.on_log(
        None,
        SimpleNamespace(epoch=0.7, global_step=7),
        None,
        {"loss": 0.7, "grad_norm": 1.2, "learning_rate": 0.0001},
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["sample"] for row in rows] == [5, 6, 7]


def test_prepare_resume_logs_truncates_abandoned_branch_and_repairs_partial_tail(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        json.dumps({"global_step": 100, "epoch": 0.4}),
        encoding="utf-8",
    )
    (tmp_path / "train_metrics.jsonl").write_text(
        "\n".join(
            json.dumps({"epoch": 1, "sample": sample, "sample_total": 10, "loss": sample})
            for sample in range(1, 7)
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "eval_metrics.jsonl").write_text(
        json.dumps({"step": 80, "eval_loss": 1.0}) + "\n" + json.dumps({"step": 120, "eval_loss": 0.9}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "phase_metrics.jsonl").write_text(
        json.dumps({"phase": "train", "step": 90}) + "\n" + '{"phase":"train"',
        encoding="utf-8",
    )

    segment = _prepare_resume_logs(tmp_path, checkpoint, samples_per_epoch=10)

    assert segment == 1
    train_rows = [json.loads(line) for line in (tmp_path / "train_metrics.jsonl").read_text().splitlines()]
    eval_rows = [json.loads(line) for line in (tmp_path / "eval_metrics.jsonl").read_text().splitlines()]
    phase_rows = [json.loads(line) for line in (tmp_path / "phase_metrics.jsonl").read_text().splitlines()]
    assert [row.get("sample") for row in train_rows if row.get("event") != "resume"] == [1, 2, 3, 4]
    assert [row.get("step") for row in eval_rows if row.get("event") != "resume"] == [80]
    assert [row.get("step") for row in phase_rows if row.get("event") != "resume"] == [90]
    for rows in (train_rows, eval_rows, phase_rows):
        assert rows[-1] == {
            "event": "resume",
            "resume_segment": 1,
            "checkpoint": str(checkpoint),
            "step": 100,
            "epoch": 0.4,
        }


def test_synchronize_resume_logs_broadcasts_rank_zero_segment(monkeypatch, tmp_path: Path) -> None:
    import torch
    import sft_training.train_sft_lora_macorag as entrypoint

    checkpoint = tmp_path / "checkpoint-100"
    calls = []
    monkeypatch.setattr(entrypoint, "_is_main_process", lambda: False)
    monkeypatch.setattr(
        entrypoint,
        "_prepare_resume_logs",
        lambda *args, **kwargs: pytest.fail("non-main rank must not rewrite shared logs"),
    )
    initialized = {"value": False}
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized["value"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def init_process_group(*, backend):
        calls.append(("init", backend))
        initialized["value"] = True

    monkeypatch.setattr(torch.distributed, "init_process_group", init_process_group)

    def broadcast(values, src):
        calls.append(("broadcast", src))
        values[0] = 7

    monkeypatch.setattr(torch.distributed, "broadcast_object_list", broadcast)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: calls.append(("barrier", None)))

    segment = _synchronize_resume_logs(tmp_path, checkpoint, samples_per_epoch=10)

    assert segment == 7
    assert calls == [("init", "gloo"), ("broadcast", 0), ("barrier", None)]


def test_pad_batch_left_aligns_target_suffixes() -> None:
    batch = _pad_batch(
        [
            {
                "input_ids": [1, 2, 3, 4],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, 3, 4],
            },
            {
                "input_ids": [5, 6],
                "attention_mask": [1, 1],
                "labels": [-100, 6],
            },
        ],
        pad_token_id=0,
    )

    assert batch["input_ids"].tolist() == [[1, 2, 3, 4], [0, 0, 5, 6]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1, 1], [0, 0, 1, 1]]
    assert batch["labels"].tolist() == [
        [-100, -100, 3, 4],
        [-100, -100, -100, 6],
    ]


def test_phase_metrics_callback_records_exact_train_and_eval_token_throughput(tmp_path) -> None:
    class DummyCallback:
        pass

    times = iter([10.0, 12.0, 20.0])
    callback = _make_phase_metrics_callback(
        tmp_path / "phase_metrics.jsonl",
        DummyCallback,
        eval_token_count=900,
        resume_segment=3,
        clock=lambda: next(times),
    )
    model = SimpleNamespace(_macorag_train_token_count=100)
    callback.on_train_begin(None, SimpleNamespace(global_step=0, epoch=0.0), None, model=model)
    model._macorag_train_token_count = 500
    callback.on_log(
        None,
        SimpleNamespace(global_step=1, epoch=0.1),
        None,
        {"loss": 1.0},
        model=model,
    )
    callback.on_evaluate(
        None,
        SimpleNamespace(global_step=10, epoch=1.0),
        None,
        {"eval_runtime": 6.0},
        model=model,
    )

    rows = [
        json.loads(line)
        for line in (tmp_path / "phase_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["phase"] == "train"
    assert rows[0]["event"] == "metric"
    assert rows[0]["resume_segment"] == 3
    assert rows[0]["token_count"] == 400
    assert rows[0]["runtime"] == 2.0
    assert rows[0]["tokens_per_second"] == 200.0
    assert rows[0]["throughput_scope"] == "global_non_padding"
    assert rows[1]["phase"] == "eval"
    assert rows[1]["token_count"] == 900
    assert rows[1]["runtime"] == 6.0
    assert rows[1]["tokens_per_second"] == 150.0
    assert rows[1]["throughput_scope"] == "logical_dataset_non_padding"


def test_phase_metrics_flushes_train_tokens_before_unlogged_epoch_eval(tmp_path) -> None:
    class DummyCallback:
        pass

    times = iter([0.0, 2.0])
    callback = _make_phase_metrics_callback(
        tmp_path / "phase_metrics.jsonl",
        DummyCallback,
        eval_token_count=0,
        clock=lambda: next(times),
    )
    model = SimpleNamespace(_macorag_train_token_count=0)
    callback.on_train_begin(None, SimpleNamespace(global_step=0, epoch=0.0), None, model=model)
    model._macorag_train_token_count = 100

    callback.on_epoch_end(
        None,
        SimpleNamespace(global_step=10, epoch=1.0),
        SimpleNamespace(should_log=False, should_evaluate=True, should_save=False),
        model=model,
    )

    rows = [json.loads(line) for line in (tmp_path / "phase_metrics.jsonl").read_text().splitlines()]
    assert rows[0]["phase"] == "train"
    assert rows[0]["token_count"] == 100


def test_distributed_token_sum_reduces_all_ranks(monkeypatch) -> None:
    import torch

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor: tensor.mul_(2))

    assert _distributed_token_sum(7) == 14


def test_eval_metrics_callback_writes_one_line_per_eval(tmp_path) -> None:
    class DummyCallback:
        pass

    log_path = tmp_path / "eval_metrics.jsonl"
    callback = _make_eval_metrics_callback(log_path, DummyCallback, resume_segment=4)

    callback.on_evaluate(
        None,
        SimpleNamespace(epoch=1.25, global_step=120),
        None,
        {"eval_loss": 0.12, "eval_runtime": 3.4, "eval_samples_per_second": 5.6},
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "event": "metric",
            "resume_segment": 4,
            "step": 120,
            "epoch": 1.25,
            "eval_loss": 0.12,
            "eval_runtime": 3.4,
            "eval_samples_per_second": 5.6,
        }
    ]


def test_run_dir_uses_output_root_child_timestamp() -> None:
    assert make_run_dir("outputs/lora_qwen2.5-7b_trajectory", "2026-07-02_12-34-56").as_posix() == (
        "outputs/lora_qwen2.5-7b_trajectory/2026-07-02_12-34-56"
    )


def test_train_sft_yaml_keeps_tuning_keys_and_removes_low_frequency_defaults() -> None:
    import yaml

    config = yaml.safe_load(Path("config/train_sft.yml").read_text(encoding="utf-8"))

    for key in [
        "model_path",
        "data_root",
        "output_root",
        "max_length",
        "max_samples",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "target_modules",
        "per_device_train_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "learning_rate",
        "max_steps",
        "logging_steps",
        "save_steps",
        "eval_strategy",
        "resume_from_checkpoint",
        "validation_split",
        "eval_split_ratio",
        "early_stopping_patience",
        "fp16",
        "bf16",
        "attn_implementation",
        "load_4bit",
        "gpu_indices",
    ]:
        assert key in config

    for key in [
        "output_dir",
        "system_prompt",
        "seed",
        "lr_scheduler_type",
        "warmup_ratio",
        "weight_decay",
        "save_total_limit",
        "early_stopping_threshold",
        "metric_for_best_model",
        "greater_is_better",
        "disable_tqdm",
        "log_jsonl_path",
        "gpu_index",
        "check_only",
        "check_only_max_samples",
        "train_test_seed",
    ]:
        assert key not in config

    assert config["eval_strategy"] == "epoch"


def test_active_sft_config_evaluates_once_per_epoch() -> None:
    args = parse_args(["--config", "config/train_sft.yml"])

    assert args.eval_strategy == "epoch"


def test_active_sft_config_enables_bf16_flash_attention() -> None:
    args = parse_args(["--config", "config/train_sft.yml"])

    assert args.bf16 is True
    assert args.fp16 is False
    assert args.attn_implementation == "flash_attention_2"


def test_model_kwargs_passes_attention_implementation() -> None:
    args = SimpleNamespace(load_4bit=False, attn_implementation="flash_attention_2")

    assert _model_kwargs(args, "bf16")["attn_implementation"] == "flash_attention_2"


def test_acceleration_runtime_rejects_missing_flash_attention() -> None:
    args = SimpleNamespace(bf16=False, fp16=False, attn_implementation="flash_attention_2")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    with pytest.raises(SystemExit, match="flash_attn is not installed"):
        _validate_acceleration_runtime(args, torch, find_spec=lambda name: None)


def test_acceleration_runtime_rejects_flash_attention_abi_import_failure() -> None:
    args = SimpleNamespace(bf16=False, fp16=False, attn_implementation="flash_attention_2")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    def broken_import(name):
        raise ImportError("flash_attn_2_cuda.so: undefined symbol: c10::Error")

    with pytest.raises(SystemExit, match="undefined symbol: c10::Error"):
        _validate_acceleration_runtime(
            args,
            torch,
            find_spec=lambda name: object(),
            import_module=broken_import,
        )


def test_acceleration_runtime_rejects_flash_attention_without_cuda() -> None:
    args = SimpleNamespace(bf16=False, fp16=False, attn_implementation="flash_attention_2")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))

    with pytest.raises(SystemExit, match="CUDA is unavailable"):
        _validate_acceleration_runtime(args, torch, find_spec=lambda name: object())


def test_acceleration_runtime_rejects_bf16_and_fp16_together() -> None:
    args = SimpleNamespace(bf16=True, fp16=True, attn_implementation="sdpa")
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))

    with pytest.raises(SystemExit, match="cannot both be enabled"):
        _validate_acceleration_runtime(args, torch)


def test_acceleration_runtime_selects_local_rank_before_bf16_check(monkeypatch) -> None:
    calls = []
    args = SimpleNamespace(bf16=True, fp16=False, attn_implementation="sdpa")
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            set_device=lambda rank: calls.append(("set_device", rank)),
            is_bf16_supported=lambda: calls.append(("is_bf16_supported", None)) or True,
        )
    )
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")

    _validate_acceleration_runtime(args, torch)

    assert calls == [("set_device", 1), ("is_bf16_supported", None)]


def test_acceleration_runtime_rejects_bf16_without_device_support() -> None:
    args = SimpleNamespace(bf16=True, fp16=False, attn_implementation="sdpa")
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: False,
        )
    )

    with pytest.raises(SystemExit, match="does not support bf16"):
        _validate_acceleration_runtime(args, torch, find_spec=lambda name: object())


def test_resolve_resume_checkpoint_requires_complete_trainer_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    for name in [
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ]:
        (checkpoint / name).write_text("state", encoding="utf-8")

    assert _resolve_resume_checkpoint(str(checkpoint)) == checkpoint

    (checkpoint / "optimizer.pt").unlink()
    with pytest.raises(SystemExit, match="optimizer.pt"):
        _resolve_resume_checkpoint(str(checkpoint))


def test_resolve_resume_checkpoint_accepts_distributed_rng_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    for name in [
        "adapter_model.safetensors",
        "adapter_config.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state_0.pth",
    ]:
        (checkpoint / name).write_text("state", encoding="utf-8")

    assert _resolve_resume_checkpoint(checkpoint) == checkpoint


def test_validate_resume_runtime_files_requires_every_rank_and_fp16_scaler(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-20"
    checkpoint.mkdir()
    (checkpoint / "rng_state_0.pth").write_text("rng", encoding="utf-8")

    with pytest.raises(SystemExit, match="rng_state_1.pth"):
        _validate_resume_runtime_files(checkpoint, world_size=2, fp16=False)

    (checkpoint / "rng_state_1.pth").write_text("rng", encoding="utf-8")
    with pytest.raises(SystemExit, match="scaler.pt"):
        _validate_resume_runtime_files(checkpoint, world_size=2, fp16=True)

    (checkpoint / "scaler.pt").write_text("scaler", encoding="utf-8")
    _validate_resume_runtime_files(checkpoint, world_size=2, fp16=True)


def test_run_trainer_passes_explicit_resume_checkpoint(tmp_path: Path) -> None:
    class DummyTrainer:
        def __init__(self) -> None:
            self.kwargs = None

        def train(self, **kwargs):
            self.kwargs = kwargs

    trainer = DummyTrainer()
    checkpoint = tmp_path / "checkpoint-20"

    _run_trainer(trainer, checkpoint)

    assert trainer.kwargs == {"resume_from_checkpoint": str(checkpoint)}


def test_resume_sampler_mode_preserves_legacy_order_and_new_random_runs(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-20"
    checkpoint.mkdir(parents=True)

    assert _resume_uses_random_sampler(checkpoint) is False

    (checkpoint.parent / "sft_run_manifest.json").write_text(
        json.dumps({"schema_version": 1, "train_sampler": "random"}),
        encoding="utf-8",
    )
    assert _resume_uses_random_sampler(checkpoint) is True


def test_resume_manifest_rejects_changed_training_contract(tmp_path: Path) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-20"
    checkpoint.mkdir(parents=True)
    (checkpoint.parent / "sft_run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_path": "model/Qwen2.5-7B-Instruct",
                "data_root": "data/sft/v2",
                "prompt_contract_fingerprint": "old-fingerprint",
                "train_sampler": "random",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="prompt_contract_fingerprint"):
        _validate_resume_compatibility(
            checkpoint,
            {
                "model_path": "model/Qwen2.5-7B-Instruct",
                "data_root": "data/sft/v2",
                "prompt_contract_fingerprint": "new-fingerprint",
                "train_sampler": "random",
            },
        )


def test_write_json_atomic_replaces_complete_payload(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"old": true}', encoding="utf-8")

    _write_json_atomic(path, {"schema_version": 1, "train_sampler": "random"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "train_sampler": "random",
    }
    assert not list(tmp_path.glob("*.tmp"))


def test_training_arguments_use_epoch_evaluation_without_eval_steps(tmp_path: Path) -> None:
    class DummyTrainingArguments:
        def __init__(self, *, eval_strategy=None, **kwargs):
            self.kwargs = {"eval_strategy": eval_strategy, **kwargs}

    args = parse_args(["--config", "config/train_sft.yml"])
    train_args = _training_arguments(args, tmp_path, True, DummyTrainingArguments)

    assert train_args.kwargs["eval_strategy"] == "epoch"
    assert train_args.kwargs["eval_steps"] is None
    assert train_args.kwargs["save_strategy"] == "steps"


def _record(qid: str, action_type: str) -> TrajectoryRecord:
    return TrajectoryRecord(
        qid=qid,
        question=f"question {qid}",
        dataset="toy",
        action_type=action_type,
        prompt_text=f"prompt {qid} {action_type}",
        target_text=f"target {qid} {action_type}",
    )


def test_split_training_samples_uses_original_samples_without_qid_overlap() -> None:
    samples = [
        TrainingSample(qid="s1", dataset="toy", records=[_record("s1", "query"), _record("s1", "evidence_update")]),
        TrainingSample(qid="s2", dataset="toy", records=[_record("s2", "query")]),
        TrainingSample(qid="s3", dataset="toy", records=[_record("s3", "query"), _record("s3", "evidence_update")]),
        TrainingSample(qid="s4", dataset="toy", records=[_record("s4", "query")]),
    ]

    train_samples, val_samples = split_training_samples(samples, ratio=0.5, seed=7)

    train_qids = {sample.qid for sample in train_samples}
    val_qids = {sample.qid for sample in val_samples}
    assert train_qids.isdisjoint(val_qids)
    assert len(train_samples) == 2
    assert len(val_samples) == 2
    assert [sample.qid for sample in train_samples] == ["s1", "s3"]
    assert [sample.qid for sample in val_samples] == ["s2", "s4"]


def test_flatten_training_samples_preserves_sample_then_action_order() -> None:
    samples = [
        TrainingSample(qid="s1", dataset="toy", records=[_record("s1", "query"), _record("s1", "evidence_update")]),
        TrainingSample(qid="s2", dataset="toy", records=[_record("s2", "query"), _record("s2", "evidence_update")]),
    ]

    flattened = flatten_training_samples(samples)

    assert [(record.qid, record.action_type) for record in flattened] == [
        ("s1", "query"),
        ("s1", "evidence_update"),
        ("s2", "query"),
        ("s2", "evidence_update"),
    ]


def test_make_train_sampler_uses_random_sampling_on_one_process() -> None:
    from torch.utils.data import RandomSampler

    sampler = _make_train_sampler(dataset=range(4), world_size=1, process_rank=0)

    assert isinstance(sampler, RandomSampler)


def test_custom_samplers_leave_distributed_sharding_to_accelerate() -> None:
    from torch.utils.data import RandomSampler

    train_sampler = _make_train_sampler(dataset=range(8), world_size=2, process_rank=1)
    eval_dataset = [{"input_ids": list(range(length))} for length in range(1, 9)]
    eval_sampler = _make_length_grouped_eval_sampler(
        eval_dataset,
        world_size=2,
        process_rank=1,
    )

    assert isinstance(train_sampler, RandomSampler)
    assert sorted(iter(eval_sampler)) == list(range(8))


def test_target_only_trainer_train_sampler_accepts_transformers_dataset_argument() -> None:
    class DummyTrainer:
        train_dataset = range(3)

    trainer = _make_target_only_trainer_cls(DummyTrainer)()

    sampler = trainer._get_train_sampler(range(4))

    from torch.utils.data import RandomSampler

    assert isinstance(sampler, RandomSampler)


def test_target_only_trainer_can_preserve_legacy_sequential_resume_order() -> None:
    class DummyTrainer:
        train_dataset = range(3)

    trainer = _make_target_only_trainer_cls(DummyTrainer, train_shuffle=False)()

    sampler = trainer._get_train_sampler(range(4))

    assert list(iter(sampler)) == [0, 1, 2, 3]


def test_target_only_trainer_counts_non_padding_training_tokens() -> None:
    import torch

    class DummyTrainer:
        def training_step(self, model, inputs, num_items_in_batch=None):
            return "loss"

    trainer = _make_target_only_trainer_cls(DummyTrainer)()
    model = SimpleNamespace()

    result = trainer.training_step(
        model,
        {"attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]])},
    )

    assert result == "loss"
    assert model._macorag_train_token_count == 3


def test_eval_target_loss_is_invariant_to_length_grouped_batch_composition() -> None:
    import torch

    logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
            [[0.0, 2.0], [2.0, 0.0], [1.0, 1.0]],
            [[1.0, 1.0], [0.0, 2.0], [2.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0, 1, -100], [1, -100, -100], [0, 1, 0]])

    full = _mean_per_example_target_loss(logits, labels)
    regrouped = (
        _mean_per_example_target_loss(logits[:2], labels[:2]) * 2
        + _mean_per_example_target_loss(logits[2:], labels[2:])
    ) / 3

    assert torch.allclose(full, regrouped)


def test_target_only_trainer_uses_per_example_macro_loss_during_eval() -> None:
    import torch

    logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]], [[0.0, 2.0], [2.0, 0.0]]])

    class DummyTrainer:
        pass

    class DummyModel:
        training = False

        def __call__(self, **kwargs):
            return {"loss": torch.tensor(99.0), "logits": logits}

    labels = torch.tensor([[-100, 0], [-100, 1]])
    trainer = _make_target_only_trainer_cls(DummyTrainer)()

    loss = trainer.compute_loss(
        DummyModel(),
        {"input_ids": torch.ones((2, 2), dtype=torch.long), "labels": labels},
    )

    shifted = torch.tensor([[0, -100], [1, -100]])
    assert torch.allclose(loss, _mean_per_example_target_loss(logits, shifted))


def test_length_grouped_eval_sampler_covers_each_item_once_and_reduces_padding() -> None:
    dataset = [
        {"input_ids": list(range(length))}
        for length in [9, 2, 8, 3, 7, 4, 6, 5]
    ]

    sampler = _make_length_grouped_eval_sampler(dataset, world_size=1, process_rank=0)
    indices = list(iter(sampler))

    assert sorted(indices) == list(range(len(dataset)))
    lengths = [len(dataset[index]["input_ids"]) for index in indices]
    assert lengths == sorted(lengths)
    assert all(max(lengths[i : i + 2]) - min(lengths[i : i + 2]) <= 1 for i in range(0, len(lengths), 2))


def test_training_shell_script_derives_gpu_visibility_from_yaml() -> None:
    script = Path("scripts/run_train_sft.sh").read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES="0,1"' not in script
    assert "yaml.safe_load" in script
    assert "YAML_GPU_INDICES" in script
    assert "gpu_index" not in script
    assert "NPROC_PER_NODE" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_GPU_INDICES}"' in script
    assert "torchrun" in script
    assert "--nproc_per_node=${NPROC_PER_NODE}" in script
