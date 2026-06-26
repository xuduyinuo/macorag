from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from sft_training.train_sft_lora_macorag import (
    TrajectoryRecord,
    _make_eval_metrics_callback,
    _make_jsonl_logging_callback,
    _make_timestamped_output_dir,
    _resolve_run_log_path,
    _tokenize_records,
    parse_args,
    trajectory_to_sft_records,
)
from sft_training.data import (
    TrainingSample,
    flatten_training_samples,
    split_training_samples,
)
from sft_training.trainer import _make_ordered_sampler, _make_target_only_trainer_cls


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

    answer_record = records[2]
    assert answer_record.action_type == "answer"
    assert answer_record.agent_role == "answer_generator"
    assert "<state>" in answer_record.prompt_text
    assert "<observation>" not in answer_record.prompt_text
    assert "<update-evidence>" not in answer_record.prompt_text
    assert "evidence leak" in answer_record.prompt_text
    assert answer_record.prompt_text.startswith("Task: answer from accumulated evidence.")
    assert "You are a retrieval-augmented reasoning assistant" not in answer_record.prompt_text
    assert "<answer>" in answer_record.target_text
    assert "<plan>" not in answer_record.target_text
    assert "<retrieval>" not in answer_record.target_text
    assert "<update-evidence>" not in answer_record.target_text


def test_parse_args_loads_yaml_config(tmp_path) -> None:
    config = tmp_path / "train.yml"
    config.write_text(
        "\n".join(
            [
                'model_path: "model/from-yaml"',
                'data_root: "data/from-yaml"',
                'output_dir: "outputs/from-yaml"',
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
                'log_jsonl_path: "outputs/from-yaml/train_metrics.jsonl"',
                'gpu_indices: "0,1"',
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.model_path == "model/from-yaml"
    assert args.data_root == "data/from-yaml"
    assert args.output_dir == "outputs/from-yaml"
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
    assert args.log_jsonl_path == "outputs/from-yaml/train_metrics.jsonl"
    assert args.gpu_indices == "0,1"


def test_parse_args_cli_overrides_yaml_config(tmp_path) -> None:
    config = tmp_path / "train.yml"
    config.write_text(
        "\n".join(
            [
                'output_dir: "outputs/from-yaml"',
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
            "--output-dir",
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

    assert args.output_dir == "outputs/from-cli"
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


def test_jsonl_logging_callback_writes_one_line_per_trained_sample(tmp_path) -> None:
    class DummyCallback:
        pass

    log_path = tmp_path / "train_metrics.jsonl"
    callback = _make_jsonl_logging_callback(
        log_path,
        DummyCallback,
        samples_per_epoch=10,
        total_epochs=1.0,
    )

    callback.on_log(
        None,
        SimpleNamespace(epoch=0.25, global_step=1),
        None,
        {"loss": 1.2, "grad_norm": 3.4, "learning_rate": 0.0001},
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"epoch": 1, "sample": 1, "sample_total": 10, "loss": 1.2, "grad_norm": 3.4, "learning_rate": 0.0001},
        {"epoch": 1, "sample": 2, "sample_total": 10, "loss": 1.2, "grad_norm": 3.4, "learning_rate": 0.0001},
    ]


def test_eval_metrics_callback_writes_one_line_per_eval(tmp_path) -> None:
    class DummyCallback:
        pass

    log_path = tmp_path / "eval_metrics.jsonl"
    callback = _make_eval_metrics_callback(log_path, DummyCallback)

    callback.on_evaluate(
        None,
        SimpleNamespace(epoch=1.25, global_step=120),
        None,
        {"eval_loss": 0.12, "eval_runtime": 3.4, "eval_samples_per_second": 5.6},
    )

    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "step": 120,
            "epoch": 1.25,
            "eval_loss": 0.12,
            "eval_runtime": 3.4,
            "eval_samples_per_second": 5.6,
        }
    ]


def test_timestamped_output_dir_uses_base_name_and_timestamp() -> None:
    assert _make_timestamped_output_dir("outputs/lora_qwen2.5-7b_trajectory", "20260625_153012").as_posix() == (
        "outputs/lora_qwen2.5-7b_trajectory_20260625_153012"
    )


def test_run_log_path_moves_base_log_into_timestamped_run_dir() -> None:
    base_output_dir = "outputs/lora_qwen2.5-7b_trajectory"
    run_output_dir = _make_timestamped_output_dir("outputs/lora_qwen2.5-7b_trajectory", "20260625_153012")

    assert _resolve_run_log_path(
        "outputs/lora_qwen2.5-7b_trajectory/train_metrics.jsonl",
        Path(base_output_dir),
        run_output_dir,
    ).as_posix() == "outputs/lora_qwen2.5-7b_trajectory_20260625_153012/train_metrics.jsonl"


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


def test_make_ordered_sampler_disables_random_training_shuffle() -> None:
    sampler = _make_ordered_sampler(dataset=range(4), world_size=1, process_rank=0)

    assert list(iter(sampler)) == [0, 1, 2, 3]


def test_ordered_trainer_train_sampler_accepts_transformers_dataset_argument() -> None:
    class DummyTrainer:
        train_dataset = range(3)

    trainer = _make_target_only_trainer_cls(DummyTrainer)()

    sampler = trainer._get_train_sampler(range(4))

    assert list(iter(sampler)) == [0, 1, 2, 3]


def test_training_shell_script_launches_two_gpu_torchrun() -> None:
    script = Path("scripts/run_train_sft_lora_gpu1.sh").read_text(encoding="utf-8")

    assert 'CUDA_VISIBLE_DEVICES="0,1"' in script
    assert "torchrun" in script
    assert "--nproc_per_node=2" in script
