from __future__ import annotations

import json
from pathlib import Path

import pytest

from evaluation.config import parse_args
from evaluation.data import load_eval_samples


from evaluation.bailian_evaluator import calculate_contain, calculate_llm_accuracy, evaluate_predictions


def test_parse_eval_config_loads_yaml_and_cli_overrides(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text(
        "\n".join(
            [
                'model_path: "model/base"',
                'adapter_path: "outputs/grpo/adapter"',
                'data_root: "data/eval_1000"',
                'retrieval_root: "data/eval_1000_retrieval"',
                'output_dir: "outputs/eval"',
                'judge_model: "qwen-plus"',
                'judge_api_key_env: "DASHSCOPE_API_KEY"',
                "max_samples: 20",
                "max_rounds: 2",
                "retrieval_top_k: 4",
                'gpu_indices: "1"',
                "skip_judge: false",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--max-samples", "3", "--skip-judge"])

    assert args.model_path == "model/base"
    assert args.adapter_path == "outputs/grpo/adapter"
    assert args.data_root == "data/eval_1000"
    assert args.retrieval_root == "data/eval_1000_retrieval"
    assert args.output_dir == "outputs/eval"
    assert args.judge_model == "qwen-plus"
    assert args.judge_api_key_env == "DASHSCOPE_API_KEY"
    assert args.max_samples == 3
    assert args.max_rounds == 2
    assert args.retrieval_top_k == 4
    assert args.gpu_indices == "1"
    assert args.skip_judge is True


def test_parse_eval_config_rejects_unknown_yaml_keys(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text("unknown_key: 1\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Unknown evaluation config keys"):
        parse_args(["--config", str(config)])


def test_parse_eval_config_rejects_missing_explicit_config_with_equals(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_config.yml"

    with pytest.raises(SystemExit, match="Evaluation config not found"):
        parse_args([f"--config={missing}"])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_eval_samples_skips_corpus_and_normalizes_gold_answer(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_dev.jsonl",
        [
            {
                "qid": "q1",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "gold_answer": "David Arquette",
                "answer_aliases": ["Arquette"],
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
                "metadata": {"split": "dev"},
            },
            {
                "qid": "bad",
                "dataset": "hotpotqa",
                "question": "",
                "answer": "missing question",
                "supporting_facts": [],
            },
        ],
    )
    _write_jsonl(data_root / "hotpotqa" / "corpus.jsonl", [{"doc_id": "d1", "text": "not a sample"}])

    samples, summary = load_eval_samples(data_root=data_root, data_files=[], max_samples=None)

    assert len(samples) == 1
    assert samples[0].qid == "q1"
    assert samples[0].dataset == "hotpotqa"
    assert samples[0].answer == "David Arquette"
    assert samples[0].answer_aliases == ["Arquette"]
    assert samples[0].metadata == {"split": "dev"}
    assert summary["loaded_samples"] == 1
    assert summary["skipped_samples"] == 1
    assert summary["source_files"] == [str(data_root / "hotpotqa" / "hotpotqa_dev.jsonl")]


def test_load_eval_samples_uses_gold_answer_when_answer_is_null(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_dev.jsonl",
        [
            {
                "qid": "q2",
                "dataset": "hotpotqa",
                "question": "Who is the director?",
                "answer": None,
                "gold_answer": "David Arquette",
                "answer_aliases": ["Arquette"],
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            },
        ],
    )

    samples, _ = load_eval_samples(data_root=data_root, data_files=[], max_samples=None)

    assert len(samples) == 1
    assert samples[0].answer == "David Arquette"


def test_load_eval_samples_stops_at_max_samples_before_later_missing_file(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(
        data_root / "hotpotqa" / "first.jsonl",
        [
            {
                "qid": "q3",
                "dataset": "hotpotqa",
                "question": "Director?",
                "answer": "David Arquette",
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            },
        ],
    )

    samples, summary = load_eval_samples(
        data_root=data_root,
        data_files=[
            str(data_root / "hotpotqa" / "first.jsonl"),
            str(data_root / "hotpotqa" / "missing.jsonl"),
        ],
        max_samples=1,
    )

    assert len(samples) == 1
    assert summary["loaded_samples"] == 1
    assert summary["source_files"] == [str(data_root / "hotpotqa" / "first.jsonl")]


def test_explicit_data_files_skips_corpus_jsonl(tmp_path: Path) -> None:
    data_root = tmp_path / "eval"
    _write_jsonl(
        data_root / "hotpotqa" / "corpus.jsonl",
        [{"doc_id": "d1", "text": "should be skipped"}],
    )
    _write_jsonl(
        data_root / "hotpotqa" / "hotpotqa_dev.jsonl",
        [
            {
                "qid": "q4",
                "dataset": "hotpotqa",
                "question": "Who directed The Tripper?",
                "answer": "David Arquette",
                "supporting_facts": [{"title": "The Tripper", "text": "Directed by David Arquette."}],
            }
        ],
    )

    samples, summary = load_eval_samples(
        data_root=data_root,
        data_files=[
            str(Path("hotpotqa") / "corpus.jsonl"),
            str(Path("hotpotqa") / "hotpotqa_dev.jsonl"),
        ],
    )

    assert len(samples) == 1
    assert summary["loaded_samples"] == 1
    assert summary["source_files"] == [str(data_root / "hotpotqa" / "hotpotqa_dev.jsonl")]


class FakeJudgeClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.messages: list[list[dict[str, str]]] = []

    def infer(self, messages: list[dict[str, str]]) -> str:
        self.messages.append(messages)
        return self.responses.pop(0)


def test_bailian_evaluator_maps_correct_response_and_contain_accuracy() -> None:
    client = FakeJudgeClient(["correct"])

    llm_acc = calculate_llm_accuracy(client, "David Arquette", "David Arquette")

    assert llm_acc == 1.0
    assert calculate_contain("The answer is David Arquette.", "David Arquette") == 1
    assert calculate_contain("The answer is Wes Craven.", "David Arquette") == 0
    assert "Respond with ONLY 'correct' or 'incorrect'." in client.messages[0][1]["content"]


def test_evaluate_predictions_updates_prediction_file_and_summary(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {"qid": "q1", "pred_answer": "David Arquette", "gold_answer": "David Arquette"},
                {"qid": "q2", "pred_answer": "wrong", "gold_answer": "Right"},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = FakeJudgeClient(["correct", "incorrect"])

    summary = evaluate_predictions(predictions_path, client=client, max_workers=1)

    updated = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert summary["llm_accuracy"] == 0.5
    assert summary["contain_accuracy"] == 0.5
    assert summary["num_samples"] == 2
    assert updated[0]["llm_accuracy"] == 1.0
    assert updated[1]["llm_accuracy"] == 0.0
    assert (tmp_path / "evaluation_results.json").exists()
