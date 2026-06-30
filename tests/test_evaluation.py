from __future__ import annotations

import json
import os
from types import SimpleNamespace
from pathlib import Path
import socket

import pytest

from evaluation.config import parse_args
from evaluation.data import EvalSample, load_eval_samples
from evaluation.evaluate_rag_model import (
    _build_retrieval_env,
    _configure_visible_gpus,
    _torch_dtype,
    format_prediction,
    main,
    run_predictions,
)


from evaluation.bailian_evaluator import calculate_contain, calculate_llm_accuracy, evaluate_predictions


def test_evaluate_configure_visible_gpus_respects_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(gpu_indices="1", gpu_index=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    _configure_visible_gpus(args)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_evaluate_build_retrieval_env_uses_eval_retrieval_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class FakeRetrievalEnv:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("evaluation.evaluate_rag_model.CachedLinearRAGRetrievalEnv", FakeRetrievalEnv)
    args = SimpleNamespace(
        retrieval_root="data/eval_1000_retrieval",
        retrieval_embedding_model="sentence-transformers/all-mpnet-base-v2",
        retrieval_spacy_model="en_core_web_trf",
        retrieval_top_k=5,
        retrieval_max_workers=4,
        retrieval_batch_size=32,
        use_vectorized_retrieval=True,
    )

    _build_retrieval_env(args)

    assert captured["retrieval_root"] == "data/eval_1000_retrieval"
    assert captured["embedding_model"] == "sentence-transformers/all-mpnet-base-v2"
    assert captured["top_k"] == 5


def test_evaluate_main_passes_judge_metadata_to_evaluate_predictions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class FakeJudgeClient:
        def __init__(self, **kwargs) -> None:
            captured["judge_client_kwargs"] = kwargs

    def fake_evaluate_predictions(predictions_path: Path, *, client, max_workers: int, judge_metadata=None):
        captured["predictions_path"] = predictions_path
        captured["client"] = client
        captured["max_workers"] = max_workers
        captured["judge_metadata"] = judge_metadata
        return {
            "llm_accuracy": 0.0,
            "contain_accuracy": 0.0,
            "num_samples": 0,
            "judge_metadata": judge_metadata,
        }

    args = SimpleNamespace(
        model_path="model/base",
        adapter_path="outputs/grpo/adapter",
        data_root="data/eval_1000",
        data_files=[],
        retrieval_root="data/eval_1000_retrieval",
        output_dir=str(tmp_path / "outputs"),
        fixed_output_dir=True,
        system_prompt="sys",
        max_samples=None,
        seed=42,
        max_rounds=3,
        max_prompt_length=128,
        max_completion_length=16,
        temperature=0.0,
        top_p=0.95,
        top_k=5,
        bf16=False,
        fp16=False,
        load_4bit=False,
        gpu_index=0,
        gpu_indices="1",
        disable_tqdm=True,
        retrieval_embedding_model="sentence-transformers/all-mpnet-base-v2",
        retrieval_spacy_model="en_core_web_trf",
        retrieval_top_k=5,
        retrieval_max_workers=4,
        retrieval_batch_size=32,
        use_vectorized_retrieval=True,
        skip_judge=False,
        judge_model="qwen-plus",
        judge_endpoint="https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        judge_api_key_env="DASHSCOPE_API_KEY",
        judge_temperature=0.0,
        judge_max_tokens=8,
        judge_timeout=120,
        judge_retries=3,
        judge_retry_sleep_seconds=2.0,
        judge_workers=4,
    )

    monkeypatch.setattr("evaluation.evaluate_rag_model.parse_args", lambda argv=None: args)
    monkeypatch.setattr("evaluation.evaluate_rag_model._configure_visible_gpus", lambda parsed_args: None)
    monkeypatch.setattr("evaluation.evaluate_rag_model._resolved_output_dir", lambda parsed_args: tmp_path / "eval_run")
    monkeypatch.setattr("evaluation.evaluate_rag_model.load_eval_samples", lambda **kwargs: ([], {"loaded_samples": 0}))
    monkeypatch.setattr("evaluation.evaluate_rag_model._load_policy", lambda parsed_args: object())
    monkeypatch.setattr("evaluation.evaluate_rag_model._build_retrieval_env", lambda parsed_args: object())
    monkeypatch.setattr("evaluation.evaluate_rag_model.run_predictions", lambda *args, **kwargs: [])
    monkeypatch.setattr("evaluation.evaluate_rag_model.BailianJudgeClient", FakeJudgeClient)
    monkeypatch.setattr("evaluation.evaluate_rag_model.evaluate_predictions", fake_evaluate_predictions)

    assert main([]) == 0
    assert captured["judge_metadata"] == {
        "judge_model": "qwen-plus",
        "judge_endpoint": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "judge_api_key_env": "DASHSCOPE_API_KEY",
        "judge_temperature": 0.0,
        "judge_max_tokens": 8,
        "judge_timeout": 120,
        "judge_retries": 3,
        "judge_retry_sleep_seconds": 2.0,
        "judge_workers": 4,
    }
    assert captured["max_workers"] == 4


def test_evaluate_torch_dtype_uses_float32_on_cpu_when_no_precision_flag_is_set() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        float16="float16",
        float32="float32",
        bfloat16="bfloat16",
    )
    args = SimpleNamespace(fp16=False, bf16=False)

    assert _torch_dtype(args, fake_torch) == fake_torch.float32


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


def test_evaluate_predictions_preserves_falsy_answers(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {"qid": "q0", "pred_answer": 0, "gold_answer": 0},
                {"qid": "q1", "pred_answer": False, "gold_answer": False},
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    client = FakeJudgeClient(["correct", "correct"])

    summary = evaluate_predictions(predictions_path, client=client, max_workers=1)

    updated = json.loads(predictions_path.read_text(encoding="utf-8"))
    assert summary["contain_accuracy"] == 1.0
    assert updated[0]["contain_accuracy"] == 1
    assert updated[1]["contain_accuracy"] == 1


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload, ensure_ascii=False).encode("utf-8")

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None


def test_bailian_judge_client_retries_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, int] = {"count": 0}

    def fake_urlopen(_request: object, timeout: int) -> _FakeHTTPResponse:
        called["count"] += 1
        if called["count"] == 1:
            raise socket.timeout("temporary timeout")
        return _FakeHTTPResponse({"choices": [{"message": {"content": "correct"}}]})

    monkeypatch.setenv("TEST_BAILIAN_API_KEY", "k")
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    from evaluation.bailian_evaluator import BailianJudgeClient

    client = BailianJudgeClient(
        model="test",
        endpoint="https://example.com/api/v1/chat/completions",
        api_key_env="TEST_BAILIAN_API_KEY",
        temperature=0.0,
        max_tokens=16,
        timeout=1,
        retries=2,
        retry_sleep_seconds=0.0,
    )
    result = client.infer([{"role": "user", "content": "Is this correct?"}])

    assert result == "correct"
    assert called["count"] == 2


def test_evaluate_predictions_includes_judge_metadata_when_provided(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps([{"qid": "q1", "pred_answer": "David Arquette", "gold_answer": "David Arquette"}], ensure_ascii=False),
        encoding="utf-8",
    )
    client = FakeJudgeClient(["correct"])
    metadata = {"judge_model": "test-model", "temperature": 0.0}

    summary = evaluate_predictions(
        predictions_path,
        client=client,
        max_workers=1,
        judge_metadata=metadata,
    )
    evaluation_results = json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8"))

    assert summary["judge_metadata"] == metadata
    assert evaluation_results["judge_metadata"] == metadata


def test_format_prediction_matches_linearrag_evaluator_schema() -> None:
    sample = EvalSample(
        qid="q1",
        dataset="hotpotqa",
        question="Who directed The Tripper?",
        answer="David Arquette",
        answer_aliases=["Arquette"],
        supporting_facts=[],
        metadata={"split": "dev"},
    )
    result = SimpleNamespace(
        final_answer="David Arquette",
        trajectory=[{"round": 0}],
        parse_errors=[],
        state=SimpleNamespace(retrieval_count=1),
    )

    prediction = format_prediction(sample, result)

    assert prediction["qid"] == "q1"
    assert prediction["dataset"] == "hotpotqa"
    assert prediction["pred_answer"] == "David Arquette"
    assert prediction["gold_answer"] == "David Arquette"
    assert prediction["answer_aliases"] == ["Arquette"]
    assert prediction["trajectory"] == [{"round": 0}]
    assert prediction["parse_errors"] == []
    assert prediction["retrieval_count"] == 1


class FakePolicy:
    pass


class FakeRetrievalEnv:
    def query(self, dataset: str, query: str) -> dict:
        return {"query": query, "passages": []}


def test_run_predictions_flushes_jsonl_progress(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = EvalSample("q1", "hotpotqa", "Question?", "Answer", [], [], {})
    args = SimpleNamespace(max_rounds=1, disable_tqdm=True)

    class FakeExecutor:
        def __init__(self, *, policy, retrieval_env, max_rounds: int) -> None:
            self.max_rounds = max_rounds

        def run(self, *, question: str, dataset: str):
            return SimpleNamespace(
                final_answer="Answer",
                trajectory=[{"round": 0}],
                parse_errors=[],
                state=SimpleNamespace(retrieval_count=0),
            )

    monkeypatch.setattr("evaluation.evaluate_rag_model.RAGLoopExecutor", FakeExecutor)

    predictions = run_predictions(args, [sample], FakePolicy(), FakeRetrievalEnv(), tmp_path)

    assert predictions[0]["pred_answer"] == "Answer"
    assert json.loads((tmp_path / "predictions.json").read_text(encoding="utf-8"))[0]["qid"] == "q1"
    progress_lines = (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(progress_lines) == 1
    assert json.loads(progress_lines[0])["qid"] == "q1"


def test_run_predictions_re_raises_missing_index_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sample = EvalSample("q1", "hotpotqa", "Question?", "Answer", [], [], {})
    args = SimpleNamespace(max_rounds=1, disable_tqdm=True)

    class FakeExecutor:
        def __init__(self, *, policy, retrieval_env, max_rounds: int) -> None:
            self.max_rounds = max_rounds

        def run(self, *, question: str, dataset: str):
            raise FileNotFoundError("LinearRAG index not found")

    monkeypatch.setattr("evaluation.evaluate_rag_model.RAGLoopExecutor", FakeExecutor)

    with pytest.raises(FileNotFoundError):
        run_predictions(args, [sample], FakePolicy(), FakeRetrievalEnv(), tmp_path)

    assert not (tmp_path / "predictions.jsonl").exists()
    assert not (tmp_path / "predictions.json").exists()
