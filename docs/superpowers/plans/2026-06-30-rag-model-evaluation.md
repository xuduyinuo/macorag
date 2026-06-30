# RAG Model Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent MACORAG post-SFT/RL evaluation module that runs the existing RAG loop on `data/eval_1000`, retrieves from `data/eval_1000_retrieval`, writes predictions, and evaluates answers with Bailian `qwen-plus`.

**Architecture:** Create `src/evaluation/` as a separate package from SFT/RL training. Reuse `HFSharedPolicy`, `RAGLoopExecutor`, and `CachedLinearRAGRetrievalEnv` for model inference and retrieval, then evaluate `pred_answer` versus `gold_answer` with a local Bailian evaluator compatible with `LinearRAG/src/evaluate.py`.

**Tech Stack:** Python 3, PyYAML, pytest, HuggingFace Transformers, PEFT, PyTorch, LinearRAG retrieval assets, Alibaba DashScope OpenAI-compatible chat endpoint.

## Global Constraints

- Test data root is `data/eval_1000`.
- Retrieval root is `data/eval_1000_retrieval`.
- Python execution code lives under `src/`.
- The shell launcher lives under `scripts/`.
- Runtime parameters are passed through `config/evaluate_rag_model.yml`.
- The module must not rebuild retrieval indexes, extract datasets, train adapters, or change SFT/RL training entrypoints.
- The evaluator uses Bailian `qwen-plus` by default through `DASHSCOPE_API_KEY`.
- New production behavior must be introduced test-first.

---

## File Structure

- Create `src/evaluation/__init__.py`
  - Re-export stable helpers used by tests.
- Create `src/evaluation/config.py`
  - Define defaults and parse YAML plus CLI overrides.
- Create `src/evaluation/data.py`
  - Load and normalize evaluation samples from JSONL files.
- Create `src/evaluation/bailian_evaluator.py`
  - Implement contain accuracy, Bailian judge calls, and evaluation artifact writing.
- Create `src/evaluation/evaluate_rag_model.py`
  - Load model/adapter, run RAG predictions, write artifacts, and optionally run judge evaluation.
- Create `config/evaluate_rag_model.yml`
  - Provide default evaluation parameters.
- Create `scripts/evaluate_rag_model.sh`
  - Thin YAML-driven launcher.
- Create `tests/test_evaluation.py`
  - Cover config, data, prediction formatting, script behavior, and judge parsing.

---

### Task 1: Evaluation Config Parser

**Files:**
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/config.py`
- Create: `tests/test_evaluation.py`

**Interfaces:**
- Produces: `evaluation.config.parse_args(argv: list[str] | None = None) -> argparse.Namespace`
- Produces: `evaluation.config.DEFAULT_CONFIG_PATH: str`
- Produces: `evaluation.config.DEFAULT_ARG_VALUES: dict[str, Any]`

- [ ] **Step 1: Write failing config tests**

Add to `tests/test_evaluation.py`:

```python
from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.config import parse_args


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
                "gpu_indices: \"1\"",
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_parse_eval_config_loads_yaml_and_cli_overrides tests/test_evaluation.py::test_parse_eval_config_rejects_unknown_yaml_keys`

Expected: FAIL because `evaluation.config` does not exist.

- [ ] **Step 3: Implement config parser**

Create `src/evaluation/__init__.py`:

```python
from __future__ import annotations
```

Create `src/evaluation/config.py`:

```python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


DEFAULT_CONFIG_PATH = "config/evaluate_rag_model.yml"

DEFAULT_ARG_VALUES: dict[str, Any] = {
    "model_path": "model/Qwen2.5-3B-Instruct",
    "adapter_path": "outputs/grpo_qwen2.5-3b/adapter",
    "data_root": "data/eval_1000",
    "data_files": (),
    "retrieval_root": "data/eval_1000_retrieval",
    "output_dir": "outputs/eval_rag_model",
    "fixed_output_dir": False,
    "system_prompt": "Follow the role-specific prompt. Output exactly the requested XML-style tag with valid JSON.",
    "max_samples": None,
    "seed": 42,
    "max_rounds": 3,
    "max_prompt_length": 4096,
    "max_completion_length": 256,
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 5,
    "bf16": False,
    "fp16": False,
    "load_4bit": True,
    "gpu_index": 0,
    "gpu_indices": "1",
    "disable_tqdm": False,
    "retrieval_embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "retrieval_spacy_model": "en_core_web_trf",
    "retrieval_top_k": 5,
    "retrieval_max_workers": 4,
    "retrieval_batch_size": 32,
    "use_vectorized_retrieval": True,
    "skip_judge": False,
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

BooleanOptionalAction = getattr(argparse, "BooleanOptionalAction", None)


def _load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to load evaluation YAML config.") from exc

    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid config format at {path}: expected a mapping.")

    config = {str(key).replace("-", "_"): value for key, value in payload.items()}
    allowed = {*DEFAULT_ARG_VALUES, "config"}
    unknown = sorted(set(config) - allowed)
    if unknown:
        raise SystemExit(f"Unknown evaluation config keys in {path}: {', '.join(unknown)}")
    return config


def _defaults_from_config(config_path: str, *, explicit_config: bool) -> dict[str, Any]:
    defaults = dict(DEFAULT_ARG_VALUES)
    path = Path(config_path)
    if explicit_config and not path.exists():
        raise SystemExit(f"Evaluation config not found: {path}")
    if path.exists():
        defaults.update(_load_yaml_config(path))
    return defaults


def _build_parser(defaults: dict[str, Any]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a MACORAG SFT/RL adapter with the configured RAG loop.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML config file.")
    parser.add_argument("--model-path", default=defaults["model_path"])
    parser.add_argument("--adapter-path", default=defaults["adapter_path"])
    parser.add_argument("--data-root", default=defaults["data_root"])
    parser.add_argument("--data-files", nargs="*", default=defaults["data_files"])
    parser.add_argument("--retrieval-root", default=defaults["retrieval_root"])
    parser.add_argument("--output-dir", default=defaults["output_dir"])
    parser.add_argument("--fixed-output-dir", action=BooleanOptionalAction, default=defaults["fixed_output_dir"])
    parser.add_argument("--system-prompt", default=defaults["system_prompt"])
    parser.add_argument("--max-samples", type=int, default=defaults["max_samples"])
    parser.add_argument("--seed", type=int, default=defaults["seed"])
    parser.add_argument("--max-rounds", type=int, default=defaults["max_rounds"])
    parser.add_argument("--max-prompt-length", type=int, default=defaults["max_prompt_length"])
    parser.add_argument("--max-completion-length", type=int, default=defaults["max_completion_length"])
    parser.add_argument("--temperature", type=float, default=defaults["temperature"])
    parser.add_argument("--top-p", type=float, default=defaults["top_p"])
    parser.add_argument("--top-k", type=int, default=defaults["top_k"])
    parser.add_argument("--bf16", action=BooleanOptionalAction, default=defaults["bf16"])
    parser.add_argument("--fp16", action=BooleanOptionalAction, default=defaults["fp16"])
    parser.add_argument("--load-4bit", action=BooleanOptionalAction, default=defaults["load_4bit"])
    parser.add_argument("--gpu-index", type=int, default=defaults["gpu_index"])
    parser.add_argument("--gpu-indices", default=defaults["gpu_indices"])
    parser.add_argument("--disable-tqdm", action=BooleanOptionalAction, default=defaults["disable_tqdm"])
    parser.add_argument("--retrieval-embedding-model", default=defaults["retrieval_embedding_model"])
    parser.add_argument("--retrieval-spacy-model", default=defaults["retrieval_spacy_model"])
    parser.add_argument("--retrieval-top-k", type=int, default=defaults["retrieval_top_k"])
    parser.add_argument("--retrieval-max-workers", type=int, default=defaults["retrieval_max_workers"])
    parser.add_argument("--retrieval-batch-size", type=int, default=defaults["retrieval_batch_size"])
    parser.add_argument("--use-vectorized-retrieval", action=BooleanOptionalAction, default=defaults["use_vectorized_retrieval"])
    parser.add_argument("--skip-judge", action=BooleanOptionalAction, default=defaults["skip_judge"])
    parser.add_argument("--judge-model", default=defaults["judge_model"])
    parser.add_argument("--judge-endpoint", default=defaults["judge_endpoint"])
    parser.add_argument("--judge-api-key-env", default=defaults["judge_api_key_env"])
    parser.add_argument("--judge-temperature", type=float, default=defaults["judge_temperature"])
    parser.add_argument("--judge-max-tokens", type=int, default=defaults["judge_max_tokens"])
    parser.add_argument("--judge-timeout", type=int, default=defaults["judge_timeout"])
    parser.add_argument("--judge-retries", type=int, default=defaults["judge_retries"])
    parser.add_argument("--judge-retry-sleep-seconds", type=float, default=defaults["judge_retry_sleep_seconds"])
    parser.add_argument("--judge-workers", type=int, default=defaults["judge_workers"])
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    raw_args = sys.argv[1:] if argv is None else argv
    explicit_config = "--config" in raw_args
    defaults = _defaults_from_config(config_args.config, explicit_config=explicit_config)
    parser = _build_parser(defaults)
    return parser.parse_args(argv)
```

- [ ] **Step 4: Run config tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_parse_eval_config_loads_yaml_and_cli_overrides tests/test_evaluation.py::test_parse_eval_config_rejects_unknown_yaml_keys`

Expected: PASS.

- [ ] **Step 5: Commit task 1**

Run:

```bash
git add src/evaluation/__init__.py src/evaluation/config.py tests/test_evaluation.py
git commit -m "feat: add rag evaluation config parser"
```

---

### Task 2: Evaluation Sample Loader

**Files:**
- Create: `src/evaluation/data.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `args.data_root`, `args.data_files`, `args.max_samples`
- Produces: `evaluation.data.EvalSample`
- Produces: `evaluation.data.load_eval_samples(data_root: str | Path, data_files: list[str] | tuple[str, ...] | None = None, max_samples: int | None = None) -> tuple[list[EvalSample], dict[str, Any]]`

- [ ] **Step 1: Write failing data-loader tests**

Add to `tests/test_evaluation.py`:

```python
import json

from evaluation.data import load_eval_samples


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_load_eval_samples_skips_corpus_and_normalizes_gold_answer`

Expected: FAIL because `evaluation.data` does not exist.

- [ ] **Step 3: Implement sample loader**

Create `src/evaluation/data.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalSample:
    qid: str
    dataset: str
    question: str
    answer: str
    answer_aliases: list[str]
    supporting_facts: list[dict[str, Any]]
    metadata: dict[str, Any]


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _candidate_files(data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    return sorted(path for path in data_root.rglob("*.jsonl") if path.is_file() and path.name != "corpus.jsonl")


def _resolve_files(data_root: Path, data_files: list[str] | tuple[str, ...]) -> list[Path]:
    if data_files:
        paths = []
        for item in data_files:
            path = Path(item)
            if not path.is_absolute():
                path = data_root / path
            paths.append(path)
        return paths
    return _candidate_files(data_root)


def _build_sample(row: dict[str, Any], fallback_dataset: str) -> EvalSample | None:
    qid = str(row.get("qid") or "").strip()
    dataset = str(row.get("dataset") or fallback_dataset or "").strip()
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer", row.get("gold_answer")) or "").strip()
    supporting_facts = row.get("supporting_facts")
    if not qid or not dataset or not question or not answer or not isinstance(supporting_facts, list):
        return None
    aliases = row.get("answer_aliases")
    if not isinstance(aliases, list):
        aliases = []
    metadata = row.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    return EvalSample(
        qid=qid,
        dataset=dataset,
        question=question,
        answer=answer,
        answer_aliases=[str(item) for item in aliases if item is not None],
        supporting_facts=[item for item in supporting_facts if isinstance(item, dict)],
        metadata=metadata,
    )


def load_eval_samples(
    *,
    data_root: str | Path,
    data_files: list[str] | tuple[str, ...] | None = None,
    max_samples: int | None = None,
) -> tuple[list[EvalSample], dict[str, Any]]:
    root = Path(data_root)
    files = _resolve_files(root, tuple(data_files or ()))
    if not files:
        raise FileNotFoundError(f"No evaluation jsonl files found under {root}")

    samples: list[EvalSample] = []
    skipped = 0
    counts_by_dataset: dict[str, int] = {}
    source_files: list[str] = []
    for path in files:
        if not path.exists():
            raise FileNotFoundError(f"Evaluation data file not found: {path}")
        source_files.append(str(path))
        fallback_dataset = path.parent.name or path.stem.replace("_dev", "")
        for row in _read_jsonl(path):
            sample = _build_sample(row, fallback_dataset)
            if sample is None:
                skipped += 1
                continue
            if max_samples is None or len(samples) < max_samples:
                samples.append(sample)
                counts_by_dataset[sample.dataset] = counts_by_dataset.get(sample.dataset, 0) + 1

    if not samples:
        raise ValueError(f"No valid evaluation samples found in {root}")
    summary = {
        "data_root": str(root),
        "source_files": source_files,
        "loaded_samples": len(samples),
        "skipped_samples": skipped,
        "counts_by_dataset": counts_by_dataset,
        "max_samples": max_samples,
    }
    return samples, summary
```

- [ ] **Step 4: Run data-loader test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_load_eval_samples_skips_corpus_and_normalizes_gold_answer`

Expected: PASS.

- [ ] **Step 5: Commit task 2**

Run:

```bash
git add src/evaluation/data.py tests/test_evaluation.py
git commit -m "feat: add rag evaluation sample loader"
```

---

### Task 3: Bailian-Compatible Evaluator

**Files:**
- Create: `src/evaluation/bailian_evaluator.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: prediction dicts with `pred_answer` and `gold_answer`
- Produces: `BailianJudgeClient.infer(messages: list[dict[str, str]]) -> str`
- Produces: `calculate_contain(pre_answer: str | None, gold_answer: str | None) -> int`
- Produces: `calculate_llm_accuracy(client: Any, pre_answer: str, gold_answer: str) -> float`
- Produces: `evaluate_predictions(predictions_path: str | Path, client: Any, max_workers: int) -> dict[str, Any]`

- [ ] **Step 1: Write failing evaluator tests**

Add to `tests/test_evaluation.py`:

```python
from evaluation.bailian_evaluator import calculate_contain, calculate_llm_accuracy, evaluate_predictions


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
```

- [ ] **Step 2: Run evaluator tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_bailian_evaluator_maps_correct_response_and_contain_accuracy tests/test_evaluation.py::test_evaluate_predictions_updates_prediction_file_and_summary`

Expected: FAIL because `evaluation.bailian_evaluator` does not exist.

- [ ] **Step 3: Implement evaluator**

Create `src/evaluation/bailian_evaluator.py`:

```python
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(text.lower())))


class BailianJudgeClient:
    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        api_key_env: str,
        temperature: float,
        max_tokens: int,
        timeout: int,
        retries: int,
        retry_sleep_seconds: float,
    ) -> None:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key. Set environment variable {api_key_env} before evaluation.")
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def infer(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        last_error: BaseException | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                return str(body["choices"][0]["message"]["content"])
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
                time.sleep(self.retry_sleep_seconds * attempt)
        raise RuntimeError(f"Bailian judge request failed after {self.retries} attempts: {last_error}")


def calculate_llm_accuracy(client: Any, pre_answer: str, gold_answer: str) -> float:
    system_prompt = "You are an expert evaluator."
    user_prompt = f"""Please evaluate if the generated answer is correct by comparing it with the gold answer.
Generated answer: {pre_answer}
Gold answer: {gold_answer}

The generated answer should be considered correct if it:
1. Contains the key information from the gold answer
2. Is factually accurate and consistent with the gold answer
3. Does not contain any contradicting information

Respond with ONLY 'correct' or 'incorrect'.
Response:
"""
    response = client.infer([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    return 1.0 if response.strip().lower() == "correct" else 0.0


def calculate_contain(pre_answer: str | None, gold_answer: str | None) -> int:
    if pre_answer is None or str(pre_answer).strip() == "":
        return 0
    if gold_answer is None or str(gold_answer).strip() == "":
        return 0
    return 1 if normalize_answer(str(gold_answer)) in normalize_answer(str(pre_answer)) else 0


def _evaluate_one(index: int, prediction: dict[str, Any], client: Any) -> tuple[int, float, int, str | None]:
    try:
        pre_answer = str(prediction.get("pred_answer") or "")
        gold_answer = str(prediction.get("gold_answer") or "")
        return index, calculate_llm_accuracy(client, pre_answer, gold_answer), calculate_contain(pre_answer, gold_answer), None
    except Exception as exc:
        pre_answer = str(prediction.get("pred_answer") or "")
        gold_answer = str(prediction.get("gold_answer") or "")
        return index, 0.0, calculate_contain(pre_answer, gold_answer), str(exc)


def evaluate_predictions(predictions_path: str | Path, *, client: Any, max_workers: int) -> dict[str, Any]:
    path = Path(predictions_path)
    predictions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise ValueError(f"Invalid predictions format at {path}: expected a JSON list.")
    if not predictions:
        summary = {"llm_accuracy": 0.0, "contain_accuracy": 0.0, "num_samples": 0}
        (path.parent / "evaluation_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return summary

    llm_scores = [0.0] * len(predictions)
    contain_scores = [0] * len(predictions)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
        futures = [executor.submit(_evaluate_one, index, prediction, client) for index, prediction in enumerate(predictions)]
        for future in as_completed(futures):
            index, llm_acc, contain_acc, error = future.result()
            llm_scores[index] = llm_acc
            contain_scores[index] = contain_acc
            predictions[index]["llm_accuracy"] = llm_acc
            predictions[index]["contain_accuracy"] = contain_acc
            if error is not None:
                predictions[index]["evaluation_error"] = error

    summary = {
        "llm_accuracy": sum(llm_scores) / len(llm_scores),
        "contain_accuracy": sum(contain_scores) / len(contain_scores),
        "num_samples": len(predictions),
    }
    path.write_text(json.dumps(predictions, ensure_ascii=False, indent=2), encoding="utf-8")
    (path.parent / "evaluation_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
```

- [ ] **Step 4: Run evaluator tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_bailian_evaluator_maps_correct_response_and_contain_accuracy tests/test_evaluation.py::test_evaluate_predictions_updates_prediction_file_and_summary`

Expected: PASS.

- [ ] **Step 5: Commit task 3**

Run:

```bash
git add src/evaluation/bailian_evaluator.py tests/test_evaluation.py
git commit -m "feat: add bailian rag evaluation metrics"
```

---

### Task 4: Prediction Formatting and RAG Inference Helpers

**Files:**
- Create: `src/evaluation/evaluate_rag_model.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `EvalSample`
- Produces: `format_prediction(sample: EvalSample, result: Any, error: str | None = None) -> dict[str, Any]`
- Produces: `run_predictions(args: argparse.Namespace, samples: list[EvalSample], policy: Any, retrieval_env: Any, output_dir: Path) -> list[dict[str, Any]]`

- [ ] **Step 1: Write failing prediction-format tests**

Add to `tests/test_evaluation.py`:

```python
from types import SimpleNamespace

from evaluation.data import EvalSample
from evaluation.evaluate_rag_model import format_prediction, run_predictions


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
```

- [ ] **Step 2: Run prediction tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_format_prediction_matches_linearrag_evaluator_schema tests/test_evaluation.py::test_run_predictions_flushes_jsonl_progress`

Expected: FAIL because `evaluation.evaluate_rag_model` does not exist.

- [ ] **Step 3: Implement prediction helpers**

Create `src/evaluation/evaluate_rag_model.py` with the imports and helper functions:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

from rag import RAGLoopExecutor
from rl_training.policy import HFSharedPolicy
from rl_training.retrieval import CachedLinearRAGRetrievalEnv
from sft_training.callbacks import _make_timestamped_output_dir

from .bailian_evaluator import BailianJudgeClient, evaluate_predictions
from .config import parse_args
from .data import EvalSample, load_eval_samples


try:
    from tqdm import tqdm
except Exception:
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def format_prediction(sample: EvalSample, result: Any, error: str | None = None) -> dict[str, Any]:
    prediction = {
        "qid": sample.qid,
        "dataset": sample.dataset,
        "question": sample.question,
        "pred_answer": "" if error else str(result.final_answer or ""),
        "gold_answer": sample.answer,
        "answer_aliases": sample.answer_aliases,
        "trajectory": [] if error else list(result.trajectory),
        "parse_errors": [] if error else list(result.parse_errors),
        "retrieval_count": 0 if error else int(getattr(result.state, "retrieval_count", 0)),
    }
    if error is not None:
        prediction["error"] = error
    return prediction


def run_predictions(
    args: Any,
    samples: list[EvalSample],
    policy: Any,
    retrieval_env: Any,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "predictions.jsonl"
    predictions: list[dict[str, Any]] = []
    iterator = tqdm(samples, desc="Evaluating RAG samples", unit="sample", disable=bool(args.disable_tqdm))
    for sample in iterator:
        try:
            if hasattr(policy, "reset_trace"):
                policy.reset_trace()
            executor = RAGLoopExecutor(policy=policy, retrieval_env=retrieval_env, max_rounds=args.max_rounds)
            result = executor.run(question=sample.question, dataset=sample.dataset)
            prediction = format_prediction(sample, result)
        except Exception as exc:
            prediction = format_prediction(sample, result=None, error=str(exc))
        predictions.append(prediction)
        _append_jsonl(progress_path, prediction)
    _write_json(output_dir / "predictions.json", predictions)
    return predictions
```

- [ ] **Step 4: Run prediction tests to verify they pass**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_format_prediction_matches_linearrag_evaluator_schema tests/test_evaluation.py::test_run_predictions_flushes_jsonl_progress`

Expected: PASS.

- [ ] **Step 5: Commit task 4**

Run:

```bash
git add src/evaluation/evaluate_rag_model.py tests/test_evaluation.py
git commit -m "feat: add rag evaluation prediction output"
```

---

### Task 5: Model Loading, Retrieval Construction, and Main Entrypoint

**Files:**
- Modify: `src/evaluation/evaluate_rag_model.py`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: parsed evaluation args
- Produces: `_configure_visible_gpus(args: Any) -> None`
- Produces: `_build_retrieval_env(args: Any) -> CachedLinearRAGRetrievalEnv`
- Produces: `_load_policy(args: Any) -> HFSharedPolicy`
- Produces: `main(argv: list[str] | None = None) -> int`

- [ ] **Step 1: Write failing lightweight tests for GPU and retrieval construction**

Add to `tests/test_evaluation.py`:

```python
from evaluation.evaluate_rag_model import _build_retrieval_env, _configure_visible_gpus


def test_evaluate_configure_visible_gpus_respects_existing_env(monkeypatch: pytest.MonkeyPatch) -> None:
    args = SimpleNamespace(gpu_indices="1", gpu_index=0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")

    _configure_visible_gpus(args)

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


def test_evaluate_build_retrieval_env_uses_eval_retrieval_config(monkeypatch: pytest.MonkeyPatch) -> None:
    captured = {}

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
```

Also add `import os` near the top of the test file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_evaluate_configure_visible_gpus_respects_existing_env tests/test_evaluation.py::test_evaluate_build_retrieval_env_uses_eval_retrieval_config`

Expected: FAIL because helpers are not implemented.

- [ ] **Step 3: Implement runtime helpers and main**

Append to `src/evaluation/evaluate_rag_model.py`:

```python
def _configure_visible_gpus(args: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    gpu_indices = str(getattr(args, "gpu_indices", "") or "").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices or str(args.gpu_index)


def _load_dependencies() -> dict[str, Any]:
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Missing dependency: {exc.name}. Install transformers, peft, torch and optional bitsandbytes "
            "in the MACORAG runtime environment."
        ) from exc
    return {
        "torch": torch,
        "AutoModelForCausalLM": AutoModelForCausalLM,
        "AutoTokenizer": AutoTokenizer,
        "PeftModel": PeftModel,
    }


def _torch_dtype(args: Any, torch: Any) -> Any:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float16


def _model_kwargs(args: Any, torch: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"torch_dtype": _torch_dtype(args, torch)}
    if not args.load_4bit:
        return kwargs
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # type: ignore  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(f"4-bit quantization requested but dependency missing: {exc.name}.") from exc
    kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=_torch_dtype(args, torch),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    kwargs["device_map"] = "auto"
    return kwargs


def _device(torch: Any) -> Any:
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_policy(args: Any) -> HFSharedPolicy:
    deps = _load_dependencies()
    torch = deps["torch"]
    AutoModelForCausalLM = deps["AutoModelForCausalLM"]
    AutoTokenizer = deps["AutoTokenizer"]
    PeftModel = deps["PeftModel"]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(args.model_path, **_model_kwargs(args, torch))
    model = PeftModel.from_pretrained(base_model, args.adapter_path, is_trainable=False)
    model.eval()
    if not args.load_4bit:
        model.to(_device(torch))
    return HFSharedPolicy(
        model=model,
        tokenizer=tokenizer,
        system_prompt=args.system_prompt,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )


def _build_retrieval_env(args: Any) -> CachedLinearRAGRetrievalEnv:
    return CachedLinearRAGRetrievalEnv(
        retrieval_root=args.retrieval_root,
        embedding_model=args.retrieval_embedding_model,
        spacy_model=args.retrieval_spacy_model,
        top_k=args.retrieval_top_k,
        max_workers=args.retrieval_max_workers,
        batch_size=args.retrieval_batch_size,
        use_vectorized_retrieval=args.use_vectorized_retrieval,
    )


def _resolved_output_dir(args: Any) -> Path:
    output_dir = Path(args.output_dir)
    if args.fixed_output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    return _make_timestamped_output_dir(str(output_dir))


def _args_to_jsonable(args: Any) -> dict[str, Any]:
    return {key: value for key, value in vars(args).items() if isinstance(value, (str, int, float, bool, list, tuple, type(None)))}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_visible_gpus(args)
    random.seed(args.seed)
    output_dir = _resolved_output_dir(args)
    _write_json(output_dir / "run_config.json", _args_to_jsonable(args))

    samples, sample_summary = load_eval_samples(
        data_root=args.data_root,
        data_files=list(args.data_files or []),
        max_samples=args.max_samples,
    )
    _write_json(output_dir / "data_summary.json", sample_summary)
    policy = _load_policy(args)
    retrieval_env = _build_retrieval_env(args)
    run_predictions(args, samples, policy, retrieval_env, output_dir)

    if not args.skip_judge:
        client = BailianJudgeClient(
            model=args.judge_model,
            endpoint=args.judge_endpoint,
            api_key_env=args.judge_api_key_env,
            temperature=args.judge_temperature,
            max_tokens=args.judge_max_tokens,
            timeout=args.judge_timeout,
            retries=args.judge_retries,
            retry_sleep_seconds=args.judge_retry_sleep_seconds,
        )
        summary = evaluate_predictions(output_dir / "predictions.json", client=client, max_workers=args.judge_workers)
        summary["judge_model"] = args.judge_model
        _write_json(output_dir / "evaluation_results.json", summary)

    print(f"Evaluation artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run lightweight runtime tests**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_evaluate_configure_visible_gpus_respects_existing_env tests/test_evaluation.py::test_evaluate_build_retrieval_env_uses_eval_retrieval_config`

Expected: PASS.

- [ ] **Step 5: Commit task 5**

Run:

```bash
git add src/evaluation/evaluate_rag_model.py tests/test_evaluation.py
git commit -m "feat: add rag evaluation runtime entrypoint"
```

---

### Task 6: YAML Config and Shell Launcher

**Files:**
- Create: `config/evaluate_rag_model.yml`
- Create: `scripts/evaluate_rag_model.sh`
- Modify: `tests/test_evaluation.py`

**Interfaces:**
- Consumes: `config/evaluate_rag_model.yml`
- Produces: `scripts/evaluate_rag_model.sh`

- [ ] **Step 1: Write failing launcher test**

Add to `tests/test_evaluation.py`:

```python
def test_evaluate_shell_script_derives_gpu_visibility_from_yaml() -> None:
    script = Path("scripts/evaluate_rag_model.sh").read_text(encoding="utf-8")

    assert "CONFIG_PATH=" in script
    assert "yaml.safe_load" in script
    assert 'export CUDA_VISIBLE_DEVICES="${YAML_GPU_INDICES}"' in script
    assert 'export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"' in script
    assert "-m evaluation.evaluate_rag_model --config" in script
```

- [ ] **Step 2: Run launcher test to verify it fails**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_evaluate_shell_script_derives_gpu_visibility_from_yaml`

Expected: FAIL because `scripts/evaluate_rag_model.sh` does not exist.

- [ ] **Step 3: Add YAML config**

Create `config/evaluate_rag_model.yml`:

```yaml
model_path: "model/Qwen2.5-3B-Instruct"
adapter_path: "outputs/grpo_qwen2.5-3b/adapter"
data_root: "data/eval_1000"
data_files: []
retrieval_root: "data/eval_1000_retrieval"
output_dir: "outputs/eval_rag_model"
fixed_output_dir: false
system_prompt: "Follow the role-specific prompt. Output exactly the requested XML-style tag with valid JSON."
max_samples: null
seed: 42

max_rounds: 3
max_prompt_length: 4096
max_completion_length: 256
temperature: 0.0
top_p: 0.95
top_k: 5

retrieval_embedding_model: "sentence-transformers/all-mpnet-base-v2"
retrieval_spacy_model: en_core_web_trf
retrieval_top_k: 5
retrieval_max_workers: 4
retrieval_batch_size: 32
use_vectorized_retrieval: true

skip_judge: false
judge_model: "qwen-plus"
judge_endpoint: "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
judge_api_key_env: "DASHSCOPE_API_KEY"
judge_temperature: 0.0
judge_max_tokens: 8
judge_timeout: 120
judge_retries: 3
judge_retry_sleep_seconds: 2.0
judge_workers: 4

bf16: false
fp16: false
load_4bit: true
gpu_index: 0
gpu_indices: "1"
disable_tqdm: false
```

- [ ] **Step 4: Add shell launcher**

Create `scripts/evaluate_rag_model.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="false"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export MACORAG_SILENT_RETRIEVAL="${MACORAG_SILENT_RETRIEVAL:-1}"

cd "${REPO_ROOT}"

CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/evaluate_rag_model.yml}"

YAML_GPU_INDICES="$(
  "${PYTHON:-python}" - "${CONFIG_PATH}" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
gpu_indices = str(config.get("gpu_indices") or config.get("gpu_index") or "0").strip()
print(gpu_indices)
PY
)"

export CUDA_VISIBLE_DEVICES="${YAML_GPU_INDICES}"

"${PYTHON:-python}" -m evaluation.evaluate_rag_model --config "${CONFIG_PATH}" "$@"
```

- [ ] **Step 5: Run launcher test to verify it passes**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py::test_evaluate_shell_script_derives_gpu_visibility_from_yaml`

Expected: PASS.

- [ ] **Step 6: Commit task 6**

Run:

```bash
git add config/evaluate_rag_model.yml scripts/evaluate_rag_model.sh tests/test_evaluation.py
git commit -m "feat: add rag evaluation launcher"
```

---

### Task 7: Integration Verification

**Files:**
- Modify only if verification exposes defects:
  - `src/evaluation/*.py`
  - `config/evaluate_rag_model.yml`
  - `scripts/evaluate_rag_model.sh`
  - `tests/test_evaluation.py`

**Interfaces:**
- Consumes all interfaces from prior tasks.
- Produces a tested runnable evaluation module.

- [ ] **Step 1: Run focused evaluation tests**

Run: `PYTHONPATH=src pytest -q tests/test_evaluation.py`

Expected: PASS.

- [ ] **Step 2: Run related regression tests**

Run: `PYTHONPATH=src pytest -q tests/test_rl_training.py tests/test_rag.py tests/test_retrieval_env.py`

Expected: PASS. If unrelated pre-existing failures appear, record them with exact test names and do not change unrelated code.

- [ ] **Step 3: Run config parse smoke test**

Run: `PYTHONPATH=src python -m evaluation.evaluate_rag_model --config config/evaluate_rag_model.yml --max-samples 1 --skip-judge --fixed-output-dir --output-dir /tmp/macorag_eval_smoke --disable-tqdm`

Expected: If the runtime environment has model and retrieval dependencies available, this writes `/tmp/macorag_eval_smoke/predictions.json`. If dependencies are missing, the command fails with a direct missing-dependency message; report that exact blocker.

- [ ] **Step 4: Inspect output schema if smoke test produced artifacts**

Run: `python -m json.tool /tmp/macorag_eval_smoke/predictions.json >/tmp/macorag_eval_predictions_check.json`

Expected: exit code 0 and the first prediction contains `pred_answer` and `gold_answer`.

- [ ] **Step 5: Commit any verification fixes**

Run only if files changed during verification:

```bash
git add src/evaluation config/evaluate_rag_model.yml scripts/evaluate_rag_model.sh tests/test_evaluation.py
git commit -m "fix: harden rag evaluation verification"
```

---

## Self-Review

- Spec coverage: The plan covers the independent module, YAML config, shell launcher, `data/eval_1000`, `data/eval_1000_retrieval`, Bailian `qwen-plus` judge, LinearRAG-compatible prediction schema, and incremental prediction durability.
- Placeholder scan: No task contains placeholder-only implementation steps.
- Type consistency: `EvalSample`, `parse_args`, `format_prediction`, `run_predictions`, `BailianJudgeClient`, and `evaluate_predictions` signatures are defined before downstream usage.
- Scope check: Retrieval rebuild, dataset extraction, SFT training, RL training, and RAG behavior changes remain out of scope.
