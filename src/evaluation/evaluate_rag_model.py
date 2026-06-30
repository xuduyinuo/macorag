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
