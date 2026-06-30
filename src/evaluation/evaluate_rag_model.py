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


def _is_infrastructure_error(exc: Exception) -> bool:
    if isinstance(exc, FileNotFoundError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    msg = str(exc).lower()
    return ("index" in msg and ("not found" in msg or "missing" in msg or "no such file" in msg))


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
            if _is_infrastructure_error(exc):
                raise
            prediction = format_prediction(sample, result=None, error=str(exc))
        predictions.append(prediction)
        _append_jsonl(progress_path, prediction)
    _write_json(output_dir / "predictions.json", predictions)
    return predictions


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
    if not torch.cuda.is_available():
        return getattr(torch, "float32", torch.float16)
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
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
    }


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
        judge_metadata = {
            "judge_model": args.judge_model,
            "judge_endpoint": args.judge_endpoint,
            "judge_api_key_env": args.judge_api_key_env,
            "judge_temperature": args.judge_temperature,
            "judge_max_tokens": args.judge_max_tokens,
            "judge_timeout": args.judge_timeout,
            "judge_retries": args.judge_retries,
            "judge_retry_sleep_seconds": args.judge_retry_sleep_seconds,
            "judge_workers": args.judge_workers,
        }
        evaluate_predictions(
            output_dir / "predictions.json",
            client=client,
            max_workers=args.judge_workers,
            judge_metadata=judge_metadata,
        )

    print(f"Evaluation artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
