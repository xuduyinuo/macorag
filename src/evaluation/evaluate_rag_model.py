#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rag import (
    AgentRole,
    RAGLoopExecutor,
    RAGState,
    build_answer_generator_prompt,
    build_evidence_updater_prompt,
    build_query_retriever_prompt,
)
from rl_training.policy import HFSharedPolicy
from rl_training.retrieval import CachedLinearRAGRetrievalEnv

from .bailian_evaluator import BailianJudgeClient, evaluate_predictions
from .config import parse_args
from .data import EvalSample, load_eval_samples
from .output import make_run_dir


try:
    from tqdm import tqdm
except Exception:

    def tqdm(iterable, *args, **kwargs):
        return iterable


DEFAULT_SYSTEM_PROMPT = "Follow the role-specific prompt. Output exactly the requested XML-style tag with valid JSON."
DEFAULT_SEED = 42


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class VLLMOpenAIPolicy:
    def __init__(
        self,
        *,
        base_urls: list[str] | tuple[str, ...],
        model: str,
        api_key_env: str,
        system_prompt: str,
        max_completion_length: int,
        temperature: float,
        top_p: float,
        timeout: int,
        retries: int,
        retry_sleep_seconds: float,
    ) -> None:
        self.base_urls = [str(url).rstrip("/") for url in base_urls if str(url).strip()]
        if not self.base_urls:
            raise SystemExit("vllm_base_urls must contain at least one OpenAI-compatible endpoint.")
        self.model = str(model or "").strip()
        if not self.model:
            raise SystemExit("vllm_model must be set when inference_backend is vllm_openai.")
        self.api_key_env = str(api_key_env or "").strip()
        self.system_prompt = system_prompt
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.retry_sleep_seconds = retry_sleep_seconds
        self._thread_local = threading.local()

    def reset_trace(self) -> None:
        return None

    def set_endpoint_index(self, index: int) -> None:
        self._thread_local.endpoint_index = int(index)

    def _endpoint(self) -> str:
        index = int(getattr(self._thread_local, "endpoint_index", 0))
        return self.base_urls[index % len(self.base_urls)] + "/chat/completions"

    def _prompt_for(
        self,
        *,
        role: AgentRole,
        question: str,
        state: RAGState,
        observation: dict[str, Any] | None,
    ) -> str:
        if role == AgentRole.QUERY_RETRIEVER:
            return build_query_retriever_prompt(question=question, state=state)
        if role == AgentRole.EVIDENCE_UPDATER:
            return build_evidence_updater_prompt(
                question=question,
                state=state,
                observation=observation or {"passages": []},
            )
        return build_answer_generator_prompt(question=question, state=state)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key_env:
            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"vLLM API key environment variable is not set: {self.api_key_env}")
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _post_chat_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self._endpoint(),
            data=data,
            headers=self._headers(),
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(self.retry_sleep_seconds)
        raise RuntimeError(f"vLLM chat completion failed after {self.retries} attempt(s): {last_error}") from last_error

    def generate(
        self,
        *,
        role: AgentRole,
        question: str,
        state: RAGState,
        observation: dict[str, Any] | None = None,
    ) -> str:
        role = AgentRole(role)
        prompt = self._prompt_for(role=role, question=question, state=state, observation=observation)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_completion_length,
        }
        response = self._post_chat_completion(payload)
        try:
            return str(response["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Invalid vLLM chat completion response: {response}") from exc


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
    infra_terms = (
        "spacy",
        "sentence-transformers",
        "sentence_transformers",
        "linearrag",
        "retrieval",
        "embedding",
        "graphml",
        "igraph",
        "module",
        "dependency",
        "index",
        "vllm",
        "openai",
        "chat completion",
    )
    fatal_terms = (
        "missing",
        "cannot import",
        "install",
        "not found",
        "no such file",
        "linearrag index files missing",
        "failed",
        "connection refused",
        "timed out",
        "timeout",
    )
    return any(term in msg for term in fatal_terms) and any(term in msg for term in infra_terms)


def _run_one_prediction(
    *,
    index: int,
    sample: EvalSample,
    args: Any,
    policy: Any,
    retrieval_env: Any,
) -> tuple[int, dict[str, Any]]:
    try:
        if hasattr(policy, "set_endpoint_index"):
            policy.set_endpoint_index(index)
        if hasattr(policy, "reset_trace"):
            policy.reset_trace()
        executor = RAGLoopExecutor(policy=policy, retrieval_env=retrieval_env, max_rounds=args.max_rounds)
        result = executor.run(question=sample.question, dataset=sample.dataset)
        prediction = format_prediction(sample, result)
    except Exception as exc:
        if _is_infrastructure_error(exc):
            raise
        prediction = format_prediction(sample, result=None, error=str(exc))
    return index, prediction


def run_predictions(
    args: Any,
    samples: list[EvalSample],
    policy: Any,
    retrieval_env: Any,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "predictions.jsonl"
    if progress_path.exists():
        progress_path.unlink()
    predictions_by_index: dict[int, dict[str, Any]] = {}
    backend = str(getattr(args, "inference_backend", "hf_local") or "hf_local")
    eval_workers = max(1, int(getattr(args, "eval_request_workers", 1) or 1))
    use_threads = backend == "vllm_openai" and eval_workers > 1
    if use_threads:
        progress_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=eval_workers) as executor:
            futures = [
                executor.submit(
                    _run_one_prediction,
                    index=index,
                    sample=sample,
                    args=args,
                    policy=policy,
                    retrieval_env=retrieval_env,
                )
                for index, sample in enumerate(samples)
            ]
            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Evaluating RAG samples",
                unit="sample",
                disable=bool(getattr(args, "disable_tqdm", False)),
            )
            for future in iterator:
                index, prediction = future.result()
                predictions_by_index[index] = prediction
                with progress_lock:
                    _append_jsonl(progress_path, prediction)
    else:
        iterator = tqdm(
            samples,
            desc="Evaluating RAG samples",
            unit="sample",
            disable=bool(getattr(args, "disable_tqdm", False)),
        )
        for index, sample in enumerate(iterator):
            _, prediction = _run_one_prediction(
                index=index,
                sample=sample,
                args=args,
                policy=policy,
                retrieval_env=retrieval_env,
            )
            predictions_by_index[index] = prediction
            _append_jsonl(progress_path, prediction)
    predictions = [predictions_by_index[index] for index in range(len(samples))]
    _write_json(output_dir / "predictions.json", predictions)
    return predictions


def _configure_visible_gpus(args: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    gpu_indices = str(getattr(args, "gpu_indices", "") or "").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices or "0"


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
    if not args.load_4bit or not torch.cuda.is_available():
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


def _load_policy(args: Any) -> HFSharedPolicy | VLLMOpenAIPolicy:
    backend = str(getattr(args, "inference_backend", "hf_local") or "hf_local")
    if backend == "vllm_openai":
        return VLLMOpenAIPolicy(
            base_urls=list(getattr(args, "vllm_base_urls", []) or []),
            model=getattr(args, "vllm_model", ""),
            api_key_env=getattr(args, "vllm_api_key_env", ""),
            system_prompt=getattr(args, "system_prompt", DEFAULT_SYSTEM_PROMPT),
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            timeout=args.vllm_timeout,
            retries=args.vllm_retries,
            retry_sleep_seconds=args.vllm_retry_sleep_seconds,
        )
    if backend != "hf_local":
        raise SystemExit("Unsupported inference_backend. Expected one of: hf_local, vllm_openai.")
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
        system_prompt=getattr(args, "system_prompt", DEFAULT_SYSTEM_PROMPT),
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


def validate_retrieval_assets(retrieval_root: str | Path, datasets: list[str] | set[str] | tuple[str, ...]) -> None:
    required_files = (
        "passage_embedding.parquet",
        "entity_embedding.parquet",
        "sentence_embedding.parquet",
        "LinearRAG.graphml",
    )
    root = Path(retrieval_root)
    missing_paths: list[Path] = []
    for dataset in sorted({str(dataset).strip() for dataset in datasets if str(dataset).strip()}):
        dataset_root = root / dataset
        for file_name in required_files:
            path = dataset_root / file_name
            if not path.exists():
                missing_paths.append(path)
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"LinearRAG index files missing: {missing}")


def _resolved_output_dir(args: Any) -> Path:
    return make_run_dir(args.output_root)


def _args_to_jsonable(args: Any) -> dict[str, Any]:
    return {
        key: value
        for key, value in vars(args).items()
        if isinstance(value, (str, int, float, bool, list, tuple, type(None)))
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _configure_visible_gpus(args)
    # 固定随机种子保持历史评估顺序和采样行为稳定，配置文件无需暴露该低频参数。
    import random

    random.seed(DEFAULT_SEED)
    output_dir = _resolved_output_dir(args)
    _write_json(output_dir / "run_config.json", _args_to_jsonable(args))

    samples, sample_summary = load_eval_samples(
        data_root=args.data_root,
        data_files=list(args.data_files or []),
        max_samples=args.max_samples,
    )
    _write_json(output_dir / "data_summary.json", sample_summary)
    validate_retrieval_assets(args.retrieval_root, [sample.dataset for sample in samples])
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
