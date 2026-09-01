#!/usr/bin/env python3
from __future__ import annotations

import json
import hashlib
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rag import (
    AgentRole,
    AnswerPromptContext,
    RAGLoopExecutor,
    RAGState,
    build_answer_generator_prompt,
    build_evidence_updater_prompt,
    build_query_retriever_prompt,
)
from prompt_config import load_prompt_contract, system_prompt_for
from rag.protocol_metrics import compute_protocol_metrics
from rl_training.retrieval import create_retrieval_env
from rl_training.retrieval import validate_retrieval_assets as validate_runtime_retrieval_assets

from .config import parse_args
from .data import EvalSample, load_eval_samples
from .local_evaluator import evaluate_predictions
from .output import make_run_dir


try:
    from tqdm import tqdm
except Exception:

    def tqdm(iterable, *args, **kwargs):
        return iterable


DEFAULT_SEED = 42


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_fixed_manifest(
    *,
    args: Any,
    samples: list[EvalSample],
    sample_summary: dict[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    meta_path_value = str(getattr(args, "manifest_meta_path", "") or "").strip()
    if not meta_path_value:
        return None, {}
    meta_path = Path(meta_path_value)
    meta = _read_json(meta_path)
    source_files = [Path(value) for value in sample_summary.get("source_files", [])]
    if len(source_files) != 1:
        raise ValueError("Fixed evaluation requires exactly one manifest JSONL source file.")
    manifest_path = source_files[0]
    actual_fingerprint = _sha256_path(manifest_path)
    if actual_fingerprint != meta.get("manifest_fingerprint"):
        raise ValueError("Fixed evaluation manifest fingerprint does not match manifest_meta.json.")
    actual_counts = dict(Counter(sample.dataset for sample in samples))
    expected_counts = {str(key): int(value) for key, value in (meta.get("counts_by_dataset") or {}).items()}
    if actual_counts != expected_counts:
        raise ValueError(f"Fixed evaluation dataset counts mismatch: {actual_counts} != {expected_counts}")
    actual_qids = [sample.qid for sample in samples]
    if actual_qids != [str(value) for value in meta.get("qids", [])]:
        raise ValueError("Fixed evaluation qid order does not match manifest metadata.")
    expected_total = int(meta.get("per_dataset", 0)) * len(expected_counts)
    if len(samples) != expected_total or int(sample_summary.get("skipped_samples", 0)) != 0:
        raise ValueError("Fixed evaluation manifest contains missing or invalid samples.")
    return actual_fingerprint, meta


def _adapter_identity(args: Any) -> dict[str, Any]:
    value = str(getattr(args, "adapter_identity_path", "") or "").strip()
    if not value:
        return {}
    root = Path(value)
    required = [root / "adapter_config.json", root / "prompt_contract.json"]
    weights = [path for path in (root / "adapter_model.safetensors", root / "adapter_model.bin") if path.is_file()]
    missing = [path for path in required if not path.is_file()]
    if missing or len(weights) != 1:
        raise ValueError(f"Invalid adapter identity path: {root}")
    hashes = {path.name: _sha256_path(path) for path in [*required, weights[0]]}
    serialized = json.dumps(hashes, sort_keys=True, separators=(",", ":"))
    return {
        "path": str(root),
        "files": hashes,
        "fingerprint": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
    }


def _retrieval_metadata_fingerprints(args: Any, datasets: list[str]) -> dict[str, str]:
    root = Path(args.retrieval_root)
    fingerprints = {}
    for dataset in sorted(set(datasets)):
        path = root / dataset / "index_metadata.json"
        if path.is_file():
            fingerprints[dataset] = _sha256_path(path)
    return fingerprints


def _build_evaluation_contract(
    *,
    args: Any,
    prompt_contract: Any,
    manifest_fingerprint: str | None,
    datasets: list[str],
) -> dict[str, Any]:
    payload = {
        "manifest_fingerprint": manifest_fingerprint,
        "prompt_contract_fingerprint": prompt_contract.fingerprint,
        "retrieval": {
            "backend": getattr(args, "retrieval_backend", "linear_rag"),
            "root": str(args.retrieval_root),
            "embedding_model": args.retrieval_embedding_model,
            "max_length": getattr(args, "retrieval_max_length", None),
            "top_k": args.retrieval_top_k,
            "index_metadata_fingerprints": _retrieval_metadata_fingerprints(args, datasets),
        },
        "generation": {
            "model": getattr(args, "vllm_model", ""),
            "max_rounds": args.max_rounds,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["contract_fingerprint"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return payload


def _prepare_resume_identity(
    *,
    output_dir: Path,
    args: Any,
    contract: dict[str, Any],
    adapter_identity: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "contract_fingerprint": contract["contract_fingerprint"],
        "adapter_label": str(getattr(args, "adapter_label", "") or ""),
        "adapter_fingerprint": adapter_identity.get("fingerprint"),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["resume_fingerprint"] = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    path = output_dir / "resume_identity.json"
    progress_exists = any(output_dir.glob("*/predictions.jsonl"))
    if bool(getattr(args, "resume", False)) and progress_exists:
        if not path.is_file() or _read_json(path) != payload:
            raise ValueError("Existing fixed evaluation predictions do not match the requested resume identity.")
    temporary = path.with_suffix(".json.tmp")
    _write_json(temporary, payload)
    temporary.replace(path)
    return payload


def _dataset_name_for_path(value: Any) -> str:
    dataset = str(value or "unknown").strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in dataset)


def _group_samples_by_dataset(samples: list[EvalSample]) -> list[tuple[str, list[EvalSample]]]:
    grouped: dict[str, list[EvalSample]] = {}
    order: list[str] = []
    for sample in samples:
        if sample.dataset not in grouped:
            grouped[sample.dataset] = []
            order.append(sample.dataset)
        grouped[sample.dataset].append(sample)
    return [(dataset, grouped[dataset]) for dataset in order]


def _dataset_output_dir(output_dir: Path, dataset: str) -> Path:
    return output_dir / _dataset_name_for_path(dataset)


class VLLMOpenAIPolicy:
    def __init__(
        self,
        *,
        base_urls: list[str] | tuple[str, ...],
        model: str,
        api_key_env: str,
        system_prompt: str | None,
        max_prompt_length: int | None,
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
            raise SystemExit("vllm_model must be set for evaluation.")
        self.api_key_env = str(api_key_env or "").strip()
        self.system_prompt = system_prompt
        self.max_prompt_length = max_prompt_length
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.retries = max(1, int(retries))
        self.retry_sleep_seconds = retry_sleep_seconds
        self._thread_local = threading.local()
        self._loopback_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def reset_trace(self) -> None:
        return None

    def set_endpoint_index(self, index: int) -> None:
        self._thread_local.endpoint_index = int(index)

    def _endpoint(self) -> str:
        index = int(getattr(self._thread_local, "endpoint_index", 0))
        return self.base_urls[index % len(self.base_urls)] + "/chat/completions"

    def _open(self, request: urllib.request.Request):
        hostname = (urllib.parse.urlparse(request.full_url).hostname or "").lower()
        if hostname in {"127.0.0.1", "localhost", "::1"}:
            return self._loopback_opener.open(request, timeout=self.timeout)
        return urllib.request.urlopen(request, timeout=self.timeout)

    def _prompt_for(
        self,
        *,
        role: AgentRole,
        question: str,
        state: RAGState,
        observation: dict[str, Any] | None,
        answer_context: AnswerPromptContext | None = None,
        force_final_answer: bool | None = None,
    ) -> str:
        if role == AgentRole.QUERY_RETRIEVER:
            return build_query_retriever_prompt(question=question, state=state)
        if role == AgentRole.EVIDENCE_UPDATER:
            return build_evidence_updater_prompt(
                question=question,
                state=state,
                observation=observation or {"passages": []},
            )
        return build_answer_generator_prompt(
            question=question,
            state=state,
            context=answer_context,
            force_final_answer=force_final_answer if answer_context is None else None,
        )

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
                with self._open(request) as response:
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
        answer_context: AnswerPromptContext | None = None,
        force_final_answer: bool | None = None,
    ) -> str:
        role = AgentRole(role)
        prompt = self._prompt_for(
            role=role,
            question=question,
            state=state,
            observation=observation,
            answer_context=answer_context,
            force_final_answer=force_final_answer,
        )
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt if self.system_prompt is not None else system_prompt_for(role),
                },
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
        "parse_errors": [f"evaluation_error: {error}"] if error else list(result.parse_errors),
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
    resume = bool(getattr(args, "resume", False))
    if progress_path.exists() and not resume:
        progress_path.unlink()
    predictions_json_path = output_dir / "predictions.json"
    if predictions_json_path.exists():
        predictions_json_path.unlink()
    predictions_by_index: dict[int, dict[str, Any]] = {}
    sample_index_by_qid = {sample.qid: index for index, sample in enumerate(samples)}
    if resume and progress_path.exists():
        seen: set[str] = set()
        with progress_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                prediction = json.loads(line)
                qid = str(prediction.get("qid") or "")
                if qid in seen:
                    raise ValueError(f"Duplicate resumed prediction qid: {qid}")
                if qid not in sample_index_by_qid:
                    raise ValueError(f"Unknown resumed prediction qid: {qid}")
                sample = samples[sample_index_by_qid[qid]]
                if (
                    str(prediction.get("dataset")) != sample.dataset
                    or str(prediction.get("question")) != sample.question
                    or str(prediction.get("gold_answer")) != sample.answer
                ):
                    raise ValueError(f"Resumed prediction contract mismatch for qid: {qid}")
                seen.add(qid)
                predictions_by_index[sample_index_by_qid[qid]] = prediction
    pending = [(index, sample) for index, sample in enumerate(samples) if index not in predictions_by_index]
    eval_workers = max(1, int(getattr(args, "eval_request_workers", 1) or 1))
    use_threads = eval_workers > 1
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
                for index, sample in pending
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
            pending,
            total=len(pending),
            desc="Evaluating RAG samples",
            unit="sample",
            disable=bool(getattr(args, "disable_tqdm", False)),
        )
        for index, sample in iterator:
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
    temporary_path = progress_path.with_suffix(".jsonl.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    temporary_path.replace(progress_path)
    return predictions


def _configure_visible_gpus(args: Any) -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    gpu_indices = str(getattr(args, "gpu_indices", "") or "").strip()
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_indices or "0"


def _load_policy(args: Any) -> VLLMOpenAIPolicy:
    return VLLMOpenAIPolicy(
        base_urls=list(getattr(args, "vllm_base_urls", []) or []),
        model=getattr(args, "vllm_model", ""),
        api_key_env=getattr(args, "vllm_api_key_env", ""),
        system_prompt=getattr(args, "system_prompt", None),
        max_prompt_length=getattr(args, "max_prompt_length", 4096),
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        timeout=args.vllm_timeout,
        retries=args.vllm_retries,
        retry_sleep_seconds=args.vllm_retry_sleep_seconds,
    )


def _build_retrieval_env(args: Any) -> Any:
    return create_retrieval_env(
        backend=getattr(args, "retrieval_backend", "linear_rag"),
        retrieval_root=args.retrieval_root,
        embedding_model=args.retrieval_embedding_model,
        device=getattr(args, "retrieval_device", "cpu"),
        max_length=getattr(args, "retrieval_max_length", 512),
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
    explicit = str(getattr(args, "output_dir", "") or "").strip()
    if explicit:
        path = Path(explicit)
        path.mkdir(parents=True, exist_ok=True)
        return path
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
    prompt_contract = load_prompt_contract(getattr(args, "prompt_config_path", None))
    samples, sample_summary = load_eval_samples(
        data_root=args.data_root,
        data_files=list(args.data_files or []),
        max_samples=args.max_samples,
    )
    manifest_fingerprint, manifest_meta = _validate_fixed_manifest(
        args=args,
        samples=samples,
        sample_summary=sample_summary,
    )
    adapter_identity = _adapter_identity(args)
    contract_payload = _build_evaluation_contract(
        args=args,
        prompt_contract=prompt_contract,
        manifest_fingerprint=manifest_fingerprint,
        datasets=[sample.dataset for sample in samples],
    )
    resume_identity = _prepare_resume_identity(
        output_dir=output_dir,
        args=args,
        contract=contract_payload,
        adapter_identity=adapter_identity,
    )
    run_config = _args_to_jsonable(args)
    run_config.update(
        {
            "prompt_contract_version": prompt_contract.version,
            "prompt_contract_fingerprint": prompt_contract.fingerprint,
            "prompt_config_path": str(prompt_contract.source_path),
            "manifest_fingerprint": manifest_fingerprint,
            "adapter_identity": adapter_identity,
            "resume_identity": resume_identity,
        }
    )
    _write_json(output_dir / "run_config.json", run_config)
    _write_json(output_dir / "data_summary.json", sample_summary)
    validate_runtime_retrieval_assets(
        backend=getattr(args, "retrieval_backend", "linear_rag"),
        retrieval_root=args.retrieval_root,
        datasets=[sample.dataset for sample in samples],
        embedding_model=args.retrieval_embedding_model,
    )
    policy = _load_policy(args)
    retrieval_env = _build_retrieval_env(args)

    dataset_metrics: dict[str, dict[str, Any]] = {}
    all_predictions: list[dict[str, Any]] = []
    for dataset, dataset_samples in _group_samples_by_dataset(samples):
        dataset_dir = _dataset_output_dir(output_dir, dataset)
        predictions = run_predictions(args, dataset_samples, policy, retrieval_env, dataset_dir)
        protocol_metrics = compute_protocol_metrics(
            [
                {"trajectory": item.get("trajectory", []), "parse_errors": item.get("parse_errors", [])}
                for item in predictions
            ]
        )
        protocol_metrics["error_count"] = sum(bool(item.get("error")) for item in predictions)
        protocol_metrics["error_rate"] = (
            protocol_metrics["error_count"] / len(predictions) if predictions else 0.0
        )
        _write_json(dataset_dir / "protocol_metrics.json", protocol_metrics)
        dataset_metrics[dataset] = evaluate_predictions(dataset_dir / "predictions.jsonl")
        all_predictions.extend(predictions)

    aggregate_metrics = {
        "macro_f1": sum(item["f1"] for item in dataset_metrics.values()) / len(dataset_metrics),
        "datasets": dataset_metrics,
        "num_samples": len(all_predictions),
    }
    aggregate_protocol = compute_protocol_metrics(
        [
            {"trajectory": item.get("trajectory", []), "parse_errors": item.get("parse_errors", [])}
            for item in all_predictions
        ]
    )
    aggregate_protocol["error_count"] = sum(bool(item.get("error")) for item in all_predictions)
    aggregate_protocol["error_rate"] = (
        aggregate_protocol["error_count"] / len(all_predictions) if all_predictions else 0.0
    )
    contract_payload["adapter_label"] = str(getattr(args, "adapter_label", "") or "")
    contract_payload["adapter_identity"] = adapter_identity
    contract_payload["manifest_meta"] = manifest_meta
    _write_json(output_dir / "aggregate_metrics.json", aggregate_metrics)
    _write_json(output_dir / "aggregate_protocol_metrics.json", aggregate_protocol)
    _write_json(output_dir / "evaluation_contract.json", contract_payload)

    print(f"Evaluation artifacts written to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
