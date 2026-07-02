#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections import Counter
import copy
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from data_processing.io_utils import read_jsonl, write_json, write_jsonl
from data_processing.retrieval import create_linear_rag_query_engine, query_linear_rag


DATASETS = ("hotpotqa", "2wiki", "musique")
DEFAULT_SOURCE_ROOT = "data/trajectory_test"
DEFAULT_RETRIEVAL_ROOT = "data/trajectory_test_retrieval"
DEFAULT_OUTPUT_DIR = "data/sft/teacher_qwen_plus_trajectory_test"
DEFAULT_MODEL = "qwen-plus"
DEFAULT_ENDPOINT = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
_RETRIEVAL_THREAD_LOCAL = threading.local()

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current.parents[2]


REPO_ROOT = _resolve_repo_root()
DEFAULT_CONFIG = REPO_ROOT / "config" / "generate_teacher_sft.yml"


@dataclass
class SFTConfig:
    datasets: list[str] = field(default_factory=lambda: list(DATASETS))
    source_root: Union[str, Path] = DEFAULT_SOURCE_ROOT
    retrieval_root: Union[str, Path] = DEFAULT_RETRIEVAL_ROOT
    output_dir: Union[str, Path] = DEFAULT_OUTPUT_DIR
    split: str = "train"
    max_examples_per_dataset: int = 1
    target_valid_per_dataset: int = 1000
    max_rounds: int = 5
    retrieval_top_k: int = 5
    seed: int = 42
    dry_run: bool = False
    resume: bool = True
    force_final_answer: bool = False
    model: str = DEFAULT_MODEL
    endpoint: str = DEFAULT_ENDPOINT
    api_key_env: str = "DASHSCOPE_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 1200
    request_timeout: int = 120
    request_retries: int = 3
    retry_sleep_seconds: float = 2.0
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    spacy_model: str = "en_core_web_sm"
    sft_sample_workers: int = 4
    retrieval_workers: int = 8
    batch_size: int = 128
    use_vectorized_retrieval: bool = True


def _load_yaml_config(path: Union[str, Path]) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    if config_path.suffix.lower() not in {".yml", ".yaml"}:
        raise ValueError(f"Unsupported config type: {config_path.suffix}")

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyYAML is required to load .yml/.yaml config.") from exc

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config format at {config_path}: expected mapping.")
    return {key.replace("-", "_"): value for key, value in data.items()}


def _coalesce(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def _coerce_list(value: Any, *, fallback: list[str], name: str) -> list[str]:
    if value is None:
        return list(fallback)
    if isinstance(value, str):
        return [value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, list):
        return [str(item) for item in value]
    raise TypeError(f"{name} must be a list; got {type(value)}")


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if value is None:
        return fallback
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y"}:
            return True
        if normalized in {"0", "false", "no", "n"}:
            return False
    return bool(value)


def _coerce_config(config: dict[str, Any], args: argparse.Namespace) -> SFTConfig:
    defaults = SFTConfig()
    return SFTConfig(
        datasets=_coerce_list(
            _coalesce(getattr(args, "datasets", None), config.get("datasets")),
            fallback=defaults.datasets,
            name="datasets",
        ),
        source_root=_coalesce(getattr(args, "source_root", None), config.get("source_root", defaults.source_root)),
        retrieval_root=_coalesce(
            getattr(args, "retrieval_root", None),
            config.get("retrieval_root", defaults.retrieval_root),
        ),
        output_dir=_coalesce(getattr(args, "output_dir", None), config.get("output_dir", defaults.output_dir)),
        split=str(_coalesce(getattr(args, "split", None), config.get("split", defaults.split))),
        max_examples_per_dataset=int(
            _coalesce(
                getattr(args, "max_examples_per_dataset", None),
                config.get("max_examples_per_dataset", defaults.max_examples_per_dataset),
            )
        ),
        target_valid_per_dataset=int(
            _coalesce(
                getattr(args, "target_valid_per_dataset", None),
                config.get("target_valid_per_dataset", defaults.target_valid_per_dataset),
            )
        ),
        max_rounds=int(_coalesce(getattr(args, "max_rounds", None), config.get("max_rounds", defaults.max_rounds))),
        retrieval_top_k=int(
            _coalesce(getattr(args, "retrieval_top_k", None), config.get("retrieval_top_k", defaults.retrieval_top_k))
        ),
        seed=int(_coalesce(getattr(args, "seed", None), config.get("seed", defaults.seed))),
        dry_run=_coerce_bool(_coalesce(getattr(args, "dry_run", None), config.get("dry_run")), defaults.dry_run),
        resume=_coerce_bool(_coalesce(getattr(args, "resume", None), config.get("resume")), defaults.resume),
        force_final_answer=_coerce_bool(config.get("force_final_answer"), defaults.force_final_answer),
        model=str(_coalesce(getattr(args, "model", None), config.get("model", defaults.model))),
        endpoint=str(_coalesce(getattr(args, "endpoint", None), config.get("endpoint", defaults.endpoint))),
        api_key_env=str(_coalesce(getattr(args, "api_key_env", None), config.get("api_key_env", defaults.api_key_env))),
        temperature=float(_coalesce(config.get("temperature"), defaults.temperature)),
        max_tokens=int(_coalesce(config.get("max_tokens"), defaults.max_tokens)),
        request_timeout=int(_coalesce(config.get("request_timeout"), defaults.request_timeout)),
        request_retries=int(_coalesce(config.get("request_retries"), defaults.request_retries)),
        retry_sleep_seconds=float(_coalesce(config.get("retry_sleep_seconds"), defaults.retry_sleep_seconds)),
        embedding_model=str(_coalesce(config.get("embedding_model"), defaults.embedding_model)),
        spacy_model=str(_coalesce(config.get("spacy_model"), defaults.spacy_model)),
        sft_sample_workers=int(
            _coalesce(
                getattr(args, "sft_sample_workers", None),
                _coalesce(
                config.get("sft_sample_workers"),
                config.get("sample_workers", defaults.sft_sample_workers),
                ),
            )
        ),
        retrieval_workers=int(
            _coalesce(
                getattr(args, "retrieval_workers", None),
                _coalesce(
                config.get("retrieval_workers"),
                config.get("max_workers", defaults.retrieval_workers),
                ),
            )
        ),
        batch_size=int(_coalesce(config.get("batch_size"), defaults.batch_size)),
        use_vectorized_retrieval=_coerce_bool(config.get("use_vectorized_retrieval"), defaults.use_vectorized_retrieval),
    )


def _split_path(source_root: Union[str, Path], dataset: str, split: str) -> Path:
    dataset_dir = Path(source_root) / dataset
    candidates = [
        dataset_dir / f"{dataset}_{split}.jsonl",
        dataset_dir / f"examples.{split}.jsonl",
        dataset_dir / f"{split}.jsonl",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {dataset}/{split}. Tried: {', '.join(str(path) for path in candidates)}")


def read_examples(config: SFTConfig) -> list[dict[str, Any]]:
    rng = random.Random(config.seed)
    examples: list[dict[str, Any]] = []
    for dataset in config.datasets:
        path = _split_path(config.source_root, dataset, config.split)
        rows = [row for row in read_jsonl(path) if row.get("usable_for_sft", True)]
        for row in rows:
            row.setdefault("dataset", dataset)
        if len(rows) > config.max_examples_per_dataset:
            rows = rng.sample(rows, config.max_examples_per_dataset)
            rows.sort(key=lambda item: str(item.get("qid", "")))
        examples.extend(rows)
    return examples


def parse_teacher_json(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, flags=re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Teacher response is not valid JSON: {content[:500]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Teacher response must be a JSON object.")
    return parsed


def _json_block(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _compact_example(example: dict[str, Any]) -> dict[str, Any]:
    return {
        "qid": example.get("qid"),
        "dataset": example.get("dataset"),
        "question": example.get("question"),
        "question_type": example.get("question_type"),
        "hop_count": example.get("hop_count"),
    }


def build_planning_messages(example: dict[str, Any], state: dict[str, Any]) -> list[dict[str, str]]:
    system = (
        "Task: plan the next knowledge-base query. Return strict JSON only."
    )
    user = (
        "Use only the question and verified facts in <state>. Avoid repeated queries and unsupported intermediate facts.\n"
        "- Do not include inferred intermediate answers that are not already present in the question or current state evidence.\n"
        "- Use entity names and attributes from the question or selected evidence only.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "query_retriever": {"sub_goal": string, "query": string}\n'
        "}\n\n"
        f"QA example:\n{_json_block(_compact_example(example))}\n\n"
        f"<state>{_json_block(state)}</state>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_update_messages(
    example: dict[str, Any],
    state: dict[str, Any],
    plan: dict[str, Any],
    observation: dict[str, Any],
    *,
    force_final_answer: bool = False,
) -> list[dict[str, str]]:
    _ = force_final_answer
    _ = plan
    system = (
        "Task: select evidence from the latest observation. Return strict JSON only."
    )
    user = (
        "Pick only passage IDs from <observation> that support the question, current sub-goal, or a needed reasoning step.\n"
        "- Keep rationale fields short, no more than one brief sentence.\n\n"
        "Required JSON schema:\n"
        "{\n"
        '  "update_evidence": {\n'
        '    "selected_passage_ids": [integer],\n'
        '    "rationale": string\n'
        "  }\n"
        "}\n\n"
        f"QA example:\n{_json_block(_compact_example(example))}\n\n"
        f"<state>{_json_block(state)}</state>\n"
        f"<observation>{_json_block(observation)}</observation>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_answer_messages(
    example: dict[str, Any],
    state: dict[str, Any],
    *,
    force_final_answer: bool = False,
) -> list[dict[str, str]]:
    _ = force_final_answer
    system = (
        "Task: answer from accumulated evidence. Return strict JSON only."
    )
    user = (
        "Use selected evidence in <state>, not the latest observation alone.\n"
        'If the evidence is insufficient but retrieval budget remains, set "can_answer" to false and "answer" to null.\n'
        'If the evidence is insufficient and retrieval budget is exhausted, you may provide a fallback guess using '
        'parametric knowledge, but you must mark it as "fallback_guess" and explain what evidence is missing.\n\n'
        "Required JSON schema:\n"
        "{\n"
        '  "answer": {"can_answer": boolean, "answer": string or null, "rationale": string}\n'
        "}\n\n"
        f"QA example:\n{_json_block(_compact_example(example))}\n\n"
        f"<state>{_json_block(state)}</state>"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_chat_sample(
    *,
    example: dict[str, Any],
    state: dict[str, Any],
    teacher_output: dict[str, Any],
    observation: Optional[dict[str, Any]] = None,
    round_index: int = 0,
) -> dict[str, Any]:
    prompt = (
        "<question>\n"
        f"{example.get('question', '')}\n"
        "</question>\n\n"
        "<state>\n"
        f"{_json_block(state)}\n"
        "</state>\n\n"
        "<instruction>\n"
        "Produce the next structured output from the question, state, and available knowledge using exactly these tags: "
        "<query-retriever>, <update-evidence>, <answer>.\n"
        "</instruction>"
    )
    if observation is not None:
        prompt += f"\n\n<observation>\n{_json_block(observation)}\n</observation>"

    completion = (
        "<query-retriever>\n"
        f"{_json_block(teacher_output.get('query_retriever', {}))}\n"
        "</query-retriever>\n"
        "<update-evidence>\n"
        f"{_json_block(teacher_output.get('update_evidence', {}))}\n"
        "</update-evidence>\n"
        "<answer>\n"
        f"{_json_block(teacher_output.get('answer', {}))}\n"
        "</answer>"
    )
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "Follow the requested tags. Ground answers in selected evidence."
                ),
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ],
        "metadata": {
            "qid": example.get("qid"),
            "dataset": example.get("dataset"),
            "split": example.get("split"),
            "round": round_index,
            "gold_answer": example.get("answer"),
            "question_type": example.get("question_type"),
            "hop_count": example.get("hop_count"),
        },
    }


FORBIDDEN_ASSISTANT_TERMS = (
    "gold",
    "gold answer",
    "gold supporting facts",
    "supervised trajectory",
    "completed from",
)


def _split_title_text(passage: str) -> tuple[str, str]:
    lines = [line.strip() for line in str(passage).splitlines() if line.strip()]
    if not lines:
        return "", ""
    if len(lines) == 1:
        return "", lines[0]
    return lines[0], " ".join(lines[1:]).strip()


def normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    passages = observation.get("passages", [])
    scores = observation.get("scores", [])
    normalized_passages: list[dict[str, Any]] = []
    for passage_id, passage in enumerate(passages):
        score = scores[passage_id] if passage_id < len(scores) else None
        if isinstance(passage, dict):
            title = str(passage.get("title") or "").strip()
            text = str(passage.get("text") or passage.get("passage") or "").strip()
            score = passage.get("score", score)
        else:
            title, text = _split_title_text(str(passage))
        normalized_passages.append(
            {
                "passage_id": passage_id,
                "title": title,
                "text": text,
                "score": score,
            }
        )
    return {
        "query": observation.get("query"),
        "passages": normalized_passages,
    }


def normalize_retrieval_action(action: dict[str, Any], *, observation: dict[str, Any]) -> dict[str, Any]:
    top_k = len(observation.get("passages", []))
    return {
        "query": str(action.get("query") or "").strip(),
        "top_k": top_k,
    }


def _coerce_selected_passage_ids(update: dict[str, Any], passage_count: int) -> list[int]:
    raw_ids = update.get("selected_passage_ids")
    if raw_ids is None and isinstance(update.get("selected"), list):
        raw_ids = []
        for item in update["selected"]:
            if not isinstance(item, dict):
                continue
            if item.get("passage_id") is not None:
                raw_ids.append(item.get("passage_id"))
            elif item.get("rank") is not None:
                raw_ids.append(item.get("rank"))
    if raw_ids is None:
        raw_ids = []

    ids: list[int] = []
    for raw_id in raw_ids:
        try:
            passage_id = int(raw_id)
        except (TypeError, ValueError):
            continue
        if 0 <= passage_id < passage_count and passage_id not in ids:
            ids.append(passage_id)
    return ids


def normalize_update_evidence(
    update: dict[str, Any],
    *,
    observation: dict[str, Any],
    round_index: int,
) -> dict[str, Any]:
    passages = observation.get("passages", [])
    selected_passage_ids = _coerce_selected_passage_ids(update, len(passages))
    evidence = []
    for passage_id in selected_passage_ids:
        passage = passages[passage_id]
        evidence.append(
            {
                "round": round_index,
                "passage_id": passage_id,
                "title": passage.get("title", ""),
                "text": passage.get("text", ""),
                "score": passage.get("score"),
                "source_query": observation.get("query"),
            }
        )
    return {
        "selected_passage_ids": selected_passage_ids,
        "evidence": evidence,
        "rationale": short_rationale(update.get("rationale")),
    }


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).casefold().strip()


def short_rationale(value: Any, *, max_words: int = 6) -> str:
    words = str(value or "").strip().split()
    return " ".join(words[:max_words])


def _capitalized_phrases(value: str) -> list[str]:
    phrases = re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*(?:\s+[A-Z][A-Za-zÀ-ÖØ-öø-ÿ'’-]*)*\b", value)
    stop = {
        "Are",
        "Is",
        "Was",
        "Were",
        "Who",
        "What",
        "Where",
        "When",
        "Which",
        "The",
        "A",
        "An",
    }
    cleaned = []
    for phrase in phrases:
        parts = phrase.split()
        while parts and parts[0] in stop:
            parts = parts[1:]
        if parts:
            cleaned.append(" ".join(parts))
    return [phrase for phrase in cleaned if phrase not in stop]


def _allowed_query_text(question: str, state: dict[str, Any]) -> str:
    evidence_text = " ".join(
        f"{item.get('title', '')} {item.get('text', '')}"
        for item in state.get("evidence", [])
        if isinstance(item, dict)
    )
    return f"{question} {evidence_text}"


def query_has_unseen_intermediate_terms(query: str, *, question: str, state: dict[str, Any]) -> bool:
    allowed_text = _normalize_text(_allowed_query_text(question, state))
    aliases = {
        "canada": ("canadian",),
        "america": ("american",),
        "united states": ("american",),
        "france": ("french",),
        "germany": ("german",),
        "england": ("english", "british"),
        "russia": ("russian",),
        "portugal": ("portuguese",),
    }
    for phrase in _capitalized_phrases(query):
        normalized_phrase = _normalize_text(phrase)
        alias_allowed = any(alias in allowed_text for alias in aliases.get(normalized_phrase, ()))
        if normalized_phrase and normalized_phrase not in allowed_text and not alias_allowed:
            return True
    return False


def comparison_entities_from_question(question: str) -> list[str]:
    phrases = _capitalized_phrases(question)
    entities: list[str] = []
    for phrase in phrases:
        normalized = _normalize_text(phrase)
        if len(normalized) < 3:
            continue
        if normalized not in {_normalize_text(entity) for entity in entities}:
            entities.append(phrase)
    return entities


def answer_supported_by_evidence(
    answer: dict[str, Any],
    evidence: list[dict[str, Any]],
    *,
    question: str = "",
) -> bool:
    if not answer.get("can_answer"):
        return False
    answer_text = _normalize_text(answer.get("answer"))
    if not answer_text:
        return False
    evidence_text = _normalize_text(
        " ".join(f"{item.get('title', '')} {item.get('text', '')}" for item in evidence)
    )
    if answer_text in {"yes", "no"}:
        entities = comparison_entities_from_question(question)
        if len(entities) >= 2:
            return all(_normalize_text(entity) in evidence_text for entity in entities[:2])
        return len(evidence) >= 2
    return answer_text in evidence_text


def normalize_answer(answer: dict[str, Any], *, evidence: list[dict[str, Any]], question: str = "") -> dict[str, Any]:
    normalized = {
        "can_answer": bool(answer.get("can_answer")),
        "answer": answer.get("answer"),
        "rationale": short_rationale(answer.get("rationale")),
    }
    if normalized["can_answer"] and not answer_supported_by_evidence(normalized, evidence, question=question):
        normalized["can_answer"] = False
        normalized["answer"] = None
        normalized["rationale"] = "Current selected evidence does not support a final answer."
    return normalized


def validate_trajectory_sample(sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    trajectory = sample.get("trajectory", [])
    if not sample.get("final_answer"):
        errors.append("sample final_answer must be non-empty")
    if not any(turn.get("answer", {}).get("can_answer") for turn in trajectory):
        errors.append("sample must contain a supported final answer turn")
    for expected_round, turn in enumerate(trajectory):
        round_index = int(turn.get("round", expected_round))
        state = turn.get("state", {})
        evidence = state.get("evidence", [])
        retrieval_history = state.get("retrieval_history", [])
        retrieval_count = int(state.get("retrieval_count", -1))
        if retrieval_count != round_index:
            errors.append(f"round {round_index}: state.retrieval_count must equal round index")
        if len(retrieval_history) != round_index:
            errors.append(f"round {round_index}: state.retrieval_history length must equal round index")
        if round_index == 0 and evidence:
            errors.append("round 0: state.evidence must be empty")
        if len(evidence) > round_index * max(1, len(turn.get("observation", {}).get("passages", []))):
            errors.append(f"round {round_index}: state.evidence appears to contain future evidence")

        retrieval = turn.get("retrieval", {})
        observation = turn.get("observation", {})
        passages = observation.get("passages", [])
        if query_has_unseen_intermediate_terms(
            str(retrieval.get("query") or ""),
            question=str(sample.get("question") or ""),
            state=state,
        ):
            errors.append(f"round {round_index}: retrieval.query contains unseen intermediate answer terms")
        if int(retrieval.get("top_k", -1)) != len(passages):
            errors.append(f"round {round_index}: retrieval.top_k must match observation.passages length")
        for passage_id, passage in enumerate(passages):
            if passage.get("passage_id") != passage_id:
                errors.append(f"round {round_index}: passage_id must be zero-based and contiguous")

        update = turn.get("update_evidence", {})
        if "selected" in update or "discarded_ranks" in update:
            errors.append(f"round {round_index}: update_evidence must use selected_passage_ids, not rank fields")
        for passage_id in update.get("selected_passage_ids", []):
            if not isinstance(passage_id, int) or not 0 <= passage_id < len(passages):
                errors.append(f"round {round_index}: selected_passage_ids contains invalid passage_id")
        for item in update.get("evidence", []):
            if "rank" in item:
                errors.append(f"round {round_index}: evidence item must not contain rank")

        if turn.get("answer", {}).get("can_answer"):
            cumulative_evidence = []
            for prior_turn in trajectory[: expected_round + 1]:
                cumulative_evidence.extend(prior_turn.get("update_evidence", {}).get("evidence", []))
            if not answer_supported_by_evidence(
                turn.get("answer", {}),
                cumulative_evidence,
                question=str(sample.get("question") or ""),
            ):
                errors.append(f"round {round_index}: final answer is not supported by selected evidence")

    content = json.dumps(sample.get("trajectory", []), ensure_ascii=False).casefold()
    for term in FORBIDDEN_ASSISTANT_TERMS:
        if term in content:
            errors.append(f"assistant output contains forbidden training label term: {term}")
    return errors


def _filter_reason_key(error: str) -> str:
    if "unseen intermediate" in error:
        return "unseen_intermediate_query"
    if "final_answer" in error or "supported final answer" in error:
        return "missing_or_unsupported_final_answer"
    if "retrieval_count" in error or "retrieval_history" in error or "state.evidence" in error:
        return "state_leakage"
    if "top_k" in error:
        return "top_k_mismatch"
    if "passage_id" in error or "rank" in error:
        return "invalid_passage_selection"
    if "forbidden" in error:
        return "forbidden_training_label_text"
    if "final answer is not supported" in error:
        return "missing_or_unsupported_final_answer"
    return "validation_error"


def build_trajectory_sample(
    *,
    example: dict[str, Any],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    truncated_turns: list[dict[str, Any]] = []
    for turn in turns:
        truncated_turns.append(turn)
        if turn.get("answer", {}).get("can_answer"):
            break
    turns = truncated_turns

    final_answer = turns[-1].get("answer", {}).get("answer") if turns else None

    return {
        "qid": example.get("qid"),
        "dataset": example.get("dataset"),
        "split": example.get("split"),
        "question": example.get("question"),
        "gold_answer": example.get("answer"),
        "question_type": example.get("question_type"),
        "hop_count": example.get("hop_count"),
        "trajectory": turns,
        "final_answer": final_answer,
    }


class BailianChatClient:
    def __init__(self, config: SFTConfig) -> None:
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Missing API key. Set environment variable {config.api_key_env} before running teacher generation."
            )
        self.config = config
        self.api_key = api_key

    def complete(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.config.endpoint,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Optional[BaseException] = None
        for attempt in range(1, self.config.request_retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.config.request_timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                content = body["choices"][0]["message"]["content"]
                return parse_teacher_json(content)
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                if attempt >= self.config.request_retries:
                    break
                time.sleep(self.config.retry_sleep_seconds * attempt)
        raise RuntimeError(f"Teacher request failed after {self.config.request_retries} attempts: {last_error}")


def finalize_teacher_output(
    teacher_output: dict[str, Any],
    *,
    example: dict[str, Any],
    force_final_answer: bool,
) -> dict[str, Any]:
    _ = example
    _ = force_final_answer
    return teacher_output


def _normalize_query_retriever_output(output: dict[str, Any]) -> dict[str, Any]:
    query_retriever = output.get("query_retriever") if isinstance(output.get("query_retriever"), dict) else None
    if query_retriever is None:
        plan = output.get("plan") if isinstance(output.get("plan"), dict) else {}
        retrieval = output.get("retrieval") if isinstance(output.get("retrieval"), dict) else {}
        query_retriever = {
            "sub_goal": plan.get("sub_goal") or plan.get("sub_query") or retrieval.get("query") or "",
            "query": retrieval.get("query") or plan.get("sub_query") or "",
        }
    sub_goal = str(query_retriever.get("sub_goal") or "").strip()
    query = str(query_retriever.get("query") or "").strip()
    return {
        "query_retriever": {"sub_goal": sub_goal, "query": query},
        "plan": {"sub_goal": sub_goal, "sub_query": query},
        "retrieval": {"query": query},
    }


def _dry_plan(example: dict[str, Any], state: dict[str, Any], top_k: int) -> dict[str, Any]:
    facts = example.get("supporting_facts", [])
    target = facts[min(int(state.get("retrieval_count", 0)), max(len(facts) - 1, 0))] if facts else {}
    title = target.get("title") or ""
    query = f"{title} {example.get('question', '')}".strip() or str(example.get("question", ""))
    return {
        "plan": {
            "sub_goal": "Locate the next supporting fact needed for the multi-hop question.",
            "sub_query": query,
        },
        "retrieval": {
            "query": query,
        },
    }


def _dry_update(example: dict[str, Any], state: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    selected_passage_ids = []
    if observation.get("passages"):
        selected_passage_ids.append(0)
    enough = int(state.get("retrieval_count", 0)) + 1 >= int(example.get("hop_count") or 1)
    return {
        "update_evidence": {
            "selected_passage_ids": selected_passage_ids,
            "rationale": "Dry-run evidence update for pipeline validation.",
        },
        "answer": {
            "can_answer": enough,
            "answer": example.get("answer") if enough else None,
            "rationale": "Dry-run mode emits the answer after the expected hop count.",
        },
    }


def _retrieval_engine_cache() -> dict[tuple[Any, ...], Any]:
    cache = getattr(_RETRIEVAL_THREAD_LOCAL, "engines", None)
    if cache is None:
        cache = {}
        _RETRIEVAL_THREAD_LOCAL.engines = cache
    return cache


def _get_retrieval_engine(config: SFTConfig, dataset: str) -> Any:
    key = (
        str(config.retrieval_root),
        dataset,
        config.embedding_model,
        config.spacy_model,
        config.retrieval_top_k,
        config.retrieval_workers,
        config.batch_size,
        config.use_vectorized_retrieval,
    )
    cache = _retrieval_engine_cache()
    engine = cache.get(key)
    if engine is None:
        engine = create_linear_rag_query_engine(
            retrieval_root=config.retrieval_root,
            dataset=dataset,
            embedding_model=config.embedding_model,
            spacy_model=config.spacy_model,
            top_k=config.retrieval_top_k,
            max_workers=config.retrieval_workers,
            batch_size=config.batch_size,
            use_vectorized_retrieval=config.use_vectorized_retrieval,
        )
        cache[key] = engine
    return engine


def _retrieve(config: SFTConfig, dataset: str, query: str) -> dict[str, Any]:
    os.environ.setdefault("MACORAG_SILENT_RETRIEVAL", "1")
    with _disable_nested_tqdm():
        try:
            result = _get_retrieval_engine(config, dataset).query(query)
        except Exception:
            with _suppress_retrieval_output():
                result = query_linear_rag(
                    retrieval_root=config.retrieval_root,
                    dataset=dataset,
                    query=query,
                    embedding_model=config.embedding_model,
                    spacy_model=config.spacy_model,
                    top_k=config.retrieval_top_k,
                    max_workers=config.retrieval_workers,
                    batch_size=config.batch_size,
                    use_vectorized_retrieval=config.use_vectorized_retrieval,
                )
    return normalize_observation({"query": result.query, "passages": result.passages, "scores": result.scores})


@contextmanager
def _suppress_retrieval_output() -> Any:
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            yield


@contextmanager
def _disable_nested_tqdm() -> Any:
    previous = os.environ.get("TQDM_DISABLE")
    os.environ["TQDM_DISABLE"] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TQDM_DISABLE", None)
        else:
            os.environ["TQDM_DISABLE"] = previous


def _seen_qids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    seen = set()
    for row in read_jsonl(output_path):
        qid = row.get("qid") or row.get("metadata", {}).get("qid")
        if qid is not None:
            seen.add(str(qid))
    return seen


def _existing_output_state(output_path: Path) -> tuple[set[str], Counter[str]]:
    if not output_path.exists():
        return set(), Counter()
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row in read_jsonl(output_path):
        qid = row.get("qid") or row.get("metadata", {}).get("qid")
        if qid is not None:
            seen.add(str(qid))
        counts[str(row.get("dataset") or "unknown")] += 1
    return seen, counts


def _dataset_sft_path(output_dir: Path, dataset: str) -> Path:
    return output_dir / f"{dataset}_sft.jsonl"


def _existing_dataset_output_state(output_paths: dict[str, Path]) -> tuple[set[str], Counter[str]]:
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for dataset, output_path in output_paths.items():
        dataset_seen, dataset_counts = _existing_output_state(output_path)
        seen.update(dataset_seen)
        counts.update(dataset_counts)
        if output_path.exists() and not dataset_counts:
            counts[dataset] += 0
    return seen, counts


def _append_jsonl(handle: Any, item: dict[str, Any]) -> None:
    handle.write(json.dumps(item, ensure_ascii=False))
    handle.write("\n")
    handle.flush()


def _empty_dataset_stats() -> dict[str, Any]:
    return {
        "samples_seen": 0,
        "samples_processed": 0,
        "samples_skipped_existing": 0,
        "samples_written": 0,
        "samples_filtered": 0,
        "samples_failed": 0,
        "samples_skipped_target": 0,
        "target_reached": False,
        "filter_reasons": Counter(),
    }


def _json_ready_dataset_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    ready: dict[str, dict[str, Any]] = {}
    for dataset, values in stats.items():
        ready[dataset] = {
            "samples_seen": values["samples_seen"],
            "samples_processed": values["samples_processed"],
            "samples_skipped_existing": values["samples_skipped_existing"],
            "samples_written": values["samples_written"],
            "samples_filtered": values["samples_filtered"],
            "samples_failed": values["samples_failed"],
            "samples_skipped_target": values["samples_skipped_target"],
            "target_reached": values["target_reached"],
            "target_valid_per_dataset": values.get("target_valid_per_dataset"),
            "filter_reasons": dict(sorted(values["filter_reasons"].items())),
        }
    return ready


def _sample_error_record(example: dict[str, Any], *, stage: str, exc: BaseException) -> dict[str, Any]:
    return {
        "dataset": example.get("dataset"),
        "qid": example.get("qid"),
        "question": example.get("question"),
        "stage": stage,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }


def _process_sft_example(
    *,
    example: dict[str, Any],
    dataset_name: str,
    config: SFTConfig,
    client: Optional[BailianChatClient],
) -> dict[str, Any]:
    stage = "planning"
    try:
        state: dict[str, Any] = {
            "question": example.get("question"),
            "current_sub_goal": None,
            "evidence": [],
            "retrieval_history": [],
            "retrieval_count": 0,
        }
        turns: list[dict[str, Any]] = []
        for round_index in range(config.max_rounds):
            is_final_round = round_index == config.max_rounds - 1
            state_before = copy.deepcopy(state)

            stage = "planning"
            query_output = _dry_plan(example, state, config.retrieval_top_k) if config.dry_run else client.complete(
                build_planning_messages(example, state)
            )
            plan = _normalize_query_retriever_output(query_output)
            query = str(plan.get("query_retriever", {}).get("query") or example["question"])
            if query_has_unseen_intermediate_terms(
                query,
                question=str(example.get("question") or ""),
                state=state_before,
            ):
                return {"status": "filtered", "filter_reasons": ["unseen_intermediate_query"]}

            stage = "retrieval"
            observation = normalize_observation(_retrieve(config, dataset_name, query))
            plan["retrieval"] = normalize_retrieval_action(
                plan.get("retrieval", {}),
                observation=observation,
            )

            stage = "update"
            if config.dry_run:
                update = _dry_update(example, state, observation)
                answer_output = {"answer": update.get("answer", {})}
            else:
                update = client.complete(
                    build_update_messages(
                        example,
                        state,
                        plan,
                        observation,
                        force_final_answer=config.force_final_answer and is_final_round,
                    )
                )
                interim_update = normalize_update_evidence(
                    update.get("update_evidence", {}),
                    observation=observation,
                    round_index=round_index,
                )
                answer_state = {
                    **state,
                    "evidence": [*state["evidence"], *interim_update["evidence"]],
                }
                answer_output = client.complete(
                    build_answer_messages(
                        example,
                        answer_state,
                        force_final_answer=config.force_final_answer and is_final_round,
                    )
                )
            teacher_output = finalize_teacher_output(
                {**plan, **update, **answer_output},
                example=example,
                force_final_answer=config.force_final_answer and is_final_round,
            )
            normalized_update = normalize_update_evidence(
                teacher_output.get("update_evidence", {}),
                observation=observation,
                round_index=round_index,
            )
            cumulative_evidence = [*state["evidence"], *normalized_update["evidence"]]
            normalized_answer = normalize_answer(
                teacher_output.get("answer", {}),
                evidence=cumulative_evidence,
                question=str(example.get("question") or ""),
            )
            turns.append(
                {
                    "round": round_index,
                    "state": state_before,
                    "query_retriever": teacher_output.get("query_retriever", {}),
                    "plan": teacher_output.get("plan", {}),
                    "retrieval": teacher_output.get("retrieval", {}),
                    "observation": observation,
                    "update_evidence": normalized_update,
                    "answer": normalized_answer,
                }
            )

            state["evidence"] = cumulative_evidence
            state["current_sub_goal"] = teacher_output.get("plan", {}).get("sub_goal")
            state["retrieval_history"] = [
                *state["retrieval_history"],
                {
                    "query": query,
                    "top_score": observation.get("passages", [{}])[0].get("score")
                    if observation.get("passages")
                    else None,
                },
            ]
            state["retrieval_count"] = int(state["retrieval_count"]) + 1
            if normalized_answer.get("can_answer"):
                break

        stage = "validation"
        sample = build_trajectory_sample(example=example, turns=turns)
        validation_errors = validate_trajectory_sample(sample)
        if validation_errors:
            return {
                "status": "filtered",
                "filter_reasons": [_filter_reason_key(error) for error in validation_errors],
            }
        return {"status": "written", "sample": sample}
    except BaseException as exc:
        return {"status": "failed", "error": _sample_error_record(example, stage=stage, exc=exc)}


def generate_sft_dataset(config: SFTConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    error_path = output_dir / "teacher_errors.jsonl"
    trace_path = output_dir / "teacher_traces.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    client = None if config.dry_run else BailianChatClient(config)
    samples_seen = 0
    samples_processed = 0
    samples_skipped_existing = 0
    samples_written = 0
    samples_filtered = 0
    samples_failed = 0
    samples_skipped_target = 0
    filter_reasons: Counter[str] = Counter()

    examples = read_examples(config)
    examples_by_dataset: dict[str, list[dict[str, Any]]] = {dataset: [] for dataset in config.datasets}
    for example in examples:
        dataset_name = str(example.get("dataset") or "unknown")
        examples_by_dataset.setdefault(dataset_name, []).append(example)
    dataset_stats = {dataset: _empty_dataset_stats() for dataset in examples_by_dataset}
    output_paths = {dataset: _dataset_sft_path(output_dir, dataset) for dataset in examples_by_dataset}
    if config.resume:
        existing, existing_counts = _existing_dataset_output_state(output_paths)
    else:
        existing, existing_counts = set(), Counter()
        for dataset_output_path in output_paths.values():
            dataset_output_path.write_text("", encoding="utf-8")
    total_samples = sum(existing_counts.values())

    output_mode = "a" if config.resume else "w"
    with error_path.open(
        output_mode,
        encoding="utf-8",
    ) as error_handle:
        for dataset_name, dataset_examples in examples_by_dataset.items():
            current_dataset_stats = dataset_stats[dataset_name]
            current_dataset_stats["target_valid_per_dataset"] = config.target_valid_per_dataset
            dataset_valid_total = existing_counts.get(dataset_name, 0)
            if dataset_valid_total >= config.target_valid_per_dataset:
                current_dataset_stats["target_reached"] = True
                continue
            if not dataset_examples:
                continue

            sft_sample_workers = max(1, int(config.sft_sample_workers))
            next_index = 0
            pending: set[Future[dict[str, Any]]] = set()
            future_examples: dict[Future[dict[str, Any]], dict[str, Any]] = {}
            progress = tqdm(
                total=len(dataset_examples),
                desc=f"{dataset_name} SFT",
                unit="sample",
                dynamic_ncols=True,
                leave=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_inv_fmt}]",
            )

            def submit_available(executor: ThreadPoolExecutor) -> None:
                nonlocal next_index, samples_seen, samples_processed, samples_skipped_existing
                while (
                    len(pending) < sft_sample_workers
                    and next_index < len(dataset_examples)
                    and dataset_valid_total < config.target_valid_per_dataset
                ):
                    example = dataset_examples[next_index]
                    next_index += 1
                    progress.update(1)
                    current_dataset_stats["samples_seen"] += 1
                    samples_seen += 1
                    qid = str(example.get("qid"))
                    if qid in existing:
                        current_dataset_stats["samples_skipped_existing"] += 1
                        samples_skipped_existing += 1
                        continue
                    current_dataset_stats["samples_processed"] += 1
                    samples_processed += 1
                    future = executor.submit(
                        _process_sft_example,
                        example=example,
                        dataset_name=dataset_name,
                        config=config,
                        client=client,
                    )
                    pending.add(future)
                    future_examples[future] = example

            with output_paths[dataset_name].open(output_mode, encoding="utf-8") as output_handle:
                with ThreadPoolExecutor(max_workers=sft_sample_workers) as executor:
                    submit_available(executor)
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            example = future_examples.pop(future)
                            if future.cancelled():
                                continue
                            result = future.result()
                            status = result.get("status")
                            if status == "written":
                                if dataset_valid_total >= config.target_valid_per_dataset:
                                    current_dataset_stats["samples_skipped_target"] += 1
                                    samples_skipped_target += 1
                                    continue
                                sample = result["sample"]
                                _append_jsonl(output_handle, sample)
                                existing.add(str(sample.get("qid")))
                                current_dataset_stats["samples_written"] += 1
                                samples_written += 1
                                dataset_valid_total += 1
                                total_samples += 1
                                if dataset_valid_total >= config.target_valid_per_dataset:
                                    current_dataset_stats["target_reached"] = True
                                    for pending_future in pending:
                                        pending_future.cancel()
                            elif status == "filtered":
                                current_dataset_stats["samples_filtered"] += 1
                                samples_filtered += 1
                                for reason in result.get("filter_reasons", []):
                                    filter_reasons[reason] += 1
                                    current_dataset_stats["filter_reasons"][reason] += 1
                            elif status == "failed":
                                _append_jsonl(error_handle, result["error"])
                                current_dataset_stats["samples_failed"] += 1
                                samples_failed += 1
                        pending = {future for future in pending if not future.cancelled()}
                        submit_available(executor)
            progress.close()

    if trace_path.exists():
        trace_path.unlink()

    summary = {
        "output": str(output_dir),
        "outputs": {dataset: str(path) for dataset, path in output_paths.items()},
        "errors": str(error_path),
        "samples_seen": samples_seen,
        "samples_processed": samples_processed,
        "samples_skipped_existing": samples_skipped_existing,
        "samples_written": samples_written,
        "total_samples": total_samples,
        "samples_filtered": samples_filtered,
        "samples_failed": samples_failed,
        "samples_skipped_target": samples_skipped_target,
        "filter_reasons": dict(sorted(filter_reasons.items())),
        "dataset_stats": _json_ready_dataset_stats(dataset_stats),
        "sample_format": "trajectory_per_question",
        "contains_messages": False,
        "dry_run": config.dry_run,
        "datasets": config.datasets,
        "source_root": str(config.source_root),
        "retrieval_root": str(config.retrieval_root),
        "target_valid_per_dataset": config.target_valid_per_dataset,
        "sft_sample_workers": config.sft_sample_workers,
        "retrieval_workers": config.retrieval_workers,
        "model": config.model,
    }
    write_json(output_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate teacher-model SFT data for RAG actions.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=None)
    parser.add_argument("--source-root", default=None)
    parser.add_argument("--retrieval-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-examples-per-dataset", type=int, default=None)
    parser.add_argument("--target-valid-per-dataset", type=int, default=None)
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--retrieval-top-k", type=int, default=None)
    parser.add_argument("--sft-sample-workers", type=int, default=None)
    parser.add_argument("--retrieval-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", default=None)
    parser.add_argument("--no-resume", dest="resume", action="store_false", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--endpoint", default=None)
    parser.add_argument("--api-key-env", default=None)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _coerce_config(_load_yaml_config(args.config), args)
    summary = generate_sft_dataset(config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
