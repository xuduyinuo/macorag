from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from prompt_config import load_prompt_contract
from rag import (
    AnswerPromptContext,
    RAGState,
    advance_rag_state,
    build_answer_generator_prompt,
    build_evidence_updater_prompt,
    build_query_retriever_prompt,
)

@dataclass
class TrajectoryRecord:
    qid: str
    question: str
    dataset: str
    action_type: str
    prompt_text: str
    target_text: str
    agent_role: str = ""
    round_index: int = 0
    max_rounds: int = 4
    prompt_contract_fingerprint: str = ""


@dataclass
class TrainingSample:
    qid: str
    dataset: str
    records: list[TrajectoryRecord]


@dataclass
class TrainingData:
    samples: list[TrainingSample]
    records: list[TrajectoryRecord]
    source_sample_count: int
    source_sample_counts_by_dataset: dict[str, int]


def validate_teacher_dataset_contract(
    data_root: Path,
    *,
    expected_contract: Any,
    max_rounds: int,
    retrieval_top_k: int,
) -> dict[str, Any]:
    metadata_path = Path(data_root) / "run_config.json"
    if not metadata_path.is_file():
        raise SystemExit(f"Teacher dataset provenance is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if str(metadata.get("prompt_contract_version")) != str(expected_contract.version):
        raise SystemExit(
            "Teacher prompt contract version mismatch: "
            f"expected {expected_contract.version}, got {metadata.get('prompt_contract_version')}"
        )
    if str(metadata.get("prompt_contract_fingerprint")) != str(expected_contract.fingerprint):
        raise SystemExit(
            "Teacher prompt contract fingerprint mismatch: "
            f"expected {expected_contract.fingerprint}, got {metadata.get('prompt_contract_fingerprint')}"
        )
    if int(metadata.get("max_rounds", -1)) != int(max_rounds):
        raise SystemExit(
            f"Teacher max_rounds mismatch: expected {max_rounds}, got {metadata.get('max_rounds')}"
        )
    if int(metadata.get("retrieval_top_k", -1)) != int(retrieval_top_k):
        raise SystemExit(
            "Teacher retrieval_top_k mismatch: "
            f"expected {retrieval_top_k}, got {metadata.get('retrieval_top_k')}"
        )
    target = int(metadata.get("target_valid_per_dataset", 0) or 0)
    datasets = [str(item) for item in metadata.get("datasets", [])]
    if target <= 0 or not datasets:
        raise SystemExit("Teacher dataset run_config is missing target or dataset identity")
    for dataset in datasets:
        path = Path(data_root) / f"{dataset}_sft.jsonl"
        if not path.is_file():
            raise SystemExit(f"Teacher dataset file is missing: {path}")
        count = sum(1 for _ in _load_jsonl_records(path))
        if count != target:
            raise SystemExit(f"Teacher dataset count mismatch for {dataset}: expected {target}, got {count}")
    index_provenance = metadata.get("index_provenance") or {}
    missing_indexes = [dataset for dataset in datasets if not (index_provenance.get(dataset) or {}).get("available")]
    if missing_indexes:
        raise SystemExit("Teacher retrieval index provenance is incomplete: " + ", ".join(missing_indexes))
    return metadata


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _mask_update_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    allowed_keys = ("selected_passage_ids", "rationale", "can_answer", "answer")
    return {key: payload[key] for key in allowed_keys if key in payload}


def _tagged_json(tag: str, payload: Any) -> str:
    return f"<{tag}>{json.dumps(payload, ensure_ascii=False)}</{tag}>"


def _rag_state_from_dict(question: str, payload: dict[str, Any]) -> RAGState:
    return RAGState(
        question=question,
        current_sub_goal=payload.get("current_sub_goal"),
        evidence=[dict(item) for item in payload.get("evidence", []) if isinstance(item, dict)],
        retrieval_history=[
            dict(item) for item in payload.get("retrieval_history", []) if isinstance(item, dict)
        ],
        retrieval_count=int(payload.get("retrieval_count") or 0),
    )


def _build_query_retriever_target(plan: dict[str, Any] | None, retrieval: dict[str, Any]) -> dict[str, Any]:
    plan = plan or {}
    return {
        "sub_goal": plan.get("sub_goal") or plan.get("sub_query") or retrieval.get("query") or "",
        "query": retrieval.get("query") or plan.get("sub_query") or "",
    }


def _clean_evidence_for_state(update_evidence: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = update_evidence.get("evidence")
    if not isinstance(evidence, list):
        return []
    cleaned = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        cleaned.append(
            {
                key: item.get(key)
                for key in ("passage_id", "title", "text")
                if key in item
            }
        )
    return cleaned


def _state_with_update_evidence(state: dict[str, Any], update_evidence: dict[str, Any]) -> dict[str, Any]:
    merged = dict(state)
    current_evidence = list(merged.get("evidence") or [])
    merged["evidence"] = [*current_evidence, *_clean_evidence_for_state(update_evidence)]
    return merged


def trajectory_to_sft_records(row: dict[str, Any]) -> list[TrajectoryRecord]:
    question = str(row.get("question", "")).strip()
    if not question:
        return []
    qid = str(row.get("qid", "")).strip() or "unknown"
    dataset = str(row.get("dataset", "")).strip() or "unknown"
    trajectory = row.get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        return []

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    max_rounds = int(row.get("max_rounds") or metadata.get("max_rounds") or 4)
    contract_fingerprint = str(
        row.get("prompt_contract_fingerprint")
        or metadata.get("prompt_contract_fingerprint")
        or load_prompt_contract().fingerprint
    )
    records: list[TrajectoryRecord] = []
    for turn_offset, turn in enumerate(trajectory):
        if not isinstance(turn, dict):
            continue
        state = turn.get("state") if isinstance(turn.get("state"), dict) else {}
        plan = turn.get("plan") if isinstance(turn.get("plan"), dict) else None
        retrieval = turn.get("retrieval") if isinstance(turn.get("retrieval"), dict) else None
        observation = turn.get("observation") if isinstance(turn.get("observation"), dict) else None
        update_evidence = turn.get("update_evidence") if isinstance(turn.get("update_evidence"), dict) else None
        answer = turn.get("answer") if isinstance(turn.get("answer"), dict) else None
        round_index = int(turn.get("round", turn_offset))
        state_before = _rag_state_from_dict(question, state)
        query_target = _build_query_retriever_target(plan, retrieval or {})

        if retrieval:
            records.append(
                TrajectoryRecord(
                    qid=qid,
                    question=question,
                    dataset=dataset,
                    action_type="query_retriever",
                    prompt_text=build_query_retriever_prompt(question=question, state=state_before),
                    target_text=_tagged_json("query-retriever", query_target),
                    agent_role="query_retriever",
                    round_index=round_index,
                    max_rounds=max_rounds,
                    prompt_contract_fingerprint=contract_fingerprint,
                )
            )

        if update_evidence:
            masked_update_evidence = _mask_update_evidence(update_evidence)
        else:
            masked_update_evidence = None
        if retrieval and observation and masked_update_evidence:
            updater_state = RAGState(
                question=state_before.question,
                current_sub_goal=query_target.get("sub_goal"),
                evidence=[dict(item) for item in state_before.evidence],
                retrieval_history=[dict(item) for item in state_before.retrieval_history],
                retrieval_count=state_before.retrieval_count,
            )
            records.append(
                TrajectoryRecord(
                    qid=qid,
                    question=question,
                    dataset=dataset,
                    action_type="evidence_update",
                    prompt_text=build_evidence_updater_prompt(
                        question=question,
                        state=updater_state,
                        observation=observation,
                    ),
                    target_text=_tagged_json("update-evidence", masked_update_evidence),
                    agent_role="evidence_updater",
                    round_index=round_index,
                    max_rounds=max_rounds,
                    prompt_contract_fingerprint=contract_fingerprint,
                )
            )
        if retrieval and observation and masked_update_evidence and answer:
            answer_state = advance_rag_state(
                state_before,
                query_action=query_target,
                observation=observation,
                update_action=masked_update_evidence,
            )
            records.append(
                TrajectoryRecord(
                    qid=qid,
                    question=question,
                    dataset=dataset,
                    action_type="answer",
                    prompt_text=build_answer_generator_prompt(
                        question=question,
                        state=answer_state,
                        context=AnswerPromptContext(round_index=round_index, max_rounds=max_rounds),
                    ),
                    target_text=_tagged_json("answer", answer),
                    agent_role="answer_generator",
                    round_index=round_index,
                    max_rounds=max_rounds,
                    prompt_contract_fingerprint=contract_fingerprint,
                )
            )
    return records


def _resolve_dataset_paths(path: str) -> list[Path]:
    data_root = Path(path)
    if not data_root.is_dir():
        return []
    return [
        data_root / "hotpotqa_sft.jsonl",
        data_root / "2wiki_sft.jsonl",
        data_root / "musique_sft.jsonl",
    ]


def flatten_training_samples(samples: list[TrainingSample]) -> list[TrajectoryRecord]:
    return [record for sample in samples for record in sample.records]


def build_train_records(data_root: Path, max_samples: int | None = None) -> list[TrajectoryRecord]:
    return build_training_data(data_root, max_samples=max_samples).records


def build_training_data(data_root: Path, max_samples: int | None = None) -> TrainingData:
    dataset_files = _resolve_dataset_paths(str(data_root))
    samples: list[TrainingSample] = []
    source_sample_counts_by_dataset: dict[str, int] = {}
    for path in dataset_files:
        if not path.exists():
            raise FileNotFoundError(f"missing trajectory file: {path}")
        dataset = path.stem.replace("_sft", "")
        for row in _load_jsonl_records(path):
            row.setdefault("dataset", dataset)
            row_records = trajectory_to_sft_records(row)
            if row_records:
                qid = row_records[0].qid
                samples.append(TrainingSample(qid=qid, dataset=dataset, records=row_records))
                source_sample_counts_by_dataset[dataset] = source_sample_counts_by_dataset.get(dataset, 0) + 1
            if max_samples is not None and len(samples) >= max_samples:
                break
        if max_samples is not None and len(samples) >= max_samples:
            break
    records = flatten_training_samples(samples)
    return TrainingData(
        samples=samples,
        records=records,
        source_sample_count=len(samples),
        source_sample_counts_by_dataset=source_sample_counts_by_dataset,
    )


def split_training_samples(
    samples: list[TrainingSample],
    ratio: float,
    seed: int,
) -> tuple[list[TrainingSample], list[TrainingSample]]:
    if not ratio or ratio <= 0.0:
        return samples, []
    ratio = max(0.0, min(1.0, ratio))
    if ratio == 0.0:
        return samples, []
    if ratio >= 1.0:
        return [], samples

    rng = random.Random(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)
    split = int(math.floor(len(samples) * ratio))
    val_idx = set(indices[:split])
    val_samples = [samples[i] for i in range(len(samples)) if i in val_idx]
    train_samples = [samples[i] for i in range(len(samples)) if i not in val_idx]
    return train_samples, val_samples


def split_records(records: list[TrajectoryRecord], ratio: float, seed: int) -> tuple[list[TrajectoryRecord], list[TrajectoryRecord]]:
    samples_by_order: list[TrainingSample] = []
    current_key: tuple[str, str] | None = None
    current_records: list[TrajectoryRecord] = []
    for record in records:
        key = (record.dataset, record.qid)
        if current_key is not None and key != current_key:
            samples_by_order.append(TrainingSample(qid=current_key[1], dataset=current_key[0], records=current_records))
            current_records = []
        current_key = key
        current_records.append(record)
    if current_key is not None:
        samples_by_order.append(TrainingSample(qid=current_key[1], dataset=current_key[0], records=current_records))

    train_samples, val_samples = split_training_samples(samples_by_order, ratio, seed)
    return flatten_training_samples(train_samples), flatten_training_samples(val_samples)
