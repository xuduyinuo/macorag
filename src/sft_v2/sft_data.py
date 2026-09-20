from __future__ import annotations

import json
import hashlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .policy_prompts import PolicyPromptContract, compact_json, policy_messages, tagged_target


@dataclass(frozen=True)
class SFTDecision:
    sample_id: str
    trajectory_id: str
    qid: str
    dataset: str
    role: str
    round_index: int
    final_round: bool
    messages: list[dict[str, str]]
    target: str


@dataclass(frozen=True)
class LoadedSplit:
    path: Path
    trajectory_count: int
    decisions: list[SFTDecision]
    trajectory_counts_by_dataset: dict[str, int]
    decision_counts_by_role: dict[str, int]
    answer_counts: dict[str, int]


def _read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"Expected an object in {path}:{line_number}")
            yield line_number, payload


def _selected_evidence(items: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if not isinstance(items, list):
        return result
    for item in items:
        if not isinstance(item, dict):
            continue
        cleaned = {
            "title": str(item.get("title") or "").strip(),
            "text": str(item.get("text") or "").strip(),
        }
        if cleaned["title"] or cleaned["text"]:
            result.append(cleaned)
    return result


def _retrieval_history(items: Any) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    if not isinstance(items, list):
        return result
    for item in items:
        if isinstance(item, dict) and str(item.get("query") or "").strip():
            result.append({"query": str(item["query"]).strip()})
    return result


def _passages(observation: dict[str, Any]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    raw = observation.get("passages")
    if not isinstance(raw, list):
        raise ValueError("observation.passages must be a list")
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("every observation passage must be an object")
        passage_id = str(item.get("passage_id") or "").strip()
        if not passage_id.startswith("P") or not passage_id[1:].isdigit():
            raise ValueError(f"invalid passage pointer: {passage_id!r}")
        result.append(
            {
                "passage_id": passage_id,
                "title": str(item.get("title") or "").strip(),
                "text": str(item.get("text") or "").strip(),
            }
        )
    if len({item["passage_id"] for item in result}) != len(result):
        raise ValueError("observation contains duplicate passage pointers")
    return result


def _query_target(turn: dict[str, Any]) -> dict[str, str]:
    value = turn.get("query_retriever")
    if not isinstance(value, dict) or set(value) != {"sub_goal", "query"}:
        raise ValueError("query_retriever must contain exactly sub_goal and query")
    target = {key: str(value.get(key) or "").strip() for key in ("sub_goal", "query")}
    if not all(target.values()):
        raise ValueError("query_retriever fields must be non-empty strings")
    return target


def _evidence_target(turn: dict[str, Any], passages: list[dict[str, str]]) -> dict[str, list[str]]:
    value = turn.get("update_evidence")
    if not isinstance(value, dict):
        raise ValueError("update_evidence must be an object")
    pointers = value.get("selected_passage_ids")
    if not isinstance(pointers, list) or any(not isinstance(item, str) for item in pointers):
        raise ValueError("selected_passage_ids must be a list of strings")
    if len(set(pointers)) != len(pointers):
        raise ValueError("selected_passage_ids must not contain duplicates")
    available = {item["passage_id"] for item in passages}
    unknown = sorted(set(pointers) - available)
    if unknown:
        raise ValueError(f"selected_passage_ids not in current observation: {unknown}")
    expanded = value.get("evidence")
    if not isinstance(expanded, list):
        raise ValueError("update_evidence.evidence must be a list")
    expanded_ids = [str(item.get("passage_id") or "") for item in expanded if isinstance(item, dict)]
    if len(expanded_ids) != len(expanded) or expanded_ids != pointers:
        raise ValueError("expanded evidence must exactly match selected_passage_ids")
    passage_by_id = {item["passage_id"]: item for item in passages}
    for item in expanded:
        source = passage_by_id[str(item["passage_id"])]
        if (
            str(item.get("title") or "").strip() != source["title"]
            or str(item.get("text") or "").strip() != source["text"]
        ):
            raise ValueError(f"expanded evidence content differs from observation: {item['passage_id']}")
    return {"selected_passage_ids": pointers}


def _answer_target(turn: dict[str, Any], *, final_round: bool) -> dict[str, Any]:
    value = turn.get("answer")
    if not isinstance(value, dict) or set(value) != {"can_answer", "answer"}:
        raise ValueError("answer must contain exactly can_answer and answer")
    can_answer = value.get("can_answer")
    answer = value.get("answer")
    if not isinstance(can_answer, bool):
        raise ValueError("answer.can_answer must be boolean")
    if can_answer:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("answer must be a non-empty string when can_answer=true")
        answer = answer.strip()
    elif answer is not None:
        raise ValueError("answer must be null when can_answer=false")
    if final_round and not can_answer:
        raise ValueError("the final round must have can_answer=true")
    return {"can_answer": can_answer, "answer": answer}


def _query_prompt(
    question: str,
    state: dict[str, Any],
    *,
    round_index: int,
    max_rounds: int,
) -> str:
    visible_state = {
        "current_sub_goal": state.get("current_sub_goal"),
        "selected_evidence": _selected_evidence(state.get("evidence")),
        "retrieval_history": _retrieval_history(state.get("retrieval_history")),
    }
    return (
        f"Question: {question}\n"
        f"Round: {round_index + 1} of {max_rounds}\n"
        f"Current state:\n{compact_json(visible_state)}"
    )


def _evidence_prompt(
    question: str,
    state: dict[str, Any],
    sub_goal: str,
    passages: list[dict[str, str]],
) -> str:
    return (
        f"Question: {question}\n"
        f"Current sub-goal: {sub_goal}\n"
        "Accumulated selected evidence: "
        f"{compact_json(_selected_evidence(state.get('evidence')))}\n"
        f"Latest observation:\n{compact_json(passages)}"
    )


def _answer_prompt(
    question: str,
    evidence: list[dict[str, str]],
    *,
    round_index: int,
    max_rounds: int,
    final_round: bool,
) -> str:
    marker = " (final round)" if final_round else ""
    return (
        f"Question: {question}\n"
        f"Round: {round_index + 1} of {max_rounds}{marker}\n"
        f"Accumulated selected evidence:\n{compact_json(evidence)}"
    )


def trajectory_to_decisions(
    row: dict[str, Any],
    contract: PolicyPromptContract,
) -> list[SFTDecision]:
    qid = str(row.get("qid") or "").strip()
    dataset = str(row.get("dataset") or row.get("source_dataset") or "unknown").strip()
    question = str(row.get("question") or "").strip()
    trajectory = row.get("trajectory")
    max_rounds = int(row.get("max_rounds") or 4)
    if not qid or not question:
        raise ValueError("trajectory requires non-empty qid and question")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("trajectory must be a non-empty list")
    decisions: list[SFTDecision] = []
    stopped = False
    for offset, turn in enumerate(trajectory):
        if stopped:
            raise ValueError("trajectory contains turns after can_answer=true")
        if not isinstance(turn, dict):
            raise ValueError(f"turn {offset} must be an object")
        round_index = int(turn.get("round", offset))
        if round_index != offset:
            raise ValueError(f"turn order mismatch: expected {offset}, got {round_index}")
        if round_index >= max_rounds:
            raise ValueError(f"round {round_index} exceeds max_rounds={max_rounds}")
        state = turn.get("state")
        observation = turn.get("observation")
        if not isinstance(state, dict) or not isinstance(observation, dict):
            raise ValueError(f"turn {offset} is missing state or observation")
        query = _query_target(turn)
        visible_passages = _passages(observation)
        evidence_action = _evidence_target(turn, visible_passages)
        final_round = round_index + 1 == max_rounds
        answer_action = _answer_target(turn, final_round=final_round)

        role_inputs = {
            "query_retriever": _query_prompt(
                question, state, round_index=round_index, max_rounds=max_rounds
            ),
            "evidence_updater": _evidence_prompt(
                question, state, query["sub_goal"], visible_passages
            ),
        }
        prior_evidence = _selected_evidence(state.get("evidence"))
        selected_now = _selected_evidence((turn.get("update_evidence") or {}).get("evidence"))
        answer_evidence = [*prior_evidence, *selected_now]
        role_inputs["answer_generator"] = _answer_prompt(
            question,
            answer_evidence,
            round_index=round_index,
            max_rounds=max_rounds,
            final_round=final_round,
        )
        targets = {
            "query_retriever": query,
            "evidence_updater": evidence_action,
            "answer_generator": answer_action,
        }
        for role in ("query_retriever", "evidence_updater", "answer_generator"):
            decisions.append(
                SFTDecision(
                    sample_id=f"{qid}:r{round_index}:{role}",
                    trajectory_id=qid,
                    qid=qid,
                    dataset=dataset,
                    role=role,
                    round_index=round_index,
                    final_round=final_round,
                    messages=policy_messages(
                        contract,
                        role,
                        role_inputs[role],
                        final_round=final_round,
                    ),
                    target=tagged_target(contract, role, targets[role]),
                )
            )
        stopped = bool(answer_action["can_answer"])
    if not stopped:
        raise ValueError("trajectory ended without a successful answer")
    return decisions


def load_split(path: str | Path, contract: PolicyPromptContract) -> LoadedSplit:
    source = Path(path).resolve()
    qids: set[str] = set()
    decisions: list[SFTDecision] = []
    trajectories = Counter()
    roles = Counter()
    answers = Counter()
    for line_number, row in _read_jsonl(source):
        qid = str(row.get("qid") or "").strip()
        if qid in qids:
            raise ValueError(f"Duplicate qid {qid!r} in {source}:{line_number}")
        try:
            row_decisions = trajectory_to_decisions(row, contract)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid trajectory in {source}:{line_number} ({qid}): {exc}") from exc
        qids.add(qid)
        dataset = str(row.get("dataset") or row.get("source_dataset") or "unknown").strip()
        trajectories[dataset] += 1
        for item in row_decisions:
            roles[item.role] += 1
            if item.role == "answer_generator":
                answers["can_answer_true" if '"can_answer":true' in item.target else "can_answer_false"] += 1
            decisions.append(item)
    return LoadedSplit(
        path=source,
        trajectory_count=len(qids),
        decisions=decisions,
        trajectory_counts_by_dataset=dict(sorted(trajectories.items())),
        decision_counts_by_role=dict(sorted(roles.items())),
        answer_counts=dict(sorted(answers.items())),
    )


def select_trajectory_subset(split: LoadedSplit, *, limit: int, seed: int) -> LoadedSplit:
    """Select a deterministic trajectory-level subset without splitting its decisions."""
    if limit <= 0 or limit > split.trajectory_count:
        raise ValueError(
            f"subset limit must be in [1, {split.trajectory_count}], got {limit}"
        )
    qid_to_dataset: dict[str, str] = {}
    for decision in split.decisions:
        qid_to_dataset.setdefault(decision.trajectory_id, decision.dataset)
    ranked_qids = sorted(
        qid_to_dataset,
        key=lambda qid: hashlib.sha256(f"{seed}\0{qid}".encode("utf-8")).digest(),
    )
    selected_qids = set(ranked_qids[:limit])
    decisions = [
        decision for decision in split.decisions
        if decision.trajectory_id in selected_qids
    ]
    trajectories = Counter(qid_to_dataset[qid] for qid in selected_qids)
    roles = Counter(decision.role for decision in decisions)
    answers = Counter()
    for decision in decisions:
        if decision.role == "answer_generator":
            key = "can_answer_true" if '"can_answer":true' in decision.target else "can_answer_false"
            answers[key] += 1
    return LoadedSplit(
        path=split.path,
        trajectory_count=len(selected_qids),
        decisions=decisions,
        trajectory_counts_by_dataset=dict(sorted(trajectories.items())),
        decision_counts_by_role=dict(sorted(roles.items())),
        answer_counts=dict(sorted(answers.items())),
    )
