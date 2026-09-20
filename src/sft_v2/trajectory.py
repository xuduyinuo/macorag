from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import hashlib
import json
import re
from typing import Any, Callable, Protocol

from .config import TeacherConfig
from .prompts import PromptContract, answer_messages, evidence_messages, query_messages
from .retrieval import RetrievedPassage
from .source import SourceExample


class JsonTeacher(Protocol):
    def complete_json(self, messages: list[dict[str, str]]) -> tuple[dict[str, Any], dict[str, Any]]: ...


class PassageRetriever(Protocol):
    def search(self, query: str) -> list[RetrievedPassage]: ...


def _shuffle_and_reindex_passages(
    passages: list[RetrievedPassage],
    *,
    seed: int,
    qid: str,
    round_index: int,
    query: str,
    enabled: bool,
) -> list[RetrievedPassage]:
    """Shuffle a round's candidates deterministically, then assign fresh pointers."""

    ordered = list(passages)
    if enabled:
        ordered.sort(
            key=lambda item: hashlib.sha256(
                (
                    f"sft-v2-retrieval\0{seed}\0{qid}\0{round_index}\0{query}"
                    f"\0{item.corpus_id}\0{item.corpus_row}"
                ).encode("utf-8")
            ).digest()
        )
    return [replace(item, pointer=f"P{index}") for index, item in enumerate(ordered)]


def normalize_answer_text(value: Any) -> str:
    text = re.sub(r"[^\w\s]", " ", str(value or "").casefold(), flags=re.UNICODE)
    return " ".join(token for token in text.split() if token not in {"a", "an", "the"})


def answer_matches_any(value: Any, accepted: tuple[str, ...]) -> bool:
    normalized = normalize_answer_text(value)
    return bool(normalized) and normalized in {normalize_answer_text(item) for item in accepted}


def evidence_supports(value: Any, evidence: list[dict[str, Any]]) -> bool:
    normalized = normalize_answer_text(value)
    if not normalized:
        return False
    evidence_text = normalize_answer_text(" ".join(f"{item.get('title', '')} {item.get('text', '')}" for item in evidence))
    if normalized in {"yes", "no"}:
        return len(evidence) >= 2
    return normalized in evidence_text


def _gold_query_leak(query: str, example: SourceExample, state: dict[str, Any]) -> bool:
    visible = normalize_answer_text(
        example.question
        + " "
        + " ".join(f"{item.get('title', '')} {item.get('text', '')}" for item in state["evidence"])
    )
    normalized_query = normalize_answer_text(query)
    for answer in example.answers:
        token = normalize_answer_text(answer)
        if token and token not in {"yes", "no"} and token not in visible and token in normalized_query:
            return True
    return False


def _normalize_query(payload: dict[str, Any]) -> dict[str, str]:
    nested = payload.get("query_retriever") if isinstance(payload.get("query_retriever"), dict) else payload
    query = str(nested.get("query") or "").strip()
    sub_goal = str(nested.get("sub_goal") or "").strip()
    if not query or not sub_goal:
        raise ValueError("Query Agent must return non-empty sub_goal and query")
    return {"sub_goal": sub_goal, "query": query}


def _normalize_evidence(payload: dict[str, Any], passages: list[RetrievedPassage]) -> dict[str, Any]:
    nested = payload.get("update_evidence") if isinstance(payload.get("update_evidence"), dict) else payload
    unexpected = sorted(set(nested) - {"selected_passage_ids"})
    if unexpected:
        raise ValueError(
            "Evidence Agent must output only selected_passage_ids; unexpected fields: "
            + ", ".join(unexpected)
        )
    available = {item.pointer: item for item in passages}
    raw_pointers = nested.get("selected_passage_ids")
    if not isinstance(raw_pointers, list):
        raise ValueError("Evidence Agent selected_passage_ids must be a list")
    pointers: list[str] = []
    for value in raw_pointers:
        pointer = str(value).strip().upper()
        if pointer.isdigit():
            pointer = f"P{pointer}"
        if pointer not in available:
            raise ValueError(f"Evidence Agent returned an unknown passage pointer: {value!r}")
        if pointer not in pointers:
            pointers.append(pointer)
    evidence = [
        {
            "passage_id": pointer,
            "corpus_id": available[pointer].corpus_id,
            "title": available[pointer].title,
            "text": available[pointer].text,
            "score": available[pointer].score,
        }
        for pointer in pointers
    ]
    return {
        "selected_passage_ids": pointers,
        "evidence": evidence,
    }


def _normalize_answer(payload: dict[str, Any]) -> dict[str, Any]:
    nested = payload.get("answer") if isinstance(payload.get("answer"), dict) else payload
    unexpected = sorted(set(nested) - {"can_answer", "answer"})
    if unexpected:
        raise ValueError(
            "Answer Agent must output only can_answer and answer; unexpected fields: "
            + ", ".join(unexpected)
        )
    can_answer = nested.get("can_answer")
    if not isinstance(can_answer, bool):
        raise ValueError("Answer Agent can_answer must be a boolean")
    answer = nested.get("answer")
    if can_answer and not str(answer or "").strip():
        raise ValueError("Answer Agent must return a non-empty answer when can_answer=true")
    if not can_answer:
        answer = None
    return {
        "can_answer": can_answer,
        "answer": str(answer).strip() if answer is not None else None,
    }


@dataclass
class TrajectoryGenerator:
    config: TeacherConfig
    prompts: PromptContract
    client: JsonTeacher | None
    retriever: PassageRetriever
    dry_run: bool = False
    progress_callback: Callable[[str], None] | None = None

    def _mark(self, stage: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(stage)

    def _teacher(self, role: str, messages: list[dict[str, str]], *, example: SourceExample, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        self._mark("api")
        self._mark(role)
        if not self.dry_run:
            if self.client is None:
                raise RuntimeError("DeepSeek client is required outside dry-run mode")
            return self.client.complete_json(messages)
        if role == "query":
            return {"sub_goal": "retrieve evidence needed to answer the question", "query": example.question}, {"dry_run": True}
        if role == "evidence":
            return {"selected_passage_ids": ["P0"]}, {"dry_run": True}
        evidence_text = " ".join(str(item.get("text") or "") for item in state["evidence"])
        matched = next((answer for answer in example.answers if normalize_answer_text(answer) in normalize_answer_text(evidence_text)), None)
        final_round = int(state["retrieval_count"]) >= self.config.max_rounds
        return {
            "can_answer": bool(matched or final_round),
            "answer": matched or (example.primary_answer if final_round else None),
        }, {"dry_run": True}

    @staticmethod
    def _retry_messages(
        base_messages: list[dict[str, str]],
        previous_payload: dict[str, Any],
        correction: str,
    ) -> list[dict[str, str]]:
        return [
            *base_messages,
            {
                "role": "assistant",
                "content": json.dumps(previous_payload, ensure_ascii=False, separators=(",", ":")),
            },
            {"role": "user", "content": correction},
        ]

    @staticmethod
    def _trace_with_validation_attempts(traces: list[dict[str, Any]]) -> dict[str, Any]:
        if len(traces) == 1:
            return traces[0]
        return {
            **traces[-1],
            "validation_retry_count": len(traces) - 1,
            "validation_attempts": traces,
        }

    def _generate_valid_query(
        self,
        *,
        example: SourceExample,
        state: dict[str, Any],
    ) -> tuple[dict[str, str], dict[str, Any]]:
        base_messages = query_messages(self.prompts, question=example.question, state=state)
        messages = base_messages
        traces: list[dict[str, Any]] = []
        for attempt in range(self.config.role_validation_retries + 1):
            raw, trace = self._teacher(
                "query", messages, example=example, state=state,
            )
            traces.append(trace)
            try:
                query = _normalize_query(raw)
                if _gold_query_leak(query["query"], example, state):
                    raise ValueError(
                        "Query Agent leaked an unseen accepted answer into the retrieval query"
                    )
            except ValueError as exc:
                if attempt >= self.config.role_validation_retries:
                    raise
                self._mark("query_retry")
                messages = self._retry_messages(
                    base_messages,
                    raw,
                    "The previous query action was invalid: "
                    f"{exc}. Regenerate only the required JSON fields using only "
                    "the visible question and accumulated evidence. Do not use hidden answers.",
                )
                continue
            return query, self._trace_with_validation_attempts(traces)
        raise AssertionError("unreachable query validation loop")

    def _generate_valid_answer(
        self,
        *,
        example: SourceExample,
        state: dict[str, Any],
        round_index: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_round = round_index + 1 == self.config.max_rounds
        base_messages = answer_messages(
            self.prompts,
            question=example.question,
            state=state,
            round_index=round_index,
            max_rounds=self.config.max_rounds,
        )
        messages = base_messages
        traces: list[dict[str, Any]] = []
        for attempt in range(self.config.role_validation_retries + 1):
            raw, trace = self._teacher(
                "answer", messages, example=example, state=state,
            )
            traces.append(trace)
            try:
                answer = _normalize_answer(raw)
                if final_round and not answer["can_answer"]:
                    raise ValueError("Answer Agent must set can_answer=true on the final round")
            except ValueError as exc:
                if attempt >= self.config.role_validation_retries:
                    raise
                self._mark("answer_retry")
                correction = (
                    "This is the final round. The previous answer action was invalid because "
                    "can_answer must be true with a non-empty concise answer. Regenerate exactly "
                    '{"can_answer":true,"answer":"concise answer"} using only accumulated '
                    "selected evidence. Do not output a rationale or any other field."
                    if final_round
                    else "The previous answer action was invalid: "
                    f"{exc}. Regenerate exactly the required JSON using only accumulated evidence."
                )
                messages = self._retry_messages(base_messages, raw, correction)
                continue
            return answer, self._trace_with_validation_attempts(traces)
        raise AssertionError("unreachable answer validation loop")

    def generate(self, example: SourceExample) -> dict[str, Any]:
        state: dict[str, Any] = {
            "current_sub_goal": None,
            "evidence": [],
            "retrieval_history": [],
            "retrieval_count": 0,
        }
        turns: list[dict[str, Any]] = []
        for round_index in range(self.config.max_rounds):
            state_before = copy.deepcopy(state)
            query, query_trace = self._generate_valid_query(
                example=example,
                state=state_before,
            )

            self._mark("retrieval")
            passages = _shuffle_and_reindex_passages(
                self.retriever.search(query["query"]),
                seed=self.config.seed,
                qid=example.qid,
                round_index=round_index,
                query=query["query"],
                enabled=self.config.shuffle_retrieved_passages,
            )
            observation = {
                "query": query["query"],
                "passages": [item.to_prompt_dict() for item in passages],
            }
            evidence_raw, evidence_trace = self._teacher(
                "evidence",
                evidence_messages(
                    self.prompts,
                    question=example.question,
                    state={**state_before, "current_sub_goal": query["sub_goal"]},
                    observation=observation,
                ),
                example=example,
                state=state_before,
            )
            update = _normalize_evidence(evidence_raw, passages)
            accumulated = [*state_before["evidence"], *update["evidence"]]
            answer_state = {
                **state_before,
                "current_sub_goal": query["sub_goal"],
                "evidence": accumulated,
                "retrieval_count": round_index + 1,
            }
            answer, answer_trace = self._generate_valid_answer(
                example=example,
                state=answer_state,
                round_index=round_index,
            )
            if answer["can_answer"] and not evidence_supports(answer["answer"], accumulated):
                answer = {
                    "can_answer": False,
                    "answer": None,
                }
            turns.append(
                {
                    "round": round_index,
                    "state": state_before,
                    "query_retriever": query,
                    "retrieval": {"query": query["query"], "top_k": len(passages)},
                    "observation": observation,
                    "update_evidence": update,
                    "answer": answer,
                    "teacher_traces": {
                        "query_retriever": query_trace,
                        "evidence_updater": evidence_trace,
                        "answer_generator": answer_trace,
                    },
                }
            )
            state = {
                "current_sub_goal": query["sub_goal"],
                "evidence": accumulated,
                "retrieval_history": [
                    *state_before["retrieval_history"],
                    {
                        "query": query["query"],
                        # Candidate order is shuffled, so P0 is no longer the
                        # maximum-score result. Preserve the actual retrieval
                        # maximum for diagnostics without exposing its rank.
                        "top_score": max((item.score for item in passages), default=None),
                    },
                ],
                "retrieval_count": round_index + 1,
            }
            if answer["can_answer"]:
                break

        final_answer = turns[-1]["answer"]["answer"] if turns else None
        errors = []
        if not final_answer:
            errors.append("trajectory has no supported final answer")
        elif not answer_matches_any(final_answer, example.answers):
            errors.append("final answer does not match any accepted golden answer")
        if errors:
            return {"status": "filtered", "qid": example.qid, "dataset": example.dataset, "errors": errors, "trajectory": turns}
        return {
            "status": "accepted",
            "sample": {
                "qid": example.qid,
                "dataset": example.dataset,
                "source_dataset": example.source_dataset,
                "split": example.split,
                "question": example.question,
                "gold_answer": example.primary_answer,
                "answer_aliases": list(example.answers[1:]),
                "trajectory": turns,
                "final_answer": final_answer,
                "max_rounds": self.config.max_rounds,
                "retrieval_top_k": self.config.retrieval_top_k,
                "prompt_contract_version": self.prompts.version,
                "prompt_contract_fingerprint": self.prompts.fingerprint,
            },
        }
