from __future__ import annotations

import copy
import math
import re
from collections import Counter
from typing import Any

TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_RE.findall(text))


def _cosine(query_counter: Counter[str], doc_counter: Counter[str]) -> float:
    if not query_counter or not doc_counter:
        return 0.0

    dot = sum(count * doc_counter[token] for token, count in query_counter.items())
    if dot == 0:
        return 0.0

    query_norm = math.sqrt(sum(count * count for count in query_counter.values()))
    doc_norm = math.sqrt(sum(count * count for count in doc_counter.values()))
    return dot / (query_norm * doc_norm)


class InMemoryRetrievalEnv:
    def __init__(
        self,
        questions: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        retrieval_budget: int,
    ) -> None:
        internal_questions = copy.deepcopy(questions)
        self._chunks = copy.deepcopy(chunks)
        self._questions = {
            question.get("qid", question.get("id")): question
            for question in internal_questions
        }
        self._chunk_counters = [
            _tokens(f"{chunk.get('title', '')} {chunk.get('text', '')}")
            for chunk in self._chunks
        ]
        self._retrieval_budget = retrieval_budget
        self._state: dict[str, Any] | None = None

    def reset(self, qid: str) -> dict[str, Any]:
        question = self._questions[qid]
        self._state = {
            "qid": qid,
            "dataset": question["dataset"],
            "question": question["question"],
            "evidence": [],
            "retrieval_history": [],
            "retrieval_count": 0,
            "retrieval_budget": self._retrieval_budget,
        }
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("Retrieval environment has not been reset")
        return copy.deepcopy(self._state)

    def step(self, query: str, top_k: int) -> list[dict[str, Any]]:
        if self._state is None:
            raise RuntimeError("Retrieval environment has not been reset")
        if self._state["retrieval_count"] >= self._retrieval_budget:
            raise RuntimeError("Retrieval budget exceeded")

        query_counter = _tokens(query)
        scored_chunks = [
            (_cosine(query_counter, doc_counter), index, chunk)
            for index, (chunk, doc_counter) in enumerate(
                zip(self._chunks, self._chunk_counters, strict=True)
            )
        ]
        scored_chunks.sort(key=lambda item: (-item[0], item[1]))

        observation = [
            {**copy.deepcopy(chunk), "score": score}
            for score, _index, chunk in scored_chunks[:top_k]
        ]
        self._state["retrieval_count"] += 1
        self._state["retrieval_history"].append(
            {
                "query": query,
                "top_k": top_k,
                "results": copy.deepcopy(observation),
            }
        )
        self._state["evidence"].extend(copy.deepcopy(observation))
        return observation
