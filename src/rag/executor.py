from __future__ import annotations

from typing import Any

from .parser import parse_action_text
from .schema import AgentRole, RAGLoopResult, RAGState, RetrievalEnv, SharedPolicy


def normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    passages = []
    for index, passage in enumerate(observation.get("passages", []) or []):
        item = dict(passage) if isinstance(passage, dict) else {"text": str(passage)}
        item["passage_id"] = int(item.get("passage_id", index))
        passages.append(item)
    normalized = dict(observation)
    normalized["passages"] = passages
    return normalized


def _selected_evidence(update: dict[str, Any], observation: dict[str, Any]) -> list[dict[str, Any]]:
    passages = observation.get("passages", []) or []
    evidence: list[dict[str, Any]] = []
    for passage_id in update.get("selected_passage_ids", []) or []:
        if isinstance(passage_id, int) and 0 <= passage_id < len(passages):
            passage = dict(passages[passage_id])
        else:
            passage = next(
                (item for item in passages if str(item.get("passage_id")) == str(passage_id)),
                None,
            )
            if passage is None:
                continue
            passage = dict(passage)
        evidence.append(
            {
                "passage_id": passage.get("passage_id", passage_id),
                "title": passage.get("title"),
                "text": passage.get("text"),
                "score": passage.get("score"),
            }
        )
    return evidence


class RAGLoopExecutor:
    def __init__(self, *, policy: SharedPolicy, retrieval_env: RetrievalEnv, max_rounds: int = 5) -> None:
        self.policy = policy
        self.retrieval_env = retrieval_env
        self.max_rounds = max_rounds

    def run(self, *, question: str, dataset: str) -> RAGLoopResult:
        state = RAGState(question=question)
        trajectory: list[dict[str, Any]] = []
        parse_errors: list[str] = []
        final_answer: str | None = None

        for round_index in range(self.max_rounds):
            state_before = RAGState(
                question=state.question,
                current_sub_goal=state.current_sub_goal,
                evidence=[dict(item) for item in state.evidence],
                retrieval_history=[dict(item) for item in state.retrieval_history],
                retrieval_count=state.retrieval_count,
            )
            try:
                query_text = self.policy.generate(
                    role=AgentRole.QUERY_RETRIEVER,
                    question=question,
                    state=state_before,
                )
                query_action = parse_action_text(query_text, AgentRole.QUERY_RETRIEVER)
                query_retriever = dict(query_action.query_retriever or {})
                query = str(query_retriever.get("query") or "")
                if query:
                    observation = normalize_observation(self.retrieval_env.query(dataset, query))
                else:
                    observation = normalize_observation({"query": query, "passages": []})
                retrieval = {"query": query, "sub_goal": query_retriever.get("sub_goal")}
                retrieval["top_k"] = len(observation.get("passages", []) or [])
                updater_state = RAGState(
                    question=state_before.question,
                    current_sub_goal=query_retriever.get("sub_goal"),
                    evidence=[dict(item) for item in state_before.evidence],
                    retrieval_history=[dict(item) for item in state_before.retrieval_history],
                    retrieval_count=state_before.retrieval_count,
                )

                update_text = self.policy.generate(
                    role=AgentRole.EVIDENCE_UPDATER,
                    question=question,
                    state=updater_state,
                    observation=observation,
                )
                update_action = parse_action_text(update_text, AgentRole.EVIDENCE_UPDATER)
                update = dict(update_action.update_evidence or {})
                update["evidence"] = _selected_evidence(update, observation)
                answer_state = RAGState(
                    question=state_before.question,
                    current_sub_goal=query_retriever.get("sub_goal"),
                    evidence=[*state_before.evidence, *update["evidence"]],
                    retrieval_history=[
                        *state_before.retrieval_history,
                        {
                            "query": query,
                            "sub_goal": query_retriever.get("sub_goal"),
                            "top_score": observation.get("passages", [{}])[0].get("score")
                            if observation.get("passages")
                            else None,
                        },
                    ],
                    retrieval_count=state_before.retrieval_count + (1 if query else 0),
                )

                answer_text = self.policy.generate(
                    role=AgentRole.ANSWER_GENERATOR,
                    question=question,
                    state=answer_state,
                )
                answer_action = parse_action_text(answer_text, AgentRole.ANSWER_GENERATOR)
            except ValueError as exc:
                parse_errors.append(str(exc))
                break

            answer = dict(answer_action.answer or {})
            trajectory.append(
                {
                    "round": round_index,
                    "state": state_before.to_dict(),
                    "query_retriever": query_retriever,
                    "retrieval": retrieval,
                    "observation": observation,
                    "update_evidence": update,
                    "answer": answer,
                }
            )

            state.evidence = [*state.evidence, *update["evidence"]]
            state.current_sub_goal = query_retriever.get("sub_goal")
            state.retrieval_history = [
                *state.retrieval_history,
                {
                    "query": query,
                    "sub_goal": query_retriever.get("sub_goal"),
                    "top_score": observation.get("passages", [{}])[0].get("score")
                    if observation.get("passages")
                    else None,
                },
            ]
            state.retrieval_count += 1 if query else 0
            if answer.get("can_answer"):
                final_answer = answer.get("answer")
                break

        return RAGLoopResult(
            question=question,
            dataset=dataset,
            trajectory=trajectory,
            state=state,
            final_answer=final_answer,
            parse_errors=parse_errors,
        )
