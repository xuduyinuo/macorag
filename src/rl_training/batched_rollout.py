from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rag import AgentRole, RAGLoopResult, RAGState, parse_action_text
from rag.executor import _selected_evidence, normalize_observation

from .policy import PolicyGenerationRequest, RolloutTrace


@dataclass
class BatchedRolloutResult:
    result: RAGLoopResult
    trace: RolloutTrace


@dataclass
class _Candidate:
    question: str
    state: RAGState = field(init=False)
    trajectory: list[dict[str, Any]] = field(default_factory=list)
    parse_errors: list[str] = field(default_factory=list)
    final_answer: str | None = None
    active: bool = True
    trace: RolloutTrace = field(default_factory=RolloutTrace)

    def __post_init__(self) -> None:
        self.state = RAGState(question=self.question)


def _state_snapshot(state: RAGState) -> RAGState:
    return RAGState(
        question=state.question,
        current_sub_goal=state.current_sub_goal,
        evidence=[dict(item) for item in state.evidence],
        retrieval_history=[dict(item) for item in state.retrieval_history],
        retrieval_count=state.retrieval_count,
    )


def _new_turn(round_index: int, state_before: RAGState) -> dict[str, Any]:
    return {
        "round": round_index,
        "state": state_before.to_dict(),
        "generated_roles": [],
        "query_retriever": {},
        "retrieval": {},
        "observation": normalize_observation({"query": "", "passages": []}),
        "update_evidence": {},
        "answer": {},
    }


def _record_parse_failure(
    candidate: _Candidate,
    turn: dict[str, Any],
    role: AgentRole,
    error: ValueError,
) -> None:
    candidate.parse_errors.append(str(error))
    turn["parse_error_role"] = role.value
    candidate.trajectory.append(turn)
    candidate.active = False


def _retrieval_batch(retrieval_env: Any, dataset: str, queries: list[str]) -> list[dict[str, Any]]:
    batch_query = getattr(retrieval_env, "query_batch", None)
    if callable(batch_query):
        return batch_query(dataset, queries)
    return [retrieval_env.query(dataset, query) for query in queries]


def run_batched_rollouts(
    *,
    question: str,
    dataset: str,
    group_size: int,
    max_rounds: int,
    policy: Any,
    retrieval_env: Any,
) -> list[BatchedRolloutResult]:
    candidates = [_Candidate(question=question) for _ in range(group_size)]

    for round_index in range(max_rounds):
        active_indices = [index for index, item in enumerate(candidates) if item.active]
        if not active_indices:
            break

        turns: dict[int, dict[str, Any]] = {}
        states_before: dict[int, RAGState] = {}
        for index in active_indices:
            state_before = _state_snapshot(candidates[index].state)
            states_before[index] = state_before
            turns[index] = _new_turn(round_index, state_before)

        query_requests = [
            PolicyGenerationRequest(
                role=AgentRole.QUERY_RETRIEVER,
                question=question,
                state=states_before[index],
            )
            for index in active_indices
        ]
        query_responses = policy.generate_batch(
            query_requests,
            traces=[candidates[index].trace for index in active_indices],
        )

        query_success_indices: list[int] = []
        queries_by_index: dict[int, str] = {}
        updater_states: dict[int, RAGState] = {}
        for index, response in zip(active_indices, query_responses):
            candidate = candidates[index]
            turn = turns[index]
            turn["generated_roles"].append(AgentRole.QUERY_RETRIEVER.value)
            try:
                action = parse_action_text(response, AgentRole.QUERY_RETRIEVER)
            except ValueError as exc:
                _record_parse_failure(candidate, turn, AgentRole.QUERY_RETRIEVER, exc)
                continue
            query_action = dict(action.query_retriever or {})
            query = str(query_action.get("query") or "")
            turn["query_retriever"] = query_action
            turn["retrieval"] = {"query": query, "sub_goal": query_action.get("sub_goal")}
            queries_by_index[index] = query
            query_success_indices.append(index)

        nonempty_indices = [index for index in query_success_indices if queries_by_index[index]]
        observations = _retrieval_batch(
            retrieval_env,
            dataset,
            [queries_by_index[index] for index in nonempty_indices],
        )
        if len(observations) != len(nonempty_indices):
            raise RuntimeError(
                "Retrieval returned a mismatched rollout batch size: "
                f"expected {len(nonempty_indices)}, got {len(observations)}."
            )
        observations_by_index = {
            index: normalize_observation(observation)
            for index, observation in zip(nonempty_indices, observations)
        }
        for index in query_success_indices:
            turn = turns[index]
            query = queries_by_index[index]
            observation = observations_by_index.get(
                index,
                normalize_observation({"query": query, "passages": []}),
            )
            turn["observation"] = observation
            turn["retrieval"]["top_k"] = len(observation.get("passages", []) or [])
            state_before = states_before[index]
            updater_states[index] = RAGState(
                question=state_before.question,
                current_sub_goal=turn["query_retriever"].get("sub_goal"),
                evidence=[dict(item) for item in state_before.evidence],
                retrieval_history=[dict(item) for item in state_before.retrieval_history],
                retrieval_count=state_before.retrieval_count,
            )

        evidence_requests = [
            PolicyGenerationRequest(
                role=AgentRole.EVIDENCE_UPDATER,
                question=question,
                state=updater_states[index],
                observation=turns[index]["observation"],
            )
            for index in query_success_indices
        ]
        evidence_responses = policy.generate_batch(
            evidence_requests,
            traces=[candidates[index].trace for index in query_success_indices],
        )

        evidence_success_indices: list[int] = []
        answer_states: dict[int, RAGState] = {}
        for index, response in zip(query_success_indices, evidence_responses):
            candidate = candidates[index]
            turn = turns[index]
            turn["generated_roles"].append(AgentRole.EVIDENCE_UPDATER.value)
            try:
                action = parse_action_text(response, AgentRole.EVIDENCE_UPDATER)
            except ValueError as exc:
                _record_parse_failure(candidate, turn, AgentRole.EVIDENCE_UPDATER, exc)
                continue
            update = dict(action.update_evidence or {})
            update["evidence"] = _selected_evidence(update, turn["observation"])
            turn["update_evidence"] = update
            state_before = states_before[index]
            query_action = turn["query_retriever"]
            query = queries_by_index[index]
            observation = turn["observation"]
            answer_states[index] = RAGState(
                question=state_before.question,
                current_sub_goal=query_action.get("sub_goal"),
                evidence=[*state_before.evidence, *update["evidence"]],
                retrieval_history=[
                    *state_before.retrieval_history,
                    {
                        "query": query,
                        "sub_goal": query_action.get("sub_goal"),
                        "top_score": observation.get("passages", [{}])[0].get("score")
                        if observation.get("passages")
                        else None,
                    },
                ],
                retrieval_count=state_before.retrieval_count + (1 if query else 0),
            )
            evidence_success_indices.append(index)

        answer_requests = [
            PolicyGenerationRequest(
                role=AgentRole.ANSWER_GENERATOR,
                question=question,
                state=answer_states[index],
            )
            for index in evidence_success_indices
        ]
        answer_responses = policy.generate_batch(
            answer_requests,
            traces=[candidates[index].trace for index in evidence_success_indices],
        )

        for index, response in zip(evidence_success_indices, answer_responses):
            candidate = candidates[index]
            turn = turns[index]
            turn["generated_roles"].append(AgentRole.ANSWER_GENERATOR.value)
            try:
                action = parse_action_text(response, AgentRole.ANSWER_GENERATOR)
            except ValueError as exc:
                _record_parse_failure(candidate, turn, AgentRole.ANSWER_GENERATOR, exc)
                continue
            answer = dict(action.answer or {})
            turn["answer"] = answer
            candidate.trajectory.append(turn)
            candidate.state = answer_states[index]
            if answer.get("can_answer"):
                candidate.final_answer = answer.get("answer")
                candidate.active = False

    return [
        BatchedRolloutResult(
            result=RAGLoopResult(
                question=question,
                dataset=dataset,
                trajectory=candidate.trajectory,
                state=candidate.state,
                final_answer=candidate.final_answer,
                parse_errors=candidate.parse_errors,
            ),
            trace=candidate.trace,
        )
        for candidate in candidates
    ]
