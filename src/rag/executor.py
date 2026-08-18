from __future__ import annotations

from typing import Any

from .parser import parse_action_text
from .schema import AgentRole, RAGLoopResult, RAGState, RetrievalEnv, SharedPolicy


def normalize_observation(observation: dict[str, Any]) -> dict[str, Any]:
    """统一检索返回格式，保证后续按 passage_id 选择证据时有稳定索引。"""
    passages = []
    for index, passage in enumerate(observation.get("passages", []) or []):
        item = dict(passage) if isinstance(passage, dict) else {"text": str(passage)}
        item["passage_id"] = int(item.get("passage_id", index))
        passages.append(item)
    normalized = dict(observation)
    normalized["passages"] = passages
    return normalized


def _selected_evidence(update: dict[str, Any], observation: dict[str, Any]) -> list[dict[str, Any]]:
    """根据 evidence_updater 输出的 passage_id 提取本轮选中的证据。"""
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
            # 每一轮都从当前状态快照开始，避免模型生成过程意外修改历史状态。
            state_before = RAGState(
                question=state.question,
                current_sub_goal=state.current_sub_goal,
                evidence=[dict(item) for item in state.evidence],
                retrieval_history=[dict(item) for item in state.retrieval_history],
                retrieval_count=state.retrieval_count,
            )
            query_retriever: dict[str, Any] = {}
            retrieval: dict[str, Any] = {}
            observation = normalize_observation({"query": "", "passages": []})
            update: dict[str, Any] = {}
            answer: dict[str, Any] = {}
            generated_roles: list[str] = []
            active_role = AgentRole.QUERY_RETRIEVER
            try:
                # query_retriever 只决定检索 query；真正返回多少段落由检索环境的 top_k 控制。
                query_text = self.policy.generate(
                    role=AgentRole.QUERY_RETRIEVER,
                    question=question,
                    state=state_before,
                )
                generated_roles.append(AgentRole.QUERY_RETRIEVER.value)
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

                # evidence_updater 只把本轮检索结果中有用的 passage 合并进证据池。
                active_role = AgentRole.EVIDENCE_UPDATER
                update_text = self.policy.generate(
                    role=AgentRole.EVIDENCE_UPDATER,
                    question=question,
                    state=updater_state,
                    observation=observation,
                )
                generated_roles.append(AgentRole.EVIDENCE_UPDATER.value)
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

                # answer_generator 可以在任意一轮给出最终答案；can_answer=true 时提前结束。
                active_role = AgentRole.ANSWER_GENERATOR
                answer_text = self.policy.generate(
                    role=AgentRole.ANSWER_GENERATOR,
                    question=question,
                    state=answer_state,
                )
                generated_roles.append(AgentRole.ANSWER_GENERATOR.value)
                answer_action = parse_action_text(answer_text, AgentRole.ANSWER_GENERATOR)
            except ValueError as exc:
                parse_errors.append(str(exc))
                trajectory.append(
                    {
                        "round": round_index,
                        "state": state_before.to_dict(),
                        "generated_roles": generated_roles,
                        "parse_error_role": active_role.value,
                        "query_retriever": query_retriever,
                        "retrieval": retrieval,
                        "observation": observation,
                        "update_evidence": update,
                        "answer": answer,
                    }
                )
                break

            answer = dict(answer_action.answer or {})
            trajectory.append(
                {
                    "round": round_index,
                    "state": state_before.to_dict(),
                    "generated_roles": generated_roles,
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
            if answer["can_answer"] is True:
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
