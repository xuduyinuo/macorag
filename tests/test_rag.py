from __future__ import annotations

import pytest

from rag import (
    AgentRole,
    RAGLoopExecutor,
    RAGState,
    build_answer_generator_prompt,
    build_evidence_updater_prompt,
    build_query_retriever_prompt,
    compute_reward_terms,
    parse_action_text,
    rollout_with_rewards,
)


FORBIDDEN_PROMPT_TERMS = (
    "multi-agent",
    "多智能体",
    "agent_role",
    "planner_retriever",
    "evidence_answerer",
    "你是",
)


def test_parse_query_retriever_action_requires_sub_goal_and_query() -> None:
    action = parse_action_text(
        "<query-retriever>{\"sub_goal\":\"find director\",\"query\":\"The Tripper director\"}</query-retriever>",
        AgentRole.QUERY_RETRIEVER,
    )

    assert action.role == AgentRole.QUERY_RETRIEVER
    assert action.query_retriever == {"sub_goal": "find director", "query": "The Tripper director"}
    assert action.update_evidence is None
    assert action.answer is None


def test_parse_evidence_updater_action_requires_update_evidence() -> None:
    action = parse_action_text(
        "<update-evidence>{\"selected_passage_ids\":[0],\"rationale\":\"supports answer\"}</update-evidence>",
        AgentRole.EVIDENCE_UPDATER,
    )

    assert action.role == AgentRole.EVIDENCE_UPDATER
    assert action.update_evidence == {"selected_passage_ids": [0], "rationale": "supports answer"}
    assert action.answer is None
    assert action.query_retriever is None


def test_parse_answer_generator_action_requires_answer() -> None:
    action = parse_action_text(
        "<answer>{\"can_answer\":true,\"answer\":\"David Arquette\",\"rationale\":\"selected passage says so\"}</answer>",
        AgentRole.ANSWER_GENERATOR,
    )

    assert action.role == AgentRole.ANSWER_GENERATOR
    assert action.answer == {"can_answer": True, "answer": "David Arquette", "rationale": "selected passage says so"}
    assert action.query_retriever is None
    assert action.update_evidence is None


def test_parse_action_text_rejects_missing_required_tag() -> None:
    with pytest.raises(ValueError, match="query-retriever"):
        parse_action_text("<retrieval>{\"query\":\"x\"}</retrieval>", AgentRole.QUERY_RETRIEVER)


def test_parse_action_text_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="query_retriever.query"):
        parse_action_text(
            "<query-retriever>{\"sub_goal\":\"find\"}</query-retriever>",
            AgentRole.QUERY_RETRIEVER,
        )


def test_agent_prompts_use_dedicated_english_templates() -> None:
    state = RAGState(question="Who directed The Tripper?")
    observation = {"passages": [{"passage_id": 0, "text": "Directed by David Arquette."}]}

    prompts = [
        build_query_retriever_prompt(question=state.question, state=state),
        build_evidence_updater_prompt(
            question=state.question,
            state=state,
            observation=observation,
        ),
        build_answer_generator_prompt(
            question=state.question,
            state=state,
        ),
    ]

    for prompt in prompts:
        assert "You are a retrieval-augmented reasoning assistant" not in prompt
        assert "Important constraints:" not in prompt
        assert len(prompt) < 1200
        assert not any(term in prompt for term in FORBIDDEN_PROMPT_TERMS)

    assert len({prompt.splitlines()[0] for prompt in prompts}) == 3
    assert prompts[0].startswith("Task: plan the next knowledge-base query.")
    assert 'Return exactly: <query-retriever>{"sub_goal":"...","query":"..."}</query-retriever>' in prompts[0]
    assert "<observation>" not in prompts[0]
    assert prompts[1].startswith("Task: select evidence from the latest observation.")
    assert "<observation>" in prompts[1]
    assert "<retrieval>" not in prompts[1]
    assert prompts[2].startswith("Task: answer from accumulated evidence.")
    assert "<observation>" not in prompts[2]


def test_rag_executor_reuses_one_policy_for_both_agent_roles() -> None:
    class FakePolicy:
        def __init__(self) -> None:
            self.roles: list[AgentRole] = []

        def generate(
            self,
            *,
            role: AgentRole,
            question: str,
            state: RAGState,
            observation=None,
        ) -> str:
            self.roles.append(role)
            if role == AgentRole.QUERY_RETRIEVER:
                return "<query-retriever>{\"sub_goal\":\"find director\",\"query\":\"The Tripper director\"}</query-retriever>"
            if role == AgentRole.EVIDENCE_UPDATER:
                return "<update-evidence>{\"selected_passage_ids\":[0],\"rationale\":\"supports answer\"}</update-evidence>"
            return (
                "<answer>{\"can_answer\":true,\"answer\":\"David Arquette\",\"rationale\":\"supported\"}</answer>"
            )

    class FakeRetrievalEnv:
        def query(self, dataset: str, query: str) -> dict:
            assert dataset == "hotpotqa"
            assert query == "The Tripper director"
            return {
                "query": query,
                "passages": [
                    {
                        "passage_id": 0,
                        "title": "The Tripper",
                        "text": "The Tripper was directed by David Arquette.",
                        "score": 0.9,
                    }
                ],
            }

    policy = FakePolicy()
    result = RAGLoopExecutor(policy=policy, retrieval_env=FakeRetrievalEnv(), max_rounds=3).run(
        question="Who directed The Tripper?",
        dataset="hotpotqa",
    )

    assert policy.roles == [
        AgentRole.QUERY_RETRIEVER,
        AgentRole.EVIDENCE_UPDATER,
        AgentRole.ANSWER_GENERATOR,
    ]
    assert result.final_answer == "David Arquette"
    assert result.trajectory[0]["retrieval"]["query"] == "The Tripper director"
    assert result.trajectory[0]["query_retriever"]["sub_goal"] == "find director"
    assert result.trajectory[0]["state"]["retrieval_count"] == 0
    assert result.trajectory[0]["update_evidence"]["evidence"][0]["text"].startswith("The Tripper was directed")
    assert result.state.retrieval_count == 1


def test_compute_reward_terms_scores_answer_evidence_format_and_cost() -> None:
    trajectory = [
        {
            "retrieval": {"query": "The Tripper director"},
            "update_evidence": {
                "selected_passage_ids": [0],
                "evidence": [{"title": "The Tripper", "text": "The Tripper was directed by David Arquette."}],
            },
            "answer": {"can_answer": True, "answer": "David Arquette"},
        }
    ]

    reward = compute_reward_terms(
        trajectory=trajectory,
        final_answer="David Arquette",
        gold_answer="David Arquette",
        answer_aliases=[],
        parse_errors=[],
    )

    assert reward["answer_correct"] == 1.0
    assert reward["evidence_supported"] == 1.0
    assert reward["format_valid"] == 1.0
    assert reward["retrieval_cost"] == -1.0
    assert reward["total"] > 0.0


def test_compute_reward_terms_penalizes_parse_errors() -> None:
    reward = compute_reward_terms(
        trajectory=[],
        final_answer=None,
        gold_answer="David Arquette",
        answer_aliases=[],
        parse_errors=["missing retrieval"],
    )

    assert reward["format_valid"] == 0.0
    assert reward["answer_correct"] == 0.0
    assert reward["total"] < 0.0


def test_rollout_with_rewards_reuses_executor_output_for_rl() -> None:
    class FakePolicy:
        def generate(
            self,
            *,
            role: AgentRole,
            question: str,
            state: RAGState,
            observation=None,
        ) -> str:
            if role == AgentRole.QUERY_RETRIEVER:
                return "<query-retriever>{\"sub_goal\":\"find director\",\"query\":\"The Tripper director\"}</query-retriever>"
            if role == AgentRole.EVIDENCE_UPDATER:
                return "<update-evidence>{\"selected_passage_ids\":[0],\"rationale\":\"supports answer\"}</update-evidence>"
            return (
                "<answer>{\"can_answer\":true,\"answer\":\"David Arquette\",\"rationale\":\"supported\"}</answer>"
            )

    class FakeRetrievalEnv:
        def query(self, dataset: str, query: str) -> dict:
            return {
                "query": query,
                "passages": [
                    {
                        "passage_id": 0,
                        "title": "The Tripper",
                        "text": "The Tripper was directed by David Arquette.",
                        "score": 0.9,
                    }
                ],
            }

    rollout = rollout_with_rewards(
        executor=RAGLoopExecutor(policy=FakePolicy(), retrieval_env=FakeRetrievalEnv(), max_rounds=2),
        question="Who directed The Tripper?",
        dataset="hotpotqa",
        gold_answer="David Arquette",
        answer_aliases=[],
    )

    assert rollout["result"].final_answer == "David Arquette"
    assert rollout["reward_terms"]["answer_correct"] == 1.0
    assert rollout["reward_terms"]["format_valid"] == 1.0
