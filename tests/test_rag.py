from __future__ import annotations

import pytest

from rag import (
    AnswerPromptContext,
    AgentRole,
    RAGLoopExecutor,
    RAGState,
    advance_rag_state,
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


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("TRUE", True)])
def test_parse_answer_normalizes_boolean_strings(raw: str, expected: bool) -> None:
    action = parse_action_text(
        f'<answer>{{"can_answer":"{raw}","answer":"x","rationale":"r"}}</answer>',
        AgentRole.ANSWER_GENERATOR,
    )

    assert action.answer["can_answer"] is expected


@pytest.mark.parametrize("raw", ['"yes"', "1", "null", "[]"])
def test_parse_answer_rejects_invalid_can_answer_types(raw: str) -> None:
    with pytest.raises(ValueError, match="answer.can_answer"):
        parse_action_text(
            f'<answer>{{"can_answer":{raw},"answer":"x"}}</answer>',
            AgentRole.ANSWER_GENERATOR,
        )


@pytest.mark.parametrize(
    "payload, field",
    [
        ('{"can_answer":true,"answer":123}', "answer.answer"),
        ('{"can_answer":true,"answer":"x","rationale":123}', "answer.rationale"),
    ],
)
def test_parse_answer_rejects_non_string_text_fields(payload: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        parse_action_text(f"<answer>{payload}</answer>", AgentRole.ANSWER_GENERATOR)


@pytest.mark.parametrize("selected_ids", ['["0"]', "[true]", "null", "{}"])
def test_parse_evidence_rejects_invalid_passage_ids(selected_ids: str) -> None:
    with pytest.raises(ValueError, match="update_evidence.selected_passage_ids"):
        parse_action_text(
            "<update-evidence>"
            f'{{"selected_passage_ids":{selected_ids}}}'
            "</update-evidence>",
            AgentRole.EVIDENCE_UPDATER,
        )


@pytest.mark.parametrize(
    "payload, field",
    [
        ('{"sub_goal":1,"query":"q"}', "query_retriever.sub_goal"),
        ('{"sub_goal":"s","query":1}', "query_retriever.query"),
    ],
)
def test_parse_query_rejects_non_string_fields(payload: str, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        parse_action_text(
            f"<query-retriever>{payload}</query-retriever>",
            AgentRole.QUERY_RETRIEVER,
        )


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
    assert prompts[0].startswith("Plan the next non-repeated knowledge-base query.")
    assert '<query-retriever>{"sub_goal":"...","query":"..."}</query-retriever>' in prompts[0]
    assert "<observation>" not in prompts[0]
    assert prompts[1].startswith("Task: select evidence from the latest observation.")
    assert "<observation>" in prompts[1]
    assert "<retrieval>" not in prompts[1]
    assert prompts[2].startswith("Answer using only accumulated evidence.")
    assert "<observation>" not in prompts[2]


def test_final_answer_prompt_requires_nonempty_fallback_guess() -> None:
    state = RAGState(question="Who directed Bullitt?")

    prompt = build_answer_generator_prompt(
        question=state.question,
        state=state,
        force_final_answer=True,
    )

    assert "can_answer=true" in prompt
    assert "non-empty" in prompt
    assert "fallback_guess:" in prompt
    assert "can_answer=false" not in prompt


def test_answer_prompt_context_derives_finality_and_remaining_rounds() -> None:
    normal = AnswerPromptContext(round_index=1, max_rounds=4)
    final = AnswerPromptContext(round_index=3, max_rounds=4)

    assert normal.is_final_round is False
    assert normal.remaining_rounds == 2
    assert final.is_final_round is True
    assert final.remaining_rounds == 0


@pytest.mark.parametrize("round_index,max_rounds", [(-1, 4), (0, 0), (4, 4)])
def test_answer_prompt_context_rejects_invalid_rounds(round_index: int, max_rounds: int) -> None:
    with pytest.raises(ValueError, match="round"):
        AnswerPromptContext(round_index=round_index, max_rounds=max_rounds)


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


def test_rag_executor_continues_after_string_false_is_canonicalized() -> None:
    class FakePolicy:
        answer_round = 0

        def generate(self, *, role, question, state, observation=None):
            if role == AgentRole.QUERY_RETRIEVER:
                return '<query-retriever>{"sub_goal":"find","query":"query"}</query-retriever>'
            if role == AgentRole.EVIDENCE_UPDATER:
                return '<update-evidence>{"selected_passage_ids":[]}</update-evidence>'
            self.answer_round += 1
            can_answer = "false" if self.answer_round == 1 else "true"
            answer = "null" if can_answer == "false" else '"done"'
            return f'<answer>{{"can_answer":"{can_answer}","answer":{answer}}}</answer>'

    class FakeRetrievalEnv:
        def query(self, dataset, query):
            return {"query": query, "passages": []}

    result = RAGLoopExecutor(
        policy=FakePolicy(),
        retrieval_env=FakeRetrievalEnv(),
        max_rounds=2,
    ).run(question="question", dataset="hotpotqa")

    assert len(result.trajectory) == 2
    assert result.trajectory[0]["answer"]["can_answer"] is False
    assert result.final_answer == "done"


def test_rag_executor_rejects_false_answer_in_final_round() -> None:
    class FakePolicy:
        def generate(self, *, role, question, state, observation=None, answer_context=None):
            if role == AgentRole.QUERY_RETRIEVER:
                return '<query-retriever>{"sub_goal":"find","query":"query"}</query-retriever>'
            if role == AgentRole.EVIDENCE_UPDATER:
                return '<update-evidence>{"selected_passage_ids":[]}</update-evidence>'
            assert answer_context == AnswerPromptContext(round_index=0, max_rounds=1)
            return '<answer>{"can_answer":false,"answer":null,"rationale":"insufficient"}</answer>'

    class FakeRetrievalEnv:
        def query(self, dataset, query):
            return {"query": query, "passages": []}

    result = RAGLoopExecutor(
        policy=FakePolicy(),
        retrieval_env=FakeRetrievalEnv(),
        max_rounds=1,
    ).run(question="question", dataset="hotpotqa")

    assert result.final_answer is None
    assert result.trajectory[0]["force_final_answer"] is True
    assert result.trajectory[0]["parse_error_role"] == "answer_generator"
    assert result.trajectory[0]["raw_responses"]["answer_generator"].startswith("<answer>")
    assert result.parse_errors == [
        "final_answer_required: answer.can_answer must be true in the final round"
    ]


def test_parse_answer_accepts_null_only_when_can_answer_is_false() -> None:
    action = parse_action_text(
        '<answer>{"can_answer":false,"answer":null,"rationale":"need more evidence"}</answer>',
        AgentRole.ANSWER_GENERATOR,
    )

    assert action.answer == {
        "can_answer": False,
        "answer": None,
        "rationale": "need more evidence",
    }

    with pytest.raises(ValueError, match="answer.answer"):
        parse_action_text(
            '<answer>{"can_answer":true,"answer":null}</answer>',
            AgentRole.ANSWER_GENERATOR,
        )


def test_rag_executor_records_partial_round_and_parse_error_role() -> None:
    class FakePolicy:
        def generate(self, *, role, question, state, observation=None):
            if role == AgentRole.QUERY_RETRIEVER:
                return '<query-retriever>{"sub_goal":"find director","query":"Bullitt director"}</query-retriever>'
            return "invalid evidence output"

    class FakeRetrievalEnv:
        def query(self, dataset: str, query: str) -> dict:
            return {
                "query": query,
                "passages": [
                    {
                        "passage_id": 0,
                        "doc_id": "d1",
                        "title": "Bullitt",
                        "text": "Bullitt was directed by Peter Yates.",
                    }
                ],
            }

    result = RAGLoopExecutor(policy=FakePolicy(), retrieval_env=FakeRetrievalEnv(), max_rounds=2).run(
        question="Who directed Bullitt?",
        dataset="hotpotqa",
    )

    assert len(result.trajectory) == 1
    assert result.trajectory[0]["parse_error_role"] == "evidence_updater"
    assert result.trajectory[0]["generated_roles"] == ["query_retriever", "evidence_updater"]
    assert result.trajectory[0]["raw_responses"] == {
        "query_retriever": '<query-retriever>{"sub_goal":"find director","query":"Bullitt director"}</query-retriever>',
        "evidence_updater": "invalid evidence output",
    }
    assert result.trajectory[0]["observation"]["passages"][0]["doc_id"] == "d1"
    assert len(result.parse_errors) == 1


def test_advance_rag_state_applies_one_shared_transition() -> None:
    state = RAGState(question="q", evidence=[{"text": "old"}], retrieval_count=1)
    observation = {
        "passages": [
            {"passage_id": 0, "title": "T", "text": "new", "score": 0.9},
            {"passage_id": 1, "title": "U", "text": "unused", "score": 0.8},
        ]
    }

    advanced = advance_rag_state(
        state,
        query_action={"sub_goal": "goal", "query": "lookup"},
        observation=observation,
        update_action={"selected_passage_ids": [0]},
    )

    assert advanced.current_sub_goal == "goal"
    assert advanced.retrieval_count == 2
    assert [item["text"] for item in advanced.evidence] == ["old", "new"]
    assert advanced.retrieval_history[-1] == {
        "query": "lookup",
        "sub_goal": "goal",
        "passage_ids": [0],
        "scores": [0.9],
    }


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
