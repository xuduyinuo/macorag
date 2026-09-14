from __future__ import annotations

from dataclasses import dataclass, field

import pytest
import torch

from rag import AgentRole
from rl_training.credit_assignment import (
    broadcast_advantage_to_tokens,
    compute_decision_returns,
    group_relative_normalization,
)
from rl_training.grpo_loss import compute_grpo_loss
from rl_training.rewards import (
    compute_action_rewards,
    compute_answer_reward,
    compute_evidence_reward,
    compute_global_reward,
    compute_query_reward,
)
from rl_training.trajectory import AgentDecision, decision_output_mask
from rl_training.train_grpo_macorag import (
    _validate_online_policy_sync,
    _validate_paper_credit_config,
)


G1 = {"doc_id": "g1", "title": "Gold 1", "text": "alpha"}
G2 = {"doc_id": "g2", "title": "Gold 2", "text": "beta"}
NOISE = {"doc_id": "n1", "title": "Noise", "text": "noise"}


@pytest.mark.parametrize(
    ("current", "previous", "expected"),
    [
        ([G1], [], 0.5),
        ([G1], [G1], -0.2),
        ([G1, G2], [G1], 0.4),
        ([], [], 0.0),
    ],
)
def test_query_reward_formula(current, previous, expected) -> None:
    assert compute_query_reward(
        retrieved_passages=current,
        previously_retrieved_passages=previous,
        gold_passages=[G1, G2],
        eta_query=0.2,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("retrieved", "selected", "expected"),
    [
        ([G1, G2], [G1, G2], 1.0),
        ([G1, G2], [G1], 0.5),
        ([G1, NOISE], [G1, NOISE], 0.9),
        ([G1], [], 0.0),
        ([NOISE], [], 0.0),
    ],
)
def test_evidence_reward_formula(retrieved, selected, expected) -> None:
    assert compute_evidence_reward(
        retrieved_passages=retrieved,
        selected_passages=selected,
        gold_passages=[G1, G2],
        eta_evidence=0.2,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("sufficient", "decision", "expected"),
    [
        (False, "Continue", 1.0),
        (False, "Stop", -1.0),
        (True, "Stop", 1.0),
        (True, "Continue", -1.0),
    ],
)
def test_answer_decision_reward(sufficient, decision, expected) -> None:
    assert compute_answer_reward(
        evidence_sufficient=sufficient, decision=decision
    ) == expected


def test_forced_answer_has_no_local_reward() -> None:
    assert compute_answer_reward(
        evidence_sufficient=True, decision="Stop", forced_termination=True
    ) is None


def test_invalid_structured_outputs_receive_minus_one_without_crashing() -> None:
    rollout = {
        "trajectory": [
            {
                "round": 0,
                "generated_roles": ["query_retriever", "evidence_updater"],
                "parse_error_role": "query_retriever",
                "query_retriever": {},
                "observation": {"passages": [G1]},
                "update_evidence": {"selected_passage_ids": [99]},
                "answer": {},
            }
        ],
        "final_answer": None,
        "parse_errors": ["invalid query"],
    }
    scored = compute_action_rewards(
        rollout=rollout,
        sample={"answer": "alpha", "supporting_facts": [G1]},
    )
    assert [item["local_reward"] for item in scored["action_rewards"]] == [-1.0, -1.0]
    assert all(not item["is_valid"] for item in scored["action_rewards"])


def test_empty_gold_and_empty_passages_are_finite() -> None:
    assert compute_query_reward(
        retrieved_passages=[], previously_retrieved_passages=[], gold_passages=[], eta_query=0.2
    ) == 0.0
    assert compute_evidence_reward(
        retrieved_passages=[], selected_passages=[], gold_passages=[], eta_evidence=0.2
    ) == 0.0
    assert compute_global_reward(
        final_answer=None, gold_answer="x", final_evidence=[], gold_passages=[],
        omega_answer=1.0, omega_evidence=1.0,
    )["global_reward"] == 0.0


def test_vllm_rollouts_require_synchronized_old_policy_weights() -> None:
    class Args:
        use_vllm_generation = True
        vllm_sync_after_step = True
        vllm_sync_every_steps = 2

    with pytest.raises(SystemExit, match="off-policy"):
        _validate_online_policy_sync(Args())


def test_paper_credit_config_rejects_round_groups_and_fallback() -> None:
    class Args:
        advantage_granularity = "role_round"
        degenerate_bucket_fallback_weight = 0.2

    with pytest.raises(SystemExit, match="same-question"):
        _validate_paper_credit_config(Args())


@dataclass
class Action:
    role: AgentRole
    round_index: int
    local_reward: float | None
    completion_ids: list[int] = field(default_factory=lambda: [1])
    global_reward: float = 0.0
    decision_return: float = 0.0
    advantage: float = 0.0


def _rollout(qid: str, rewards: list[tuple[AgentRole, int, float]]) -> dict:
    actions = [Action(role, round_index, reward) for role, round_index, reward in rewards]
    compute_decision_returns(
        actions,
        global_reward=0.5,
        lambda_by_agent={role.value: 1.0 for role in AgentRole},
    )
    return {"question_id": qid, "actions": actions}


def test_group_normalization_uses_question_and_role_across_variable_rounds() -> None:
    rollouts = [
        _rollout("q", [(AgentRole.QUERY_RETRIEVER, 0, 0.0), (AgentRole.ANSWER_GENERATOR, 0, 1.0)]),
        _rollout("q", [(AgentRole.QUERY_RETRIEVER, 0, 1.0), (AgentRole.QUERY_RETRIEVER, 1, 2.0)]),
        _rollout("q", [(AgentRole.QUERY_RETRIEVER, 0, 3.0), (AgentRole.EVIDENCE_UPDATER, 0, -1.0)]),
        _rollout("q", [(AgentRole.QUERY_RETRIEVER, 0, 4.0), (AgentRole.EVIDENCE_UPDATER, 0, 1.0)]),
    ]
    stats = group_relative_normalization(rollouts, advantage_eps=1e-8)
    query_actions = [
        action for rollout in rollouts for action in rollout["actions"]
        if action.role is AgentRole.QUERY_RETRIEVER
    ]
    assert len(query_actions) == 5
    assert sum(action.advantage for action in query_actions) / 5 == pytest.approx(0.0, abs=1e-7)
    assert stats["q:query_retriever"]["count"] == 5
    assert stats["q:answer_generator"]["count"] == 1
    assert rollouts[0]["actions"][1].advantage == 0.0


def test_decision_token_mask_excludes_prompt_and_padding() -> None:
    decision = AgentDecision(
        agent_type=AgentRole.QUERY_RETRIEVER,
        prompt="p",
        generated_text="o",
        input_ids=[10, 11, 12],
        output_token_ids=[20, 21],
        old_log_probs=torch.zeros(2),
    )
    assert decision_output_mask(decision) == [0, 0, 0, 1, 1]
    advantages = broadcast_advantage_to_tokens(
        torch.tensor([0.73]), torch.tensor([[1, 1, 0]], dtype=torch.bool)
    )
    assert advantages[0].tolist() == pytest.approx([0.73, 0.73, 0.0])


@pytest.mark.parametrize(
    ("ratio", "advantage", "expected_policy_loss"),
    [(1.0, 1.0, -1.0), (1.0, -1.0, 1.0), (1.5, 1.0, -1.2), (0.5, -1.0, 0.8)],
)
def test_grpo_clipping(ratio, advantage, expected_policy_loss) -> None:
    current = torch.tensor([[ratio]], dtype=torch.float64).log().requires_grad_()
    old = torch.zeros_like(current)
    loss, metrics = compute_grpo_loss(
        current_logprobs=current,
        old_logprobs=old,
        ref_logprobs=current.detach().clone(),
        action_mask=torch.ones_like(current, dtype=torch.bool),
        advantages=torch.tensor([advantage], dtype=torch.float64),
        clip_epsilon=0.2,
        kl_beta=0.0,
    )
    assert metrics["policy_loss"] == pytest.approx(expected_policy_loss)
    loss.backward()
    assert current.grad is not None


def test_global_reward_reuses_answer_f1_and_evidence_coverage() -> None:
    result = compute_global_reward(
        final_answer="Peter Yates",
        gold_answer="Peter Yates",
        final_evidence=[G1],
        gold_passages=[G1, G2],
        omega_answer=2.0,
        omega_evidence=0.5,
    )
    assert result == pytest.approx({
        "global_reward": 2.25, "answer_f1": 1.0, "evidence_coverage": 0.5,
    })


def test_grpo_backward_masks_padding_and_keeps_reference_detached() -> None:
    current = torch.tensor([[0.0, 0.0, 7.0]], requires_grad=True)
    old = torch.zeros_like(current)
    reference = torch.tensor([[0.0, -0.2, 99.0]], requires_grad=True)
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    loss, _ = compute_grpo_loss(
        current_logprobs=current,
        old_logprobs=old,
        ref_logprobs=reference,
        action_mask=mask,
        advantages=torch.tensor([1.0]),
        clip_epsilon=0.2,
        kl_beta=0.1,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert current.grad is not None and current.grad[0, :2].abs().sum() > 0
    assert current.grad[0, 2] == 0
    assert reference.grad is None
