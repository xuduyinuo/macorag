from __future__ import annotations

from .data import RLSample, load_rl_samples
from .credit_assignment import (
    broadcast_advantage_to_tokens,
    compute_decision_returns,
    group_relative_normalization,
)
from .rewards import (
    compute_answer_f1,
    compute_answer_reward,
    compute_evidence_reward,
    compute_global_reward,
    compute_query_reward,
    compute_rl_rewards,
)
from .trainer import compute_grpo_loss
from .trajectory import AgentDecision, RoundStep, Trajectory

__all__ = [
    "RLSample",
    "AgentDecision",
    "RoundStep",
    "Trajectory",
    "broadcast_advantage_to_tokens",
    "compute_answer_reward",
    "compute_answer_f1",
    "compute_decision_returns",
    "compute_evidence_reward",
    "compute_global_reward",
    "compute_grpo_loss",
    "compute_query_reward",
    "compute_rl_rewards",
    "group_relative_normalization",
    "load_rl_samples",
]
