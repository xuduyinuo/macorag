from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable


def _role_name(action: Any) -> str:
    role = getattr(action, "agent_type", getattr(action, "role", ""))
    return str(getattr(role, "value", role))


def compute_decision_returns(
    actions: Iterable[Any],
    *,
    global_reward: float,
    lambda_by_agent: dict[str, float],
) -> None:
    """Apply G_i,t^j = r_i,t^j + lambda_j R(tau_i) in place."""

    if not math.isfinite(float(global_reward)):
        raise ValueError("global_reward must be finite")
    for action in actions:
        role = _role_name(action)
        if role not in lambda_by_agent:
            raise ValueError(f"Missing lambda for agent role {role!r}")
        local = getattr(action, "local_reward", None)
        # Forced max-round answers have no stop-decision local reward.  Their
        # generated answer can still receive trajectory-level QA credit.
        local_value = 0.0 if local is None else float(local)
        value = local_value + float(lambda_by_agent[role]) * float(global_reward)
        if not math.isfinite(value):
            raise ValueError("decision return must be finite")
        action.global_reward = float(global_reward)
        if hasattr(action, "terminal_reward"):
            action.terminal_reward = float(global_reward)
        action.decision_return = value


def group_by_question_and_agent(
    trajectories: Iterable[Any],
) -> dict[tuple[str, str], list[Any]]:
    """Group all actual rounds by question and role, never by round index."""

    groups: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for index, trajectory in enumerate(trajectories):
        question_id = str(
            getattr(trajectory, "question_id", "")
            or (trajectory.get("question_id") if isinstance(trajectory, dict) else "")
            or (trajectory.get("qid") if isinstance(trajectory, dict) else "")
            or f"question-{index}"
        )
        actions = (
            list(trajectory.decisions())
            if callable(getattr(trajectory, "decisions", None))
            else list(trajectory.get("actions", []))
        )
        for action in actions:
            # Invalid structured outputs remain trainable with local reward -1;
            # only actions without generated tokens are absent from optimization.
            if getattr(action, "completion_ids", getattr(action, "output_token_ids", [])):
                groups[(question_id, _role_name(action))].append(action)
    return dict(groups)


def group_relative_normalization(
    trajectories: Iterable[Any],
    *,
    advantage_eps: float = 1.0e-8,
) -> dict[str, dict[str, Any]]:
    """Normalize decision returns over same-question, same-agent groups."""

    if not math.isfinite(advantage_eps) or advantage_eps <= 0.0:
        raise ValueError("advantage_eps must be positive and finite")
    stats: dict[str, dict[str, Any]] = {}
    for (question_id, role), actions in group_by_question_and_agent(trajectories).items():
        returns = [float(action.decision_return) for action in actions]
        if not all(math.isfinite(value) for value in returns):
            raise ValueError("decision returns must be finite")
        mean = sum(returns) / len(returns)
        variance = sum((value - mean) ** 2 for value in returns) / len(returns)
        denominator = math.sqrt(variance + advantage_eps)
        advantages = [(value - mean) / denominator for value in returns]
        for action, advantage in zip(actions, advantages):
            action.advantage = advantage
            if hasattr(action, "primary_advantage"):
                action.primary_advantage = advantage
            if hasattr(action, "fallback_advantage"):
                action.fallback_advantage = 0.0
        advantage_mean = sum(advantages) / len(advantages)
        advantage_variance = sum(
            (value - advantage_mean) ** 2 for value in advantages
        ) / len(advantages)
        key = f"{question_id}:{role}"
        stats[key] = {
            "question_id": question_id,
            "agent_type": role,
            "count": len(actions),
            "mean": mean,
            "variance": variance,
            "std": math.sqrt(variance),
            "advantage_mean": advantage_mean,
            "advantage_std": math.sqrt(advantage_variance),
        }
    return stats


def broadcast_advantage_to_tokens(advantages: Any, decision_token_mask: Any) -> Any:
    """Broadcast [B] decision advantages to masked token advantages [B, L]."""

    if advantages.ndim != 1 or decision_token_mask.ndim != 2:
        raise ValueError("advantages must be [B] and decision_token_mask must be [B, L]")
    if advantages.shape[0] != decision_token_mask.shape[0]:
        raise ValueError("batch dimensions do not match")
    return advantages.unsqueeze(-1) * decision_token_mask.to(dtype=advantages.dtype)
