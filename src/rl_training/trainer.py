from __future__ import annotations

from typing import Any


def compute_grpo_loss(
    *,
    current_logprobs: Any,
    old_logprobs: Any,
    ref_logprobs: Any,
    action_mask: Any,
    advantages: Any,
    clip_epsilon: float,
    kl_beta: float,
) -> tuple[Any, dict[str, float]]:
    import torch

    mask = action_mask.to(dtype=current_logprobs.dtype)
    token_count = mask.sum().clamp_min(1.0)
    ratio = torch.exp(current_logprobs - old_logprobs)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    expanded_advantages = advantages.to(dtype=current_logprobs.dtype).unsqueeze(-1)
    unclipped = ratio * expanded_advantages
    clipped = clipped_ratio * expanded_advantages
    policy_loss = -(torch.minimum(unclipped, clipped) * mask).sum() / token_count

    kl = ((torch.exp(ref_logprobs - current_logprobs) - (ref_logprobs - current_logprobs) - 1.0) * mask).sum()
    kl = kl / token_count
    loss = policy_loss + (kl_beta * kl)
    metrics = {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl": float(kl.detach().item()),
        "clip_fraction": float(((ratio - clipped_ratio).abs() > 1e-8).to(torch.float32).mean().detach().item()),
    }
    return loss, metrics


def normalize_group_advantages(rewards: list[float]) -> list[float]:
    if not rewards:
        return []
    if len(rewards) == 1:
        return [0.0]
    import math

    mean = sum(rewards) / len(rewards)
    variance = sum((item - mean) ** 2 for item in rewards) / len(rewards)
    std = math.sqrt(max(variance, 1e-12))
    return [(item - mean) / std for item in rewards]


def assign_action_advantages(
    rollouts: list[dict[str, Any]],
    *,
    local_weights: dict[str, float],
) -> None:
    buckets: dict[tuple[str, int], list[Any]] = {}
    for rollout in rollouts:
        reward_by_key = {
            (str(item["role"]), int(item["round_index"])): float(item["local_reward"])
            for item in rollout.get("action_rewards", [])
        }
        terminal_reward = float(rollout.get("terminal_reward", 0.0))
        for action in rollout.get("actions", []):
            role_name = getattr(action.role, "value", str(action.role))
            key = (role_name, int(action.round_index))
            action.local_reward = reward_by_key.get(key, 0.0)
            action.terminal_reward = terminal_reward
            buckets.setdefault(key, []).append(action)

    for (role_name, _), actions in buckets.items():
        local_weight = float(local_weights.get(role_name, 0.5))
        if not 0.0 <= local_weight <= 1.0:
            raise ValueError(f"Local credit weight for {role_name} must be between 0 and 1.")
        local_advantages = normalize_group_advantages([action.local_reward for action in actions])
        terminal_advantages = normalize_group_advantages([action.terminal_reward for action in actions])
        for action, local_advantage, terminal_advantage in zip(
            actions,
            local_advantages,
            terminal_advantages,
        ):
            action.advantage = (
                local_weight * local_advantage
                + (1.0 - local_weight) * terminal_advantage
            )
