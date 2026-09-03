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
    tokens_per_action = mask.sum(dim=-1).clamp_min(1.0)
    logratio = current_logprobs - old_logprobs
    ratio = torch.exp(logratio)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    expanded_advantages = advantages.to(dtype=current_logprobs.dtype).unsqueeze(-1)
    unclipped = ratio * expanded_advantages
    clipped = clipped_ratio * expanded_advantages
    policy_loss_per_action = -(
        (torch.minimum(unclipped, clipped) * mask).sum(dim=-1)
        / tokens_per_action
    )
    policy_loss = policy_loss_per_action.mean()

    kl_per_action = (
        (
            torch.exp(ref_logprobs - current_logprobs)
            - (ref_logprobs - current_logprobs)
            - 1.0
        )
        * mask
    ).sum(dim=-1) / tokens_per_action
    kl = kl_per_action.mean()
    loss = policy_loss + (kl_beta * kl)
    valid = action_mask.to(dtype=torch.bool)
    valid_logratio = logratio[valid]
    valid_ratio = ratio[valid]
    valid_clipped_ratio = clipped_ratio[valid]
    metrics = {
        "loss": float(loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl": float(kl.detach().item()),
        "clip_fraction": float(
            ((valid_ratio - valid_clipped_ratio).abs() > 1e-8)
            .to(torch.float32)
            .mean()
            .detach()
            .item()
        ),
        "preupdate_logratio_mean": float(valid_logratio.mean().detach().item()),
        "preupdate_logratio_max_abs": float(valid_logratio.abs().max().detach().item()),
        "ratio_mean": float(valid_ratio.mean().detach().item()),
        "ratio_p95": float(torch.quantile(valid_ratio.float(), 0.95).detach().item()),
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
    global_weights: dict[str, float],
    epsilon: float = 1.0e-8,
    granularity: str = "role_only",
    degenerate_bucket_fallback_weight: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Assign role-aware advantages with a conservative fallback for tied rounds.

    ``role_round`` remains the primary comparison.  If every return in one of
    those buckets is equal within ``epsilon``, normalize terminal returns only
    among candidates in that same role/round bucket and apply the configured
    fallback weight. A bucket with no terminal contrast stays exactly zero;
    this avoids inventing preferences by comparing different round states.
    """
    import math

    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Advantage epsilon must be a positive finite number.")
    if granularity not in {"role_only", "role_round"}:
        raise ValueError("Advantage granularity must be role_only or role_round.")
    if (
        not math.isfinite(degenerate_bucket_fallback_weight)
        or not 0.0 <= degenerate_bucket_fallback_weight <= 1.0
    ):
        raise ValueError("Degenerate-bucket fallback weight must be finite and in [0, 1].")
    weights = {role: float(weight) for role, weight in global_weights.items()}
    for role_name, weight in weights.items():
        if not math.isfinite(weight):
            raise ValueError(f"Global reward weight for {role_name} must be finite.")

    buckets: dict[str, list[tuple[dict[str, Any], Any]]] = {}
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
            global_weight = weights.get(role_name)
            if global_weight is None:
                raise ValueError(f"Missing global reward weight for {role_name}.")
            action.decision_return = action.local_reward + global_weight * terminal_reward
            bucket_name = (
                role_name
                if granularity == "role_only"
                else f"{role_name}@round={int(action.round_index)}"
            )
            buckets.setdefault(bucket_name, []).append((rollout, action))

    stats: dict[str, dict[str, float | int]] = {}
    for bucket_name, entries in buckets.items():
        actions = [action for _, action in entries]
        returns = [float(action.decision_return) for action in actions]
        mean = sum(returns) / len(returns)
        variance = sum((item - mean) ** 2 for item in returns) / len(returns)
        std = math.sqrt(max(variance, 0.0))
        primary_advantages = (
            [
                (action.decision_return - mean) / (std + epsilon)
                for action in actions
            ]
            if std > epsilon
            else [0.0] * len(actions)
        )
        for action, primary_advantage in zip(actions, primary_advantages):
            action.primary_advantage = primary_advantage
            action.fallback_advantage = 0.0
            action.advantage = primary_advantage
        if granularity == "role_round" and std <= epsilon:
            terminal_returns = [
                float(rollout.get("terminal_reward", 0.0))
                for rollout, _ in entries
            ]
            terminal_mean = sum(terminal_returns) / len(terminal_returns)
            terminal_variance = sum(
                (item - terminal_mean) ** 2 for item in terminal_returns
            ) / len(terminal_returns)
            terminal_std = math.sqrt(max(terminal_variance, 0.0))
            for (_, action), terminal_return in zip(entries, terminal_returns):
                action.fallback_advantage = (
                    float(degenerate_bucket_fallback_weight)
                    * (terminal_return - terminal_mean)
                    / (terminal_std + epsilon)
                    if terminal_std > epsilon
                    else 0.0
                )
                action.advantage = action.fallback_advantage
        advantages = [float(action.advantage) for action in actions]
        primary_advantages = [float(action.primary_advantage) for action in actions]
        fallback_advantages = [float(action.fallback_advantage) for action in actions]
        advantage_mean = sum(advantages) / len(advantages)
        advantage_variance = sum(
            (item - advantage_mean) ** 2 for item in advantages
        ) / len(advantages)
        primary_advantage_mean = sum(primary_advantages) / len(primary_advantages)
        fallback_advantage_mean = sum(fallback_advantages) / len(fallback_advantages)
        stats[bucket_name] = {
            "count": len(returns),
            "mean": mean,
            "std": std,
            "min": min(returns),
            "max": max(returns),
            "advantage_mean": advantage_mean,
            "advantage_std": math.sqrt(max(advantage_variance, 0.0)),
            "advantage_min": min(advantages),
            "advantage_max": max(advantages),
            "primary_advantage_mean": primary_advantage_mean,
            "primary_advantage_std": math.sqrt(
                sum(
                    (item - primary_advantage_mean) ** 2
                    for item in primary_advantages
                )
                / len(primary_advantages)
            ),
            "fallback_advantage_mean": fallback_advantage_mean,
            "fallback_advantage_std": math.sqrt(
                sum(
                    (item - fallback_advantage_mean) ** 2
                    for item in fallback_advantages
                )
                / len(fallback_advantages)
            ),
        }
    return stats
