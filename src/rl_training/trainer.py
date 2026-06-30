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
