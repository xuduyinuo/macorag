from __future__ import annotations

from collections import defaultdict
from typing import Any

from .mappo_types import Episode, MAPPOTransition


def compute_gae(episode: Episode, *, gamma: float, gae_lambda: float) -> None:
    """Compute GAE separately along each agent's turn sequence.

    This is turn-based MAPPO: an agent's next state is its next decision state,
    not the immediately following decision made by another role.
    """
    by_role: dict[Any, list[MAPPOTransition]] = defaultdict(list)
    for transition in episode.transitions:
        by_role[transition.role].append(transition)
    for transitions in by_role.values():
        gae = 0.0
        for index in range(len(transitions) - 1, -1, -1):
            item = transitions[index]
            has_next = index + 1 < len(transitions) and not item.done
            next_value = transitions[index + 1].old_value if has_next else item.next_value
            nonterminal = 1.0 if has_next else 0.0
            delta = item.reward + gamma * next_value * nonterminal - item.old_value
            gae = delta + gamma * gae_lambda * nonterminal * gae
            item.advantage = float(gae)
            item.return_ = float(gae + item.old_value)


def mappo_actor_loss(new_logprobs: Any, old_logprobs: Any, advantages: Any, mask: Any, clip_epsilon: float) -> tuple[Any, dict[str, Any]]:
    import torch
    valid = mask.to(dtype=new_logprobs.dtype)
    denominator = valid.sum().clamp_min(1.0)
    log_ratio = (new_logprobs - old_logprobs).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    expanded_advantage = advantages.unsqueeze(-1)
    unclipped = ratio * expanded_advantage
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * expanded_advantage
    loss = -(torch.minimum(unclipped, clipped) * valid).sum() / denominator
    approximate_kl = (((ratio - 1.0) - log_ratio) * valid).sum() / denominator
    clip_fraction = (((ratio - 1.0).abs() > clip_epsilon).to(valid.dtype) * valid).sum() / denominator
    return loss, {"approx_kl": approximate_kl.detach(), "clip_fraction": clip_fraction.detach()}


def clipped_value_loss(values: Any, old_values: Any, returns: Any, clip_epsilon: float) -> Any:
    import torch
    clipped = old_values + (values - old_values).clamp(-clip_epsilon, clip_epsilon)
    ordinary = (values - returns).pow(2)
    clipped_error = (clipped - returns).pow(2)
    return 0.5 * torch.maximum(ordinary, clipped_error).mean()


def normalize_advantages(transitions: list[MAPPOTransition], epsilon: float = 1e-8) -> None:
    import math
    if len(transitions) < 2:
        return
    mean = sum(item.advantage for item in transitions) / len(transitions)
    variance = sum((item.advantage - mean) ** 2 for item in transitions) / len(transitions)
    std = math.sqrt(variance)
    if std <= epsilon:
        return
    for item in transitions:
        item.advantage = (item.advantage - mean) / std


def normalize_advantages_by_role(
    transitions: list[MAPPOTransition], epsilon: float = 1e-8,
) -> None:
    """Normalize each role independently to reduce shared-actor interference.

    Singleton and tied role groups keep their original advantages: turning
    them into zeros would silently discard an otherwise valid learning signal.
    """
    by_role: dict[Any, list[MAPPOTransition]] = defaultdict(list)
    for transition in transitions:
        by_role[transition.role].append(transition)
    for role_transitions in by_role.values():
        normalize_advantages(role_transitions, epsilon=epsilon)
