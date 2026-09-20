from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any

from .mappo_types import Episode, MAPPOTransition


def gradient_conflict_metrics(
    role_gradients: dict[str, list[Any | None]],
    *,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Summarize directional interference among actual weighted role gradients.

    The tensors may live on CPU or GPU. Missing gradients are treated as zeros,
    while a role with no gradients at all is rejected so diagnostics cannot
    silently report a harmless-looking zero vector.
    """

    import itertools
    import torch

    roles = sorted(role_gradients)
    if len(roles) < 2:
        raise ValueError("gradient conflict diagnostics require at least two roles")
    width = len(role_gradients[roles[0]])
    if any(len(role_gradients[role]) != width for role in roles):
        raise ValueError("all role gradient lists must align with the same parameters")

    norm_sq: dict[str, float] = {}
    for role in roles:
        present = [item for item in role_gradients[role] if item is not None]
        if not present:
            raise ValueError(f"role {role!r} produced no gradients")
        norm_sq[role] = sum(float(item.float().pow(2).sum().item()) for item in present)

    pairwise_cosine: dict[str, float] = {}
    pairwise_dot: dict[str, float] = {}
    conflicts = 0
    for left, right in itertools.combinations(roles, 2):
        dot = 0.0
        for left_grad, right_grad in zip(role_gradients[left], role_gradients[right]):
            if left_grad is not None and right_grad is not None:
                dot += float((left_grad.float() * right_grad.float()).sum().item())
        denominator = max(epsilon, math.sqrt(norm_sq[left] * norm_sq[right]))
        cosine = dot / denominator
        key = f"{left}__{right}"
        pairwise_dot[key] = dot
        pairwise_cosine[key] = cosine
        conflicts += int(cosine < 0.0)

    combined_norm_sq = 0.0
    role_combined_dot = {role: 0.0 for role in roles}
    for index in range(width):
        available = [
            role_gradients[role][index]
            for role in roles
            if role_gradients[role][index] is not None
        ]
        if not available:
            continue
        combined = torch.stack([item.float() for item in available]).sum(dim=0)
        combined_norm_sq += float(combined.pow(2).sum().item())
        for role in roles:
            gradient = role_gradients[role][index]
            if gradient is not None:
                role_combined_dot[role] += float(
                    (gradient.float() * combined).sum().item()
                )

    combined_norm = math.sqrt(combined_norm_sq)
    role_norms = {role: math.sqrt(value) for role, value in norm_sq.items()}
    norm_sum = sum(role_norms.values())
    effective_alignment = {
        role: role_combined_dot[role]
        / max(epsilon, role_norms[role] * combined_norm)
        for role in roles
    }
    pair_count = len(pairwise_cosine)
    return {
        "role_gradient_norms": role_norms,
        "pairwise_dot": pairwise_dot,
        "pairwise_cosine": pairwise_cosine,
        "conflicting_pairs": conflicts,
        "pair_count": pair_count,
        "conflict_rate": conflicts / max(1, pair_count),
        "combined_gradient_norm": combined_norm,
        "gradient_cancellation_ratio": 1.0 - combined_norm / max(epsilon, norm_sum),
        "role_to_combined_cosine": effective_alignment,
    }


@dataclass
class AdaptiveReferenceKLController:
    """Stateful multiplicative controller for the SFT-reference KL penalty.

    The controller consumes a token-weighted KL once per rollout update and
    changes the coefficient used by the *next* update.  A deadband avoids
    reacting to ordinary minibatch noise, while the EMA and persisted state
    keep resume behavior continuous.
    """

    beta: float
    target: float
    beta_min: float
    beta_max: float
    ema_decay: float = 0.9
    deadband_low: float = 0.8
    deadband_high: float = 1.2
    controller_rate: float = 0.1
    emergency_threshold: float = 0.12
    emergency_patience: int = 5
    mode: str = "adaptive"
    ema_kl: float | None = None
    emergency_count: int = 0

    def update(self, observed_kl: float) -> dict[str, float]:
        observed = max(0.0, float(observed_kl))
        previous_beta = float(self.beta)
        self.ema_kl = (
            observed
            if self.ema_kl is None
            else self.ema_decay * self.ema_kl + (1.0 - self.ema_decay) * observed
        )
        controller_error = 0.0
        if self.mode == "adaptive" and self.target > 0.0:
            ratio = self.ema_kl / self.target
            if ratio < self.deadband_low or ratio > self.deadband_high:
                # Capping the log-space error prevents one noisy rollout from
                # changing beta abruptly, without hiding sustained drift.
                controller_error = max(-0.2, min(0.2, ratio - 1.0))
                self.beta = max(
                    self.beta_min,
                    min(self.beta_max, self.beta * math.exp(self.controller_rate * controller_error)),
                )
        # EMA drives the smooth beta controller, but a stale EMA alone must not
        # count as another consecutive emergency. Require the current rollout
        # and the EMA to exceed the hard threshold.
        if (
            self.emergency_threshold > 0.0
            and observed > self.emergency_threshold
            and self.ema_kl > self.emergency_threshold
        ):
            self.emergency_count += 1
        else:
            self.emergency_count = 0
        emergency = self.emergency_count >= self.emergency_patience
        reported_emergency_count = self.emergency_count
        if emergency:
            # Re-arm the detector so persistent drift forces another check
            # after the next full patience window instead of warning only once.
            self.emergency_count = 0
        return {
            "reference_kl_observed": observed,
            "reference_kl_ema": float(self.ema_kl),
            "reference_kl_beta": previous_beta,
            "reference_kl_beta_next": float(self.beta),
            "reference_kl_controller_error": controller_error,
            "reference_kl_emergency_count": float(reported_emergency_count),
            "reference_kl_emergency_triggered": float(emergency),
        }

    def boost_beta(self, multiplier: float) -> float:
        """Apply a bounded emergency boost used by the recovery window."""
        self.beta = max(
            self.beta_min,
            min(self.beta_max, self.beta * max(1.0, float(multiplier))),
        )
        return float(self.beta)

    def state_dict(self) -> dict[str, Any]:
        return {
            "beta": self.beta,
            "ema_kl": self.ema_kl,
            "emergency_count": self.emergency_count,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if not state:
            return
        self.beta = max(self.beta_min, min(self.beta_max, float(state["beta"])))
        ema = state.get("ema_kl")
        self.ema_kl = None if ema is None else max(0.0, float(ema))
        self.emergency_count = max(0, int(state.get("emergency_count", 0)))


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
            item.raw_advantage = float(gae)
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
