from __future__ import annotations

from typing import Any


def masked_mean(values: Any, mask: Any) -> Any:
    if values.shape != mask.shape:
        raise ValueError(f"values and mask shapes differ: {values.shape} != {mask.shape}")
    float_mask = mask.to(dtype=values.dtype)
    denominator = float_mask.sum()
    if denominator.detach().item() <= 0:
        raise ValueError("masked_mean requires at least one valid token")
    return (values * float_mask).sum() / denominator


def compute_policy_ratio(current_log_probs: Any, old_log_probs: Any) -> Any:
    import torch

    if current_log_probs.shape != old_log_probs.shape:
        raise ValueError("current and old log-probability shapes must match")
    log_ratio = current_log_probs - old_log_probs.detach()
    if not bool(torch.isfinite(log_ratio).all().item()):
        raise FloatingPointError("non-finite policy log-ratio")
    return torch.exp(log_ratio)


def compute_clipped_policy_loss(
    *,
    current_log_probs: Any,
    old_log_probs: Any,
    token_advantages: Any,
    decision_token_mask: Any,
    clip_eps: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    if not 0.0 <= clip_eps < 1.0:
        raise ValueError("clip_eps must satisfy 0 <= clip_eps < 1")
    ratio = compute_policy_ratio(current_log_probs, old_log_probs)
    if token_advantages.shape != ratio.shape or decision_token_mask.shape != ratio.shape:
        raise ValueError("ratio, token advantages and decision mask must share [B, L]")
    unclipped = ratio * token_advantages
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    clipped = clipped_ratio * token_advantages
    loss = masked_mean(-torch.minimum(unclipped, clipped), decision_token_mask)
    return loss, {"ratio": ratio, "clipped_ratio": clipped_ratio}


def compute_kl_loss(
    *, current_log_probs: Any, reference_log_probs: Any, decision_token_mask: Any
) -> Any:
    """Positive sampled-token KL estimator used by TRL-style GRPO."""

    import torch

    if current_log_probs.shape != reference_log_probs.shape:
        raise ValueError("current and reference log-probability shapes must match")
    valid = decision_token_mask.to(dtype=torch.bool)
    # Mask before exponentiation: padded sentinel values must not create inf*0.
    log_ref_over_policy = (reference_log_probs.detach() - current_log_probs).masked_fill(
        ~valid, 0.0
    )
    per_token_kl = torch.exp(log_ref_over_policy) - log_ref_over_policy - 1.0
    return masked_mean(per_token_kl, decision_token_mask)


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

    if advantages.ndim != 1 or advantages.shape[0] != current_logprobs.shape[0]:
        raise ValueError("advantages must have shape [B]")
    token_advantages = advantages.to(current_logprobs.dtype).unsqueeze(-1) * action_mask.to(
        current_logprobs.dtype
    )
    policy_loss, details = compute_clipped_policy_loss(
        current_log_probs=current_logprobs,
        old_log_probs=old_logprobs,
        token_advantages=token_advantages,
        decision_token_mask=action_mask,
        clip_eps=clip_epsilon,
    )
    kl_loss = compute_kl_loss(
        current_log_probs=current_logprobs,
        reference_log_probs=ref_logprobs,
        decision_token_mask=action_mask,
    )
    total_loss = policy_loss + float(kl_beta) * kl_loss
    ratio = details["ratio"]
    clipped_ratio = details["clipped_ratio"]
    valid = action_mask.to(torch.bool)
    logratio = current_logprobs - old_logprobs.detach()
    valid_ratio = ratio[valid]
    valid_logratio = logratio[valid]
    metrics = {
        "loss": float(total_loss.detach().item()),
        "total_loss": float(total_loss.detach().item()),
        "policy_loss": float(policy_loss.detach().item()),
        "kl": float(kl_loss.detach().item()),
        "kl_loss": float(kl_loss.detach().item()),
        "approx_kl": float(((ratio[valid] - 1.0) - valid_logratio).mean().detach().item()),
        "clip_fraction": float((ratio[valid] != clipped_ratio[valid]).float().mean().detach().item()),
        "preupdate_logratio_mean": float(valid_logratio.mean().detach().item()),
        "preupdate_logratio_max_abs": float(valid_logratio.abs().max().detach().item()),
        "ratio_mean": float(valid_ratio.mean().detach().item()),
        "mean_ratio": float(valid_ratio.mean().detach().item()),
        "ratio_p95": float(torch.quantile(valid_ratio.float(), 0.95).detach().item()),
    }
    if not bool(torch.isfinite(total_loss).item()):
        raise FloatingPointError("non-finite GRPO loss")
    return total_loss, metrics
