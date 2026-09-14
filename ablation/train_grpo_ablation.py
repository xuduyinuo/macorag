#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

from rl_training import train_grpo_macorag as training


VARIANTS = {"full", "wo_multi_round_retrieval", "wo_fine_grained_credit", "wo_local_reward"}


def _trajectory_credit(
    rollouts: list[dict[str, Any]],
    *,
    global_weights: dict[str, float],
    epsilon: float = 1.0e-8,
    granularity: str = "role_round",
    degenerate_bucket_fallback_weight: float = 0.0,
) -> dict[str, dict[str, float | int]]:
    """Broadcast one normalized terminal-return advantage to every action."""
    del global_weights, granularity, degenerate_bucket_fallback_weight
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("Advantage epsilon must be a positive finite number.")
    returns = [float(rollout.get("terminal_reward", 0.0)) for rollout in rollouts]
    mean = sum(returns) / len(returns) if returns else 0.0
    variance = sum((value - mean) ** 2 for value in returns) / len(returns) if returns else 0.0
    std = math.sqrt(max(variance, 0.0))
    advantages = [(value - mean) / (std + epsilon) for value in returns] if std > epsilon else [0.0] * len(returns)
    action_count = 0
    for rollout, advantage in zip(rollouts, advantages):
        for action in rollout.get("actions", []):
            action.local_reward = 0.0
            action.terminal_reward = float(rollout.get("terminal_reward", 0.0))
            action.decision_return = action.terminal_reward
            action.primary_advantage = advantage
            action.fallback_advantage = 0.0
            action.advantage = advantage
            action_count += 1
    return {
        "trajectory": {
            "count": len(returns),
            "action_count": action_count,
            "mean": mean,
            "std": std,
            "min": min(returns) if returns else 0.0,
            "max": max(returns) if returns else 0.0,
            "advantage_mean": sum(advantages) / len(advantages) if advantages else 0.0,
            "advantage_std": (
                math.sqrt(sum((value - (sum(advantages) / len(advantages))) ** 2 for value in advantages) / len(advantages))
                if advantages else 0.0
            ),
        }
    }


def _without_local_rewards(original: Any) -> Any:
    def wrapped(*, rollout: dict[str, Any], sample: dict[str, Any], answer_local_reward_weight: float = 1.0) -> dict[str, Any]:
        result = original(
            rollout=rollout,
            sample=sample,
            answer_local_reward_weight=answer_local_reward_weight,
        )
        for item in result["action_rewards"]:
            item["local_reward"] = 0.0
            item["components"] = {key: 0.0 for key in item.get("components", {})}
        return result
    return wrapped


def _install_variant(variant: str) -> None:
    if variant == "wo_multi_round_retrieval":
        original_validation = training._validate_sft_prompt_contract

        def validate_shared_sft_except_rounds(args: Any) -> dict[str, Any]:
            metadata_path = Path(args.sft_adapter_path) / "prompt_contract.json"
            if not metadata_path.is_file():
                return original_validation(args)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            validation_args = copy.copy(args)
            validation_args.max_rounds = int(metadata.get("max_rounds", -1))
            return original_validation(validation_args)

        training._validate_sft_prompt_contract = validate_shared_sft_except_rounds
    elif variant == "wo_fine_grained_credit":
        training.assign_action_advantages = _trajectory_credit
    elif variant == "wo_local_reward":
        training.compute_action_rewards = _without_local_rewards(training.compute_action_rewards)
    original_payload = training._resolved_args_payload

    def payload(args: Any) -> dict[str, Any]:
        result = original_payload(args)
        result["ablation_variant"] = variant
        result["fine_grained_credit_enabled"] = variant != "wo_fine_grained_credit"
        result["local_reward_enabled"] = variant != "wo_local_reward"
        result["multi_round_retrieval_enabled"] = variant != "wo_multi_round_retrieval"
        result["sft_round_contract_exception"] = variant == "wo_multi_round_retrieval"
        return result

    training._resolved_args_payload = payload


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ablation-variant", required=True, choices=sorted(VARIANTS))
    known, remaining = parser.parse_known_args()
    _install_variant(known.ablation_variant)
    sys.argv = [sys.argv[0], *remaining]
    training.main()


if __name__ == "__main__":
    main()
