from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from .checkpoint import prune_checkpoints, restore_checkpoint, save_actor_export, save_checkpoint
from .data import epoch_order, sample_stratum
from .mappo import (
    clipped_value_loss,
    compute_gae,
    mappo_actor_loss,
    normalize_advantages,
    normalize_advantages_by_role,
)
from .protocol import ProtocolWindowMonitor, protocol_metrics


def _append(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def _chunks(values: list[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _episode_payload(episode: Any, *, validation_step: int | None = None) -> dict[str, Any]:
    payload = {
        "qid": episode.qid, "dataset": episode.dataset,
        "trajectory": episode.trajectory, "final_answer": episode.final_answer,
        "global_reward": episode.global_reward, "answer_f1": episode.answer_f1,
        "evidence_coverage": episode.evidence_coverage,
        "parse_errors": episode.parse_errors,
        "invalid_actions": [
            {
                "role": item.role.value, "round": item.round_index,
                "response": item.response, "parse_error": item.parse_error,
                "action_tokens": len(item.action_ids),
            }
            for item in episode.transitions if not item.valid
        ],
    }
    if validation_step is not None:
        payload["global_step"] = validation_step
    return payload


class MAPPOTrainer:
    def __init__(self, *, actor: Any, critic: Any, collector: Any, samples: list[Any],
                 validation_samples: list[Any], config: Any, run_dir: Path) -> None:
        import torch
        self.torch = torch
        self.actor = actor
        self.critic = critic
        self.collector = collector
        self.samples = samples
        self.validation_samples = validation_samples
        self.config = config
        self.run_dir = run_dir
        actor_parameters = [item for item in actor.model.parameters() if item.requires_grad]
        if not actor_parameters:
            raise RuntimeError("MAPPO actor has no trainable parameters")
        self.actor_optimizer = torch.optim.AdamW(
            actor_parameters, lr=config.actor_learning_rate, weight_decay=config.weight_decay,
        )
        self.critic_optimizer = torch.optim.AdamW(
            critic.parameters(), lr=config.critic_learning_rate, weight_decay=config.weight_decay,
        )
        self.global_step = 0
        self.start_epoch = 0
        self.sample_offset = 0
        self.best_validation_score = -math.inf
        self.best_validation_step = 0
        self.best_protocol_rank: tuple[float, ...] | None = None
        self.best_protocol_step = 0
        self.best_protocol_checkpoint_eligible = False
        self.baseline_validation_score: float | None = None
        self.bad_validation_count = 0
        self.last_validation_step = -1
        self.stopped_early = False
        self.protocol_monitor = ProtocolWindowMonitor(
            window_size=config.protocol_window_size,
            max_parse_failure_rate=config.max_protocol_parse_failure_rate,
        )
        if config.resume_from_checkpoint:
            restored = restore_checkpoint(
                Path(config.resume_from_checkpoint), critic=critic,
                actor_optimizer=self.actor_optimizer, critic_optimizer=self.critic_optimizer,
            )
            self.global_step = restored["global_step"]
            self.start_epoch = restored["epoch"]
            self.sample_offset = restored["sample_offset"]
            if hasattr(self.actor, "generation_counter"):
                self.actor.generation_counter = restored["generation_counter"]
        if hasattr(self.actor, "sync_generation"):
            self.actor.sync_generation(
                sync_root=self.run_dir / "vllm_sync",
                step=self.global_step,
                keep=self.config.vllm_sync_snapshots_to_keep,
            )

    def _validation_protocol(self, episodes: list[Any]) -> dict[str, Any]:
        overall = protocol_metrics(
            episodes,
            max_parse_failure_rate=self.config.max_protocol_parse_failure_rate,
            max_missing_answer_tag_rate=self.config.max_validation_missing_answer_tag_rate,
            min_final_compliance_rate=self.config.min_validation_final_compliance_rate,
        )
        by_dataset = {}
        for dataset in sorted({item.dataset for item in episodes}):
            by_dataset[dataset] = protocol_metrics(
                [item for item in episodes if item.dataset == dataset],
                max_parse_failure_rate=self.config.max_protocol_parse_failure_rate,
                max_missing_answer_tag_rate=self.config.max_validation_missing_answer_tag_rate,
                min_final_compliance_rate=self.config.min_validation_final_compliance_rate,
            )
        overall["checkpoint_eligible"] = bool(
            overall["checkpoint_eligible"]
            and all(item["checkpoint_eligible"] for item in by_dataset.values())
        )
        overall["protocol_by_dataset"] = by_dataset
        return overall

    @staticmethod
    def _quality_by_dataset(episodes: list[Any]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for dataset in sorted({item.dataset for item in episodes}):
            rows = [item for item in episodes if item.dataset == dataset]
            count = len(rows)
            result[dataset] = {
                "samples": count,
                "answer_f1_mean": sum(item.answer_f1 for item in rows) / count,
                "evidence_coverage_mean": (
                    sum(item.evidence_coverage for item in rows) / count
                ),
                "reward_mean": sum(item.global_reward for item in rows) / count,
            }
        return result

    @staticmethod
    def _quality_by_stratum(
        episodes: list[Any], samples: list[Any]
    ) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[Any]] = {}
        for episode, sample in zip(episodes, samples):
            key = f"{sample.dataset}:{sample_stratum(sample)}"
            grouped.setdefault(key, []).append(episode)
        return {
            key: {
                "samples": len(rows),
                "answer_f1_mean": sum(item.answer_f1 for item in rows) / len(rows),
                "evidence_coverage_mean": (
                    sum(item.evidence_coverage for item in rows) / len(rows)
                ),
                "final_answer_rate": sum(bool(item.final_answer) for item in rows) / len(rows),
            }
            for key, rows in sorted(grouped.items())
        }

    @staticmethod
    def _protocol_rank(protocol: dict[str, Any], answer_f1: float) -> tuple[float, ...]:
        rows = list(protocol.get("protocol_by_dataset", {}).values()) or [protocol]
        return (
            float(bool(protocol["checkpoint_eligible"])),
            min(float(item["final_compliance_rate"]) for item in rows),
            -max(float(item["parse_failure_rate"]) for item in rows),
            -max(float(item["missing_answer_tag_rate"]) for item in rows),
            float(answer_f1),
        )

    def _validate(self) -> dict[str, Any] | None:
        if not self.validation_samples:
            return None
        from tqdm.auto import tqdm

        old_temperature = self.actor.temperature
        self.actor.temperature = self.config.validation_temperature
        episodes = []
        try:
            with tqdm(
                total=len(self.validation_samples),
                desc=f"MAPPO validation step {self.global_step}",
                unit="sample",
                dynamic_ncols=True,
            ) as progress:
                for sample in self.validation_samples:
                    progress.set_postfix(
                        dataset=sample.dataset,
                        qid=sample.qid[:24],
                        refresh=True,
                    )
                    episode = self.collector.collect(sample)
                    episodes.append(episode)
                    _append(
                        self.run_dir / "validation_episodes.jsonl",
                        _episode_payload(episode, validation_step=self.global_step),
                    )
                    progress.update(1)
        finally:
            self.actor.temperature = old_temperature
        count = len(episodes)
        protocol = self._validation_protocol(episodes)
        quality_by_dataset = self._quality_by_dataset(episodes)
        reward = sum(item.global_reward for item in episodes) / count
        f1 = sum(item.answer_f1 for item in episodes) / count
        coverage = sum(item.evidence_coverage for item in episodes) / count
        # Equal-weight datasets prevent the easiest/largest validation subset
        # from dominating checkpoint selection. Evidence remains diagnostic;
        # protocol metrics are hard gates rather than additive score terms.
        score = sum(
            item["answer_f1_mean"] for item in quality_by_dataset.values()
        ) / len(quality_by_dataset)
        score_improved = score > self.best_validation_score + self.config.validation_min_delta
        improved = bool(protocol["checkpoint_eligible"] and score_improved)
        protocol_rank = self._protocol_rank(protocol, score)
        previous_protocol_rank = getattr(self, "best_protocol_rank", None)
        protocol_improved = previous_protocol_rank is None or protocol_rank > previous_protocol_rank
        if protocol_improved:
            self.best_protocol_rank = protocol_rank
            self.best_protocol_step = self.global_step
            self.best_protocol_checkpoint_eligible = bool(
                protocol["checkpoint_eligible"]
            )
        if improved:
            self.best_validation_score = score
            self.best_validation_step = self.global_step
            self.bad_validation_count = 0
        elif self.global_step > 0:
            self.bad_validation_count += 1
        result = {
            "global_step": self.global_step, "samples": count,
            "validation_kind": "baseline" if self.global_step == 0 else "periodic",
            "reward_mean": reward, "answer_f1_mean": f1,
            "answer_f1_macro": score,
            "evidence_coverage_mean": coverage,
            "quality_by_dataset": quality_by_dataset,
            "quality_by_stratum": self._quality_by_stratum(
                episodes, self.validation_samples,
            ),
            "validation_score": score, "answer_selection_score": score,
            "improved": improved, "answer_improved": improved,
            "protocol_improved": protocol_improved,
            "score_improved": score_improved,
            "best_validation_score": (
                self.best_validation_score
                if math.isfinite(self.best_validation_score) else None
            ),
            "best_validation_step": self.best_validation_step,
            "bad_validation_count": self.bad_validation_count,
            **protocol,
        }
        _append(self.run_dir / "validation_metrics.jsonl", result)
        if self.global_step == 0 and self.baseline_validation_score is None:
            self.baseline_validation_score = score
            (self.run_dir / "baseline_validation.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if improved:
            save_actor_export(
                self.run_dir / "best_answer_actor", actor=self.actor, metadata=result,
                config=self.config,
            )
            # Compatibility alias for existing evaluation launchers. It now
            # always means the protocol-eligible best macro answer-F1 actor.
            save_actor_export(
                self.run_dir / "best_actor", actor=self.actor, metadata=result,
                config=self.config,
            )
            (self.run_dir / "best_validation.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if protocol_improved:
            save_actor_export(
                self.run_dir / "best_protocol_actor", actor=self.actor,
                metadata=result, config=self.config,
            )
            (self.run_dir / "best_protocol_validation.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.last_validation_step = self.global_step
        if (
            self.config.early_stopping_patience > 0
            and self.bad_validation_count >= self.config.early_stopping_patience
            and self.global_step >= int(
                getattr(self.config, "early_stopping_min_steps", 0)
            )
        ):
            self.stopped_early = True
        return result

    def _validate_initial_policy(self) -> dict[str, Any] | None:
        if (
            self.validation_samples
            and self.global_step == 0
            and self.last_validation_step < 0
        ):
            return self._validate()
        return None

    def _align_old_logprobs(self, transitions: list[Any]) -> dict[str, float]:
        """Put PPO old/new log-probabilities on the learner's numerical path.

        vLLM remains the sampling engine. Its BF16 log-probabilities are useful
        for measuring inference/learner drift, but are not mixed directly with
        the 4-bit learner scores in the PPO importance ratio.
        """
        recompute_behavior = not (
            not self.config.use_vllm_generation
            or self.config.ppo_old_logprob_source == "vllm"
        )

        torch = self.torch
        sequences = [(item.prompt_ids, item.action_ids) for item in transitions]
        stats = {
            "behavior_logprob_abs_diff": 0.0,
            "behavior_approx_kl": 0.0,
            "old_logprobs_recomputed": 0.0,
            "reference_logprobs_computed": 0.0,
        }
        if recompute_behavior:
            local_logp, _ = self.actor.score_batch_no_grad(sequences)
            absolute_difference = torch.zeros((), dtype=torch.float32, device=local_logp.device)
            approximate_kl = torch.zeros((), dtype=torch.float32, device=local_logp.device)
            token_count = 0
            for row, item in enumerate(transitions):
                length = len(item.action_ids)
                server_logp = item.old_token_logprobs[:length].to(
                    device=local_logp.device, dtype=local_logp.dtype,
                )
                learner_logp = local_logp[row, :length]
                if server_logp.numel() != learner_logp.numel():
                    raise RuntimeError(
                        "vLLM/local actor log-probability length mismatch: "
                        f"{server_logp.numel()} != {learner_logp.numel()}"
                    )
                log_ratio = (learner_logp - server_logp).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()
                absolute_difference += (learner_logp - server_logp).abs().sum()
                approximate_kl += ((ratio - 1.0) - log_ratio).sum()
                token_count += length
                item.old_token_logprobs = learner_logp.detach().cpu()
            denominator = max(1, token_count)
            stats.update({
                "behavior_logprob_abs_diff": float((absolute_difference / denominator).cpu()),
                "behavior_approx_kl": float((approximate_kl / denominator).cpu()),
                "old_logprobs_recomputed": 1.0,
            })

        if float(getattr(self.config, "reference_kl_beta", 0.0)) > 0:
            reference_logp, _ = self.actor.score_reference_batch_no_grad(sequences)
            for row, item in enumerate(transitions):
                item.reference_token_logprobs = reference_logp[
                    row, :len(item.action_ids)
                ].detach().cpu()
            stats["reference_logprobs_computed"] = 1.0
        return stats

    def _entropy_coefficient(self) -> float:
        initial = float(getattr(self.config, "entropy_coef", 0.0))
        final = float(getattr(self.config, "entropy_final_coef", initial))
        start = int(getattr(self.config, "entropy_anneal_start_step", 0))
        end = int(getattr(self.config, "entropy_anneal_end_step", 0))
        if end <= start:
            return final if self.global_step >= end else initial
        progress = min(1.0, max(0.0, (self.global_step - start) / (end - start)))
        return initial + progress * (final - initial)

    def _update(self, transitions: list[Any]) -> dict[str, float]:
        torch = self.torch
        sequence_lengths = [len(item.prompt_ids) + len(item.action_ids) for item in transitions]
        prompt_lengths = [len(item.prompt_ids) for item in transitions]
        action_lengths = [len(item.action_ids) for item in transitions]
        preflight = {
            "event": "mappo_update_preflight",
            "global_step": self.global_step,
            "transitions": len(transitions),
            "sequence_tokens_max": max(sequence_lengths, default=0),
            "prompt_tokens_max": max(prompt_lengths, default=0),
            "action_tokens_max": max(action_lengths, default=0),
        }
        if torch.cuda.is_available():
            device = self.actor.device
            preflight.update({
                "cuda_allocated_gib": torch.cuda.memory_allocated(device) / (1024 ** 3),
                "cuda_reserved_gib": torch.cuda.memory_reserved(device) / (1024 ** 3),
            })
            torch.cuda.empty_cache()
        _append(self.run_dir / "update_preflight.jsonl", preflight)
        behavior_stats = self._align_old_logprobs(transitions)
        if self.config.normalize_advantages:
            if getattr(self.config, "advantage_normalization_scope", "role") == "role":
                normalize_advantages_by_role(transitions)
            else:
                normalize_advantages(transitions)
        aggregate: dict[str, list[float]] = {
            "actor_loss": [], "critic_loss": [], "entropy": [],
            "reference_kl": [],
            "approx_kl": [], "clip_fraction": [], "grad_norm_actor": [], "grad_norm_critic": [],
        }
        entropy_coefficient = self._entropy_coefficient()
        reference_kl_beta = float(getattr(self.config, "reference_kl_beta", 0.0))
        indices = list(range(len(transitions)))
        stop_for_kl = False
        policy_updates_applied = 0
        kl_rejected_updates = 0
        for _ in range(self.config.ppo_epochs):
            random.shuffle(indices)
            for batch_indices in _chunks(indices, self.config.minibatch_size):
                batch = [transitions[index] for index in batch_indices]
                sequences = [(item.prompt_ids, item.action_ids) for item in batch]
                new_logp, entropy = self.actor.score_batch(sequences)
                mask = torch.zeros_like(new_logp, dtype=torch.bool)
                old_logp = torch.zeros_like(new_logp)
                for row, item in enumerate(batch):
                    length = len(item.action_ids)
                    mask[row, :length] = True
                    old_logp[row, :length] = item.old_token_logprobs.to(new_logp.device)
                reference_kl = torch.zeros((), dtype=new_logp.dtype, device=new_logp.device)
                if reference_kl_beta > 0:
                    reference_logp = torch.zeros_like(new_logp)
                    for row, item in enumerate(batch):
                        if item.reference_token_logprobs is None:
                            raise RuntimeError("Missing SFT reference log-probabilities")
                        length = len(item.action_ids)
                        reference_logp[row, :length] = item.reference_token_logprobs.to(
                            device=new_logp.device, dtype=new_logp.dtype,
                        )
                    ref_log_ratio = (reference_logp - new_logp).clamp(-20.0, 20.0)
                    reference_kl = (
                        (ref_log_ratio.exp() - 1.0 - ref_log_ratio) * mask
                    ).sum() / mask.sum().clamp_min(1)
                advantages = torch.tensor([item.advantage for item in batch], dtype=torch.float32, device=new_logp.device)
                actor_loss, policy_stats = mappo_actor_loss(
                    new_logp, old_logp, advantages, mask, self.config.clip_epsilon,
                )
                entropy_mean = (entropy * mask).sum() / mask.sum().clamp_min(1)
                # Reject the candidate minibatch before mutating parameters.
                # The previous implementation checked only after optimizer.step,
                # so every KL violation still changed the policy irreversibly.
                if float(policy_stats["approx_kl"].detach().float().cpu()) > self.config.target_kl:
                    aggregate["actor_loss"].append(float(actor_loss.detach().float().cpu()))
                    aggregate["entropy"].append(float(entropy_mean.detach().float().cpu()))
                    aggregate["reference_kl"].append(float(reference_kl.detach().float().cpu()))
                    aggregate["approx_kl"].append(float(policy_stats["approx_kl"].detach().float().cpu()))
                    aggregate["clip_fraction"].append(float(policy_stats["clip_fraction"].detach().float().cpu()))
                    kl_rejected_updates += 1
                    stop_for_kl = True
                    break
                actor_objective = (
                    actor_loss
                    - entropy_coefficient * entropy_mean
                    + reference_kl_beta * reference_kl
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_objective.backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.actor.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                self.actor_optimizer.step()
                policy_updates_applied += 1

                values = self.critic([item.central_state for item in batch])
                old_values = torch.tensor([item.old_value for item in batch], dtype=torch.float32, device=values.device)
                returns = torch.tensor([item.return_ for item in batch], dtype=torch.float32, device=values.device)
                critic_loss = clipped_value_loss(values, old_values, returns, self.config.value_clip_epsilon)
                critic_objective = self.config.value_loss_coef * critic_loss
                self.critic_optimizer.zero_grad(set_to_none=True)
                critic_objective.backward()
                critic_norm = torch.nn.utils.clip_grad_norm_(list(self.critic.parameters()), self.config.max_grad_norm)
                self.critic_optimizer.step()

                values_to_add = {
                    "actor_loss": actor_loss, "critic_loss": critic_loss,
                    "entropy": entropy_mean, "approx_kl": policy_stats["approx_kl"],
                    "reference_kl": reference_kl,
                    "clip_fraction": policy_stats["clip_fraction"],
                    "grad_norm_actor": actor_norm, "grad_norm_critic": critic_norm,
                }
                for name, value in values_to_add.items():
                    number = float(value.detach().float().cpu())
                    if not math.isfinite(number):
                        raise FloatingPointError(f"Non-finite MAPPO metric {name}: {number}")
                    aggregate[name].append(number)
            if stop_for_kl:
                break
        return {name: sum(values) / max(1, len(values)) for name, values in aggregate.items()} | {
            "ppo_early_stop": float(stop_for_kl),
            "policy_updates_applied": float(policy_updates_applied),
            "kl_rejected_updates": float(kl_rejected_updates),
            "entropy_coefficient": entropy_coefficient,
            "reference_kl_beta": reference_kl_beta,
            **behavior_stats,
        }

    def train(self) -> None:
        from tqdm.auto import tqdm

        self._validate_initial_policy()
        total_epochs = max(1, math.ceil(self.config.num_train_epochs))
        metrics_path = self.run_dir / "train_metrics.jsonl"
        episodes_path = self.run_dir / "episodes.jsonl"
        for epoch in range(self.start_epoch, total_epochs):
            ordered = epoch_order(self.samples, seed=self.config.seed, epoch=epoch)
            start = self.sample_offset if epoch == self.start_epoch else 0
            with tqdm(
                total=len(ordered), initial=start,
                desc=f"MAPPO epoch {epoch + 1}/{total_epochs}",
                unit="sample", dynamic_ncols=True,
            ) as progress:
                for batch_start in range(start, len(ordered), self.config.rollout_batch_size):
                    if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                        return self._final_save(epoch, batch_start)
                    batch_samples = ordered[batch_start:batch_start + self.config.rollout_batch_size]
                    episodes = []
                    for sample_index, sample in enumerate(batch_samples, 1):
                        progress.set_postfix(
                            stage="rollout",
                            step=self.global_step,
                            batch=f"{sample_index}/{len(batch_samples)}",
                            qid=sample.qid[:24],
                            refresh=True,
                        )
                        episodes.append(self.collector.collect(sample))
                        progress.update(1)
                    transitions = [
                        transition
                        for episode in episodes
                        for transition in episode.transitions
                        if transition.action_ids
                    ]
                    if not transitions:
                        progress.set_postfix(stage="skip-empty", step=self.global_step)
                        continue
                    for episode in episodes:
                        compute_gae(episode, gamma=self.config.gamma, gae_lambda=self.config.gae_lambda)
                        _append(episodes_path, _episode_payload(episode))
                    protocol = self._validation_protocol(episodes)
                    protocol_status = self.protocol_monitor.add(episodes)
                    if protocol_status["should_warn"]:
                        _append(self.run_dir / "protocol_events.jsonl", {
                            "event": "protocol_warning", "global_step": self.global_step,
                            **protocol_status,
                        })
                    progress.set_postfix(
                        stage="ppo", step=self.global_step,
                        transitions=len(transitions), refresh=True,
                    )
                    update = self._update(transitions)
                    self.global_step += 1
                    if (
                        self.config.vllm_sync_after_update
                        and hasattr(self.actor, "sync_generation")
                    ):
                        progress.set_postfix(stage="vllm-sync", step=self.global_step, refresh=True)
                        self.actor.sync_generation(
                            sync_root=self.run_dir / "vllm_sync",
                            step=self.global_step,
                            keep=self.config.vllm_sync_snapshots_to_keep,
                        )
                        update["vllm_synced"] = 1.0
                        update["vllm_sync_step"] = float(self.global_step)
                    metric = {
                        "global_step": self.global_step, "epoch": epoch,
                        "samples_consumed": batch_start + len(batch_samples),
                        "episodes": len(episodes), "transitions": len(transitions),
                        "reward_mean": sum(x.global_reward for x in episodes) / len(episodes),
                        "answer_f1_mean": sum(x.answer_f1 for x in episodes) / len(episodes),
                        "evidence_coverage_mean": sum(x.evidence_coverage for x in episodes) / len(episodes),
                        "invalid_action_fraction": sum(not x.valid for x in transitions) / len(transitions),
                        "parse_failure_rate": protocol["parse_failure_rate"],
                        "missing_answer_tag_rate": protocol["missing_answer_tag_rate"],
                        "final_compliance_rate": protocol["final_compliance_rate"],
                        "protocol_window_parse_failure_rate": protocol_status["window_parse_failure_rate"],
                        **update,
                    }
                    _append(metrics_path, metric)
                    if self.global_step % self.config.logging_steps == 0:
                        progress.set_postfix(
                            stage="ready", step=self.global_step,
                            reward=f"{metric['reward_mean']:.3f}",
                            f1=f"{metric['answer_f1_mean']:.3f}",
                            invalid=f"{metric['invalid_action_fraction']:.1%}",
                            kl=f"{metric['approx_kl']:.4f}",
                            updates=int(metric["policy_updates_applied"]),
                            refresh=True,
                        )
                    if self.global_step % self.config.validation_steps == 0:
                        progress.set_postfix(stage="validation", step=self.global_step, refresh=True)
                        validation = self._validate()
                        if validation is not None:
                            progress.set_postfix(
                                stage="validated", step=self.global_step,
                                val_f1=f"{validation['answer_f1_mean']:.3f}",
                                val_parse=f"{validation['parse_failure_rate']:.1%}",
                                best=self.best_validation_step, refresh=True,
                            )
                    if self.global_step % self.config.save_steps == 0:
                        progress.set_postfix(stage="checkpoint", step=self.global_step, refresh=True)
                        self._save(epoch, batch_start + len(batch_samples))
                    if self.stopped_early:
                        return self._final_save(epoch, batch_start + len(batch_samples))
            self.sample_offset = 0
        self._final_save(total_epochs, 0)

    def _save(self, epoch: int, sample_offset: int) -> None:
        save_checkpoint(
            self.run_dir / f"checkpoint-{self.global_step}", actor=self.actor,
            critic=self.critic, actor_optimizer=self.actor_optimizer,
            critic_optimizer=self.critic_optimizer, global_step=self.global_step,
            epoch=epoch, sample_offset=sample_offset, config=self.config,
        )
        prune_checkpoints(self.run_dir, self.config.save_total_limit)

    def _final_save(self, epoch: int, sample_offset: int) -> None:
        if self.validation_samples and self.last_validation_step != self.global_step:
            self._validate()
        self._save(epoch, sample_offset)
        summary = {
            "algorithm": "mappo", "global_step": self.global_step, "complete": True,
            "stopped_early": self.stopped_early,
            "train_samples": len(self.samples),
            "validation_samples": len(self.validation_samples),
            "baseline_validation_score": self.baseline_validation_score,
            "best_validation_step": self.best_validation_step,
            "best_answer_actor": (
                str(self.run_dir / "best_answer_actor")
                if (self.run_dir / "best_answer_actor").is_dir() else None
            ),
            "best_answer_actor_checkpoint_eligible": bool(
                (self.run_dir / "best_answer_actor").is_dir()
            ),
            "best_protocol_step": getattr(self, "best_protocol_step", 0),
            "best_protocol_actor_checkpoint_eligible": bool(getattr(
                self, "best_protocol_checkpoint_eligible", False,
            )),
            "best_protocol_actor": (
                str(self.run_dir / "best_protocol_actor")
                if (self.run_dir / "best_protocol_actor").is_dir() else None
            ),
            "best_validation_score": (
                self.best_validation_score if math.isfinite(self.best_validation_score) else None
            ),
            "best_validation_gain_over_baseline": (
                self.best_validation_score - self.baseline_validation_score
                if self.baseline_validation_score is not None
                and math.isfinite(self.best_validation_score)
                else None
            ),
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
