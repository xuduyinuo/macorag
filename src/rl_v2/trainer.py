from __future__ import annotations

import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .checkpoint import prune_checkpoints, restore_checkpoint, save_actor_export, save_checkpoint
from .data import evenly_spaced_steps, sample_stratum, stratified_epoch_order
from .mappo import (
    AdaptiveReferenceKLController,
    clipped_value_loss,
    compute_gae,
    mappo_actor_loss,
    gradient_conflict_metrics,
    normalize_advantages,
    normalize_advantages_by_role,
)
from .mappo_types import AgentRole
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
        "evidence_duplicates_filtered": episode.evidence_duplicates_filtered,
        "parse_errors": episode.parse_errors,
        "timing": episode.timing,
        "invalid_actions": [
            {
                "role": item.role.value, "round": item.round_index,
                "response": item.response, "parse_error": item.parse_error,
                "action_tokens": len(item.action_ids),
            }
            for item in episode.transitions if not item.valid
        ],
        "format_recoveries": [
            {
                "role": item.role.value,
                "round": item.round_index,
                "method": item.format_recovery,
            }
            for item in episode.transitions if item.format_recovery is not None
        ],
        "tokenization_recoveries": [
            {
                "role": item.role.value,
                "round": item.round_index,
                "method": item.tokenization_recovery,
            }
            for item in episode.transitions if item.tokenization_recovery is not None
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
        self.best_observed_validation_score = -math.inf
        self.best_early_stopping_score = -math.inf
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
        base_controller = {
            "beta": float(getattr(config, "reference_kl_beta", 0.0)),
            "target": float(getattr(config, "reference_kl_target", 0.03)),
            "beta_min": float(getattr(config, "reference_kl_beta_min", 0.0)),
            "beta_max": float(getattr(config, "reference_kl_beta_max", 0.2)),
            "ema_decay": float(getattr(config, "reference_kl_ema_decay", 0.9)),
            "deadband_low": float(getattr(config, "reference_kl_deadband_low", 0.8)),
            "deadband_high": float(getattr(config, "reference_kl_deadband_high", 1.2)),
            "controller_rate": float(getattr(config, "reference_kl_controller_rate", 0.1)),
            "emergency_threshold": float(getattr(config, "reference_kl_emergency_threshold", 0.12)),
            "emergency_patience": int(getattr(config, "reference_kl_emergency_patience", 5)),
            "mode": str(getattr(config, "reference_kl_mode", "fixed")),
        }
        overrides = getattr(config, "reference_kl_role_overrides", {}) or {}
        self.reference_kl_controllers = {}
        self.reference_kl_recovery_steps = {role: 0 for role in AgentRole}
        for role in AgentRole:
            values = {**base_controller, **dict(overrides.get(role.value, {}))}
            self.reference_kl_controllers[role] = AdaptiveReferenceKLController(**values)
        if config.resume_from_checkpoint:
            restored = restore_checkpoint(
                Path(config.resume_from_checkpoint), critic=critic,
                actor_optimizer=self.actor_optimizer, critic_optimizer=self.critic_optimizer,
            )
            self.global_step = restored["global_step"]
            self.start_epoch = restored["epoch"]
            self.sample_offset = restored["sample_offset"]
            restored_controllers = restored.get("reference_kl_controllers") or {}
            legacy_controller = restored.get("reference_kl_controller", {})
            for role, controller in self.reference_kl_controllers.items():
                controller.load_state_dict(
                    restored_controllers.get(role.value, legacy_controller),
                )
            for role in AgentRole:
                self.reference_kl_recovery_steps[role] = max(0, int(
                    (restored.get("reference_kl_recovery_steps") or {}).get(
                        role.value, 0,
                    )
                ))
            restored_best_observed = restored.get("best_observed_validation_score")
            if restored_best_observed is not None:
                self.best_observed_validation_score = float(restored_best_observed)
            restored_best = restored.get("best_validation_score")
            if restored_best is not None:
                self.best_validation_score = float(restored_best)
                self.best_validation_step = int(restored["best_validation_step"])
            restored_baseline = restored.get("baseline_validation_score")
            if restored_baseline is not None:
                self.baseline_validation_score = float(restored_baseline)
            restored_early_score = restored.get("best_early_stopping_score")
            if restored_early_score is not None:
                self.best_early_stopping_score = float(restored_early_score)
            self.bad_validation_count = int(restored.get("bad_validation_count", 0))
            if hasattr(self.actor, "generation_counter"):
                self.actor.generation_counter = restored["generation_counter"]
        if hasattr(self.actor, "sync_generation"):
            self.actor.sync_generation(
                sync_root=self.run_dir / "vllm_sync",
                step=self.global_step,
                keep=self.config.vllm_sync_snapshots_to_keep,
            )

    def _collect_episodes(
        self, samples: list[Any], *, workers: int, on_complete: Any | None = None,
    ) -> list[Any]:
        """Collect independent trajectories concurrently, preserving input order.

        Each trajectory remains causally sequential. Concurrency exists only
        across samples, which lets vLLM continuously batch generation requests
        and lets the retrieval worker coalesce independent queries.
        """
        if not samples:
            return []
        worker_count = min(max(1, int(workers)), len(samples))
        if worker_count == 1:
            episodes = []
            for index, sample in enumerate(samples):
                episode = self.collector.collect(sample)
                episodes.append(episode)
                if on_complete is not None:
                    on_complete(index, sample, episode)
            return episodes

        episodes: list[Any | None] = [None] * len(samples)
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="rl-v2-rollout",
        ) as executor:
            pending = {
                executor.submit(self.collector.collect, sample): (index, sample)
                for index, sample in enumerate(samples)
            }
            for future in as_completed(pending):
                index, sample = pending[future]
                episode = future.result()
                episodes[index] = episode
                if on_complete is not None:
                    on_complete(index, sample, episode)
        if any(item is None for item in episodes):
            raise RuntimeError("parallel rollout collection returned an incomplete batch")
        return list(episodes)

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
        # The hard gate applies to all validation samples together. Applying a
        # 1-2% threshold independently to a 50-sample dataset would silently
        # turn it into a zero-error requirement.
        overall["checkpoint_eligibility_scope"] = "overall"
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
    def _validation_selection_score(
        quality_by_dataset: dict[str, dict[str, float]],
        protocol: dict[str, Any],
        config: Any,
    ) -> dict[str, float]:
        """Compute the equal-dataset checkpoint-selection objective."""
        dataset_quality = list(quality_by_dataset.values())
        dataset_protocol = list(
            protocol.get("protocol_by_dataset", {}).values()
        ) or [protocol]
        answer_f1 = sum(
            item["answer_f1_mean"] for item in dataset_quality
        ) / len(dataset_quality)
        evidence_coverage = sum(
            item["evidence_coverage_mean"] for item in dataset_quality
        ) / len(dataset_quality)
        format_compliance = sum(
            item["final_compliance_rate"] for item in dataset_protocol
        ) / len(dataset_protocol)
        parse_penalty = float(protocol["parse_failure_rate"])
        score = (
            float(config.validation_score_answer_weight) * answer_f1
            + float(config.validation_score_evidence_weight) * evidence_coverage
            + float(config.validation_score_format_weight) * format_compliance
            - float(config.validation_score_parse_penalty) * parse_penalty
        )
        return {
            "answer_f1_macro": answer_f1,
            "evidence_coverage_macro": evidence_coverage,
            "format_compliance_macro": format_compliance,
            "parse_failure_penalty": parse_penalty,
            "validation_score": score,
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

    def _validate(self, *, kind: str = "scheduled") -> dict[str, Any] | None:
        if not self.validation_samples:
            return None
        if kind not in {"baseline", "scheduled", "kl_emergency", "final"}:
            raise ValueError(f"Unknown validation kind: {kind}")
        validation_kind = "baseline" if self.global_step == 0 else kind
        early_stopping_enabled = self.config.early_stopping_patience > 0
        affects_early_stopping = bool(
            early_stopping_enabled
            and validation_kind in {"baseline", "scheduled"}
        )
        from tqdm.auto import tqdm

        validation_started = time.perf_counter()
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
                def validation_complete(index: int, sample: Any, episode: Any) -> None:
                    progress.set_postfix(
                        dataset=sample.dataset,
                        qid=sample.qid[:24],
                        workers=min(
                            self.config.validation_rollout_workers,
                            len(self.validation_samples),
                        ),
                        refresh=True,
                    )
                    _append(
                        self.run_dir / "validation_episodes.jsonl",
                        _episode_payload(episode, validation_step=self.global_step),
                    )
                    progress.update(1)
                episodes = self._collect_episodes(
                    self.validation_samples,
                    workers=self.config.validation_rollout_workers,
                    on_complete=validation_complete,
                )
        finally:
            self.actor.temperature = old_temperature
        count = len(episodes)
        validation_elapsed = time.perf_counter() - validation_started
        timing_totals: dict[str, float] = {}
        for episode in episodes:
            for name, value in episode.timing.items():
                timing_totals[name] = timing_totals.get(name, 0.0) + float(value)
        protocol = self._validation_protocol(episodes)
        quality_by_dataset = self._quality_by_dataset(episodes)
        reward = sum(item.global_reward for item in episodes) / count
        f1 = sum(item.answer_f1 for item in episodes) / count
        coverage = sum(item.evidence_coverage for item in episodes) / count
        # Equal-weight datasets prevent the easiest/largest validation subset
        # from dominating any component of checkpoint selection.
        selection = self._validation_selection_score(
            quality_by_dataset, protocol, self.config,
        )
        score = selection["validation_score"]
        previous_best_observed = float(getattr(
            self, "best_observed_validation_score", -math.inf,
        ))
        quality_improved = (
            score > previous_best_observed + self.config.validation_min_delta
        )
        if quality_improved:
            self.best_observed_validation_score = score
        previous_early_score = float(getattr(
            self, "best_early_stopping_score", -math.inf,
        ))
        early_stopping_improved: bool | None = None
        if not early_stopping_enabled:
            early_stopping_improved = None
            self.bad_validation_count = 0
        elif validation_kind == "baseline":
            self.best_early_stopping_score = score
            self.bad_validation_count = 0
            early_stopping_improved = True
        elif validation_kind == "scheduled":
            early_stopping_improved = (
                score > previous_early_score + self.config.validation_min_delta
            )
            if early_stopping_improved:
                self.best_early_stopping_score = score
                self.bad_validation_count = 0
            else:
                self.bad_validation_count += 1
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
        result = {
            "global_step": self.global_step, "samples": count,
            "validation_rollout_workers": min(
                self.config.validation_rollout_workers, count,
            ),
            "validation_elapsed_seconds": validation_elapsed,
            "validation_seconds_per_sample": validation_elapsed / count,
            "validation_target_seconds_per_sample": (
                self.config.validation_target_seconds_per_sample
            ),
            "validation_target_total_minutes": (
                self.config.validation_target_total_minutes
            ),
            "validation_seconds_per_sample_target_met": (
                validation_elapsed / count
                <= self.config.validation_target_seconds_per_sample
            ),
            "validation_total_time_target_met": (
                validation_elapsed
                <= self.config.validation_target_total_minutes * 60.0
            ),
            "rollout_timing_totals": timing_totals,
            "rollout_timing_per_sample": {
                name: value / count for name, value in timing_totals.items()
            },
            "retrieval_batch_stats": self.collector.retrieval.batch_stats(),
            "validation_kind": validation_kind,
            "early_stopping_enabled": early_stopping_enabled,
            "affects_early_stopping": affects_early_stopping,
            "early_stopping_improved": early_stopping_improved,
            "best_early_stopping_score": (
                self.best_early_stopping_score
                if math.isfinite(self.best_early_stopping_score) else None
            ),
            "reward_mean": reward, "answer_f1_mean": f1,
            "answer_f1_macro": selection["answer_f1_macro"],
            "evidence_coverage_mean": coverage,
            "evidence_coverage_macro": selection["evidence_coverage_macro"],
            "format_compliance_macro": selection["format_compliance_macro"],
            "validation_score_components": {
                "answer_f1": selection["answer_f1_macro"],
                "evidence_coverage": selection["evidence_coverage_macro"],
                "format_compliance": selection["format_compliance_macro"],
                "parse_failure_penalty": selection["parse_failure_penalty"],
            },
            "quality_by_dataset": quality_by_dataset,
            "quality_by_stratum": self._quality_by_stratum(
                episodes, self.validation_samples,
            ),
            "validation_score": score,
            "answer_selection_score": selection["answer_f1_macro"],
            "improved": improved, "answer_improved": None,
            "protocol_improved": protocol_improved,
            "score_improved": score_improved,
            "quality_improved": quality_improved,
            "best_observed_validation_score": self.best_observed_validation_score,
            "best_validation_score": (
                self.best_validation_score
                if math.isfinite(self.best_validation_score) else None
            ),
            "best_validation_step": self.best_validation_step,
            "best_checkpoint": (
                str(self.run_dir / f"checkpoint-{self.best_validation_step}")
                if (
                    self.best_validation_step > 0
                    and (
                        self.run_dir
                        / f"checkpoint-{self.best_validation_step}"
                        / "COMPLETE"
                    ).is_file()
                ) else None
            ),
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
                self.run_dir / "best_composite_actor", actor=self.actor,
                metadata=result, config=self.config,
            )
            save_actor_export(
                self.run_dir / "best_answer_actor", actor=self.actor, metadata=result,
                config=self.config,
            )
            # Compatibility alias for existing evaluation launchers. It now
            # means the protocol-eligible best composite-score actor.
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
            early_stopping_enabled
            and validation_kind == "scheduled"
            and self.config.early_stopping_patience > 0
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
            return self._validate(kind="baseline")
        return None

    def _learner_memory_preflight(self) -> dict[str, Any]:
        """Exercise the worst configured learner sequence before validation.

        Baseline validation is expensive and uses vLLM only. This trainable
        forward/backward catches an unsafe prompt budget before spending tens
        of minutes producing rollouts that cannot be optimized.
        """
        enabled = bool(getattr(self.config, "learner_memory_preflight", True))
        report: dict[str, Any] = {
            "enabled": enabled,
            "max_prompt_length": int(self.config.max_prompt_length),
            "status": "skipped",
        }
        path = self.run_dir / "learner_memory_preflight.json"
        if not enabled or not self.torch.cuda.is_available():
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            return report

        action_length = max(
            int(self.config.max_completion_length),
            int(self.config.answer_max_completion_length),
            int(self.config.evidence_max_completion_length),
        )
        synthetic_state = (
            "Question: learner memory preflight?\n"
            "Accumulated selected evidence: "
            + ("memory-budget-token " * (int(self.config.max_prompt_length) * 3))
        )
        prompt_ids = self.actor.encode_prompt(
            AgentRole.ANSWER, synthetic_state,
        )
        token_id = int(
            self.actor.tokenizer.eos_token_id
            if self.actor.tokenizer.eos_token_id is not None
            else self.actor.tokenizer.pad_token_id
        )
        action_ids = [token_id] * action_length
        device = self.actor.device
        started = time.perf_counter()
        cpu_rng_state = self.torch.get_rng_state()
        cuda_rng_state = self.torch.cuda.get_rng_state(device)
        self.actor_optimizer.zero_grad(set_to_none=True)
        self.torch.cuda.empty_cache()
        self.torch.cuda.reset_peak_memory_stats(device)
        try:
            logp, entropy, _ = self.actor.score_batch_detailed([
                (prompt_ids, action_ids),
            ])
            (-(logp.mean()) - 0.001 * entropy.mean()).backward()
            self.torch.cuda.synchronize(device)
            report.update({
                "status": "ok",
                "prompt_tokens": len(prompt_ids),
                "action_tokens": len(action_ids),
                "sequence_tokens": len(prompt_ids) + len(action_ids),
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gib": (
                    self.torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                ),
                "peak_reserved_gib": (
                    self.torch.cuda.max_memory_reserved(device) / (1024 ** 3)
                ),
            })
        except self.torch.OutOfMemoryError as exc:
            report.update({
                "status": "oom",
                "prompt_tokens": len(prompt_ids),
                "action_tokens": len(action_ids),
                "sequence_tokens": len(prompt_ids) + len(action_ids),
                "elapsed_seconds": time.perf_counter() - started,
            })
            path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raise RuntimeError(
                "Learner memory preflight OOM before validation: reduce "
                "max_prompt_length or completion budgets"
            ) from exc
        finally:
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.torch.cuda.empty_cache()
            self.torch.set_rng_state(cpu_rng_state)
            self.torch.cuda.set_rng_state(cuda_rng_state, device)
        path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report

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
        constraints = [item.token_constraints for item in transitions]
        has_constraints = any(item is not None for item in constraints)
        stats = {
            "behavior_logprob_abs_diff": 0.0,
            "behavior_approx_kl": 0.0,
            "old_logprobs_recomputed": 0.0,
            "reference_logprobs_computed": 0.0,
            "behavior_tokenization_recoveries": float(sum(
                item.tokenization_recovery is not None for item in transitions
            )),
        }
        if recompute_behavior:
            local_logp, _ = self.actor.score_batch_no_grad(
                sequences, token_constraints=constraints,
            ) if has_constraints else self.actor.score_batch_no_grad(sequences)
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
                # A canonicalized action no longer has the vLLM token path's
                # log-probabilities. It is still valid for PPO after this local
                # recomputation, but must not contaminate the server/learner
                # tokenization-drift diagnostic.
                if item.tokenization_recovery is None:
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

        if self._reference_kl_beta() > 0:
            if has_constraints:
                reference_logp, _, raw_reference_logp = (
                    self.actor.score_reference_batch_no_grad(
                        sequences, token_constraints=constraints, return_raw=True,
                    )
                )
            else:
                reference_logp, _ = self.actor.score_reference_batch_no_grad(sequences)
                raw_reference_logp = reference_logp
            for row, item in enumerate(transitions):
                item.reference_token_logprobs = reference_logp[
                    row, :len(item.action_ids)
                ].detach().cpu()
                item.reference_raw_token_logprobs = raw_reference_logp[
                    row, :len(item.action_ids)
                ].detach().cpu()
            stats["reference_logprobs_computed"] = 1.0
        return stats

    def _reference_kl_beta(self, role: AgentRole | None = None) -> float:
        controllers = getattr(self, "reference_kl_controllers", None)
        if controllers:
            if role is not None:
                return float(controllers[role].beta)
            return max(float(item.beta) for item in controllers.values())
        return float(getattr(self.config, "reference_kl_beta", 0.0))

    def _actor_minibatch_groups(self, transitions: list[Any]) -> list[list[list[Any]]]:
        """Return logical optimizer steps containing role-homogeneous microbatches."""
        size = int(self.config.minibatch_size)
        if getattr(self.config, "actor_minibatch_mode", "random") == "random":
            shuffled = list(transitions)
            random.shuffle(shuffled)
            return [[[item for item in chunk]] for chunk in _chunks(shuffled, size)]

        role_batches: dict[Any, list[list[Any]]] = {}
        for role in sorted({item.role for item in transitions}, key=lambda item: item.value):
            pool = [item for item in transitions if item.role is role]
            random.shuffle(pool)
            role_batches[role] = list(_chunks(pool, size))
        logical_steps = max((len(items) for items in role_batches.values()), default=0)
        return [
            [batches[index] for batches in role_batches.values() if index < len(batches)]
            for index in range(logical_steps)
        ]

    @staticmethod
    def _role_metrics(transitions: list[Any]) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for role in sorted({item.role for item in transitions}, key=lambda item: item.value):
            rows = [item for item in transitions if item.role is role]

            def moments(name: str) -> tuple[float, float]:
                values = [float(getattr(item, name, 0.0)) for item in rows]
                average = sum(values) / len(values)
                variance = sum((item - average) ** 2 for item in values) / len(values)
                return average, math.sqrt(variance)

            selected_counts = [
                len(item.parsed_action.get("selected_passage_ids", []))
                for item in rows if item.parsed_action is not None
                and "selected_passage_ids" in item.parsed_action
            ]
            answer_actions = [
                item.parsed_action for item in rows
                if item.parsed_action is not None and "can_answer" in item.parsed_action
            ]
            local_mean, local_std = moments("local_reward")
            team_mean, team_std = moments("team_reward")
            reward_mean, reward_std = moments("reward")
            raw_advantage_mean, raw_advantage_std = moments("raw_advantage")
            advantage_mean, advantage_std = moments("advantage")
            result[role.value] = {
                "transitions": float(len(rows)),
                "valid_action_rate": sum(item.valid for item in rows) / len(rows),
                "parse_failure_rate": sum(item.parse_error is not None for item in rows) / len(rows),
                "format_recovery_rate": sum(
                    item.format_recovery is not None for item in rows
                ) / len(rows),
                "format_repair_rate": sum(
                    item.format_recovery == "repaired" for item in rows
                ) / len(rows),
                "format_retry_rate": sum(
                    item.format_recovery == "retried" for item in rows
                ) / len(rows),
                "tokenization_recovery_rate": sum(
                    item.tokenization_recovery is not None for item in rows
                ) / len(rows),
                "action_tokens_mean": sum(len(item.action_ids) for item in rows) / len(rows),
                "local_reward_mean": local_mean,
                "local_reward_std": local_std,
                "team_reward_mean": team_mean,
                "team_reward_std": team_std,
                "total_reward_mean": reward_mean,
                "total_reward_std": reward_std,
                "raw_advantage_mean": raw_advantage_mean,
                "raw_advantage_std": raw_advantage_std,
                "normalized_advantage_mean": advantage_mean,
                "normalized_advantage_std": advantage_std,
                "positive_advantage_rate": sum(item.advantage > 0 for item in rows) / len(rows),
                "selected_evidence_count_mean": (
                    sum(selected_counts) / len(selected_counts) if selected_counts else 0.0
                ),
                "empty_evidence_selection_rate": (
                    sum(count == 0 for count in selected_counts) / len(selected_counts)
                    if selected_counts else 0.0
                ),
                "can_answer_rate": (
                    sum(bool(item["can_answer"]) for item in answer_actions) / len(answer_actions)
                    if answer_actions else 0.0
                ),
            }
        return result

    def _entropy_coefficient(self) -> float:
        initial = float(getattr(self.config, "entropy_coef", 0.0))
        final = float(getattr(self.config, "entropy_final_coef", initial))
        start = int(getattr(self.config, "entropy_anneal_start_step", 0))
        end = int(getattr(self.config, "entropy_anneal_end_step", 0))
        if end <= start:
            return final if self.global_step >= end else initial
        progress = min(1.0, max(0.0, (self.global_step - start) / (end - start)))
        return initial + progress * (final - initial)

    def _update(self, transitions: list[Any]) -> dict[str, Any]:
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
            "reference_kl": [], "selection_kl": [], "selection_raw_kl": [],
            "format_kl": [],
            "approx_kl": [], "clip_fraction": [], "grad_norm_actor": [], "grad_norm_critic": [],
        }
        role_metrics = self._role_metrics(transitions)
        role_optimizer: dict[str, dict[str, list[float]]] = {
            role: {
                "actor_loss": [], "entropy": [], "reference_kl": [],
                "selection_kl": [], "selection_raw_kl": [],
                "format_kl": [],
                "approx_kl": [], "clip_fraction": [],
            }
            for role in role_metrics
        }
        entropy_coefficient = self._entropy_coefficient()
        stop_for_kl = False
        policy_updates_applied = 0
        kl_rejected_updates = 0
        controllers = getattr(self, "reference_kl_controllers", {})
        observed_by_role = {
            role: {"sum": 0.0, "tokens": 0}
            for role in {item.role for item in transitions}
        }
        actor_parameters = [p for p in self.actor.model.parameters() if p.requires_grad]
        diagnostic_interval = int(getattr(
            self.config, "gradient_diagnostics_steps", 0,
        ))
        diagnostic_groups_remaining = (
            int(getattr(self.config, "gradient_diagnostics_max_groups", 1))
            if diagnostic_interval > 0 and self.global_step % diagnostic_interval == 0
            else 0
        )
        gradient_diagnostics: list[dict[str, Any]] = []
        if not hasattr(self, "reference_kl_recovery_steps"):
            self.reference_kl_recovery_steps = {role: 0 for role in AgentRole}
        recovery_steps_applied = dict(self.reference_kl_recovery_steps)
        accumulation_steps = int(getattr(
            self.config, "gradient_accumulation_steps", 1,
        ))
        logical_groups_applied = 0
        for _ in range(self.config.ppo_epochs):
            logical_groups = self._actor_minibatch_groups(transitions)
            for window_start in range(0, len(logical_groups), accumulation_steps):
                accumulation_window = logical_groups[
                    window_start:window_start + accumulation_steps
                ]
                window_size = len(accumulation_window)
                self.actor_optimizer.zero_grad(set_to_none=True)
                self.critic_optimizer.zero_grad(set_to_none=True)
                window_rejected = False
                window_critic_losses: list[Any] = []
                accepted_groups = 0
                for window_offset, microbatches in enumerate(accumulation_window):
                    logical_group_index = window_start + window_offset
                    logical_rows: list[Any] = []
                    capture_role_gradients = diagnostic_groups_remaining > 0
                    captured_gradients: dict[str, list[Any | None]] = {}
                    configured_role_weights = getattr(
                        self.config, "actor_role_weights", {},
                    ) or {}
                    raw_role_weights = [
                        float(configured_role_weights.get(batch[0].role.value, 1.0))
                        for batch in microbatches
                    ]
                    role_weight_denominator = sum(raw_role_weights)
                    for batch, raw_role_weight in zip(microbatches, raw_role_weights):
                        logical_rows.extend(batch)
                        batch_role = batch[0].role
                        role_weight = raw_role_weight / max(
                            1.0e-12, role_weight_denominator,
                        )
                        recovery_policy_scale = 1.0
                        if recovery_steps_applied.get(batch_role, 0) > 0:
                            recovery_policy_scale = float(getattr(
                                self.config, "reference_kl_recovery_role_scale", 0.25,
                            ))
                        reference_kl_beta = self._reference_kl_beta(batch_role)
                        sequences = [(item.prompt_ids, item.action_ids) for item in batch]
                        constraints = [item.token_constraints for item in batch]
                        has_constraints = any(item is not None for item in constraints)
                        if has_constraints:
                            new_logp, entropy, raw_new_logp = self.actor.score_batch_detailed(
                                sequences, token_constraints=constraints,
                            )
                        else:
                            new_logp, entropy = self.actor.score_batch(sequences)
                            raw_new_logp = new_logp
                        mask = torch.zeros_like(new_logp, dtype=torch.bool)
                        old_logp = torch.zeros_like(new_logp)
                        for row, item in enumerate(batch):
                            length = len(item.action_ids)
                            if item.role is AgentRole.EVIDENCE:
                                if item.optimization_token_mask is not None:
                                    mask[row, :length] = torch.tensor(
                                        item.optimization_token_mask,
                                        dtype=torch.bool, device=new_logp.device,
                                    )
                            else:
                                mask[row, :length] = True
                            old_logp[row, :length] = item.old_token_logprobs.to(new_logp.device)
                        token_count = int(mask.sum().item())
                        reference_kl = torch.zeros((), dtype=new_logp.dtype, device=new_logp.device)
                        segment_values = {name: 0.0 for name in ("selection", "format")}
                        if reference_kl_beta > 0:
                            reference_logp = torch.zeros_like(new_logp)
                            raw_reference_logp = torch.zeros_like(new_logp)
                            for row, item in enumerate(batch):
                                if item.reference_token_logprobs is None:
                                    raise RuntimeError("Missing SFT reference log-probabilities")
                                length = len(item.action_ids)
                                reference_logp[row, :length] = item.reference_token_logprobs.to(
                                    device=new_logp.device, dtype=new_logp.dtype,
                                )
                                raw_reference_logp[row, :length] = item.reference_raw_token_logprobs.to(
                                    device=new_logp.device, dtype=new_logp.dtype,
                                )
                            ref_log_ratio = (reference_logp - new_logp).clamp(-20.0, 20.0)
                            token_reference_kl = ref_log_ratio.exp() - 1.0 - ref_log_ratio
                            reference_kl = (
                                token_reference_kl * mask
                            ).sum() / mask.sum().clamp_min(1)
                            raw_ratio = (raw_reference_logp - raw_new_logp).clamp(-20.0, 20.0)
                            raw_token_kl = raw_ratio.exp() - 1.0 - raw_ratio
                            for segment in segment_values:
                                segment_mask = torch.zeros_like(mask)
                                for row, item in enumerate(batch):
                                    labels = item.token_segments or []
                                    if labels:
                                        segment_mask[row, :len(labels)] = torch.tensor(
                                            [name == segment for name in labels],
                                            dtype=torch.bool, device=new_logp.device,
                                        )
                                count = int(segment_mask.sum().item())
                                if count:
                                    segment_values[segment] = float(
                                        ((raw_token_kl * segment_mask).sum() / count).detach().float().cpu()
                                    )
                        advantages = torch.tensor(
                            [item.advantage for item in batch], dtype=torch.float32,
                            device=new_logp.device,
                        )
                        actor_loss, policy_stats = mappo_actor_loss(
                            new_logp, old_logp, advantages, mask, self.config.clip_epsilon,
                        )
                        entropy_mean = (entropy * mask).sum() / mask.sum().clamp_min(1)
                        numbers = {
                            "actor_loss": float(actor_loss.detach().float().cpu()),
                            "entropy": float(entropy_mean.detach().float().cpu()),
                            "reference_kl": float(reference_kl.detach().float().cpu()),
                            "selection_kl": (
                                float(reference_kl.detach().float().cpu())
                                if batch_role is AgentRole.EVIDENCE else 0.0
                            ),
                            "selection_raw_kl": segment_values["selection"],
                            "format_kl": segment_values["format"],
                            "approx_kl": float(policy_stats["approx_kl"].detach().float().cpu()),
                            "clip_fraction": float(policy_stats["clip_fraction"].detach().float().cpu()),
                        }
                        for name, number in numbers.items():
                            if not math.isfinite(number):
                                raise FloatingPointError(f"Non-finite MAPPO metric {name}: {number}")
                            aggregate[name].append(number)
                            role_optimizer[batch[0].role.value][name].append(number)
                        observed_by_role[batch_role]["sum"] += numbers["reference_kl"] * token_count
                        observed_by_role[batch_role]["tokens"] += token_count
                        if numbers["approx_kl"] > self.config.target_kl:
                            window_rejected = True
                            break
                        # Recovery suppresses the noisy PPO/entropy signal while
                        # preserving the full reference-KL restoring gradient.
                        actor_objective = role_weight * (
                            recovery_policy_scale * (
                                actor_loss - entropy_coefficient * entropy_mean
                            )
                            + reference_kl_beta * reference_kl
                        )
                        if capture_role_gradients:
                            role_gradients = torch.autograd.grad(
                                actor_objective,
                                actor_parameters,
                                allow_unused=True,
                            )
                            captured_gradients[batch_role.value] = [
                                None if gradient is None else gradient.detach().float().cpu()
                                for gradient in role_gradients
                            ]
                            for parameter, gradient in zip(actor_parameters, role_gradients):
                                if gradient is None:
                                    continue
                                detached = gradient.detach() / window_size
                                if parameter.grad is None:
                                    parameter.grad = detached
                                else:
                                    parameter.grad.add_(detached)
                        else:
                            (actor_objective / window_size).backward()
                    if window_rejected:
                        break
                    if capture_role_gradients and len(captured_gradients) >= 2:
                        diagnostic = gradient_conflict_metrics(captured_gradients)
                        diagnostic.update({
                            "event": "role_gradient_conflict",
                            "global_step": self.global_step + 1,
                            "ppo_epoch": _,
                            "logical_group_index": logical_group_index,
                            "roles": sorted(captured_gradients),
                            "objective": "weighted_actor_loss_entropy_reference_kl",
                        })
                        gradient_diagnostics.append(diagnostic)
                        _append(self.run_dir / "gradient_diagnostics.jsonl", diagnostic)
                        diagnostic_groups_remaining -= 1

                    values = self.critic([item.central_state for item in logical_rows])
                    old_values = torch.tensor(
                        [item.old_value for item in logical_rows],
                        dtype=torch.float32, device=values.device,
                    )
                    returns = torch.tensor(
                        [item.return_ for item in logical_rows],
                        dtype=torch.float32, device=values.device,
                    )
                    critic_loss = clipped_value_loss(
                        values, old_values, returns, self.config.value_clip_epsilon,
                    )
                    critic_objective = self.config.value_loss_coef * critic_loss
                    (critic_objective / window_size).backward()
                    window_critic_losses.append(critic_loss)
                    accepted_groups += 1

                if window_rejected:
                    # A KL violation rejects every pending gradient in this
                    # accumulation window; optimizer state is left unchanged.
                    self.actor_optimizer.zero_grad(set_to_none=True)
                    self.critic_optimizer.zero_grad(set_to_none=True)
                    kl_rejected_updates += 1
                    stop_for_kl = True
                    break
                # Gradients have been averaged across this window and across
                # roles inside each logical group.
                for parameter in actor_parameters:
                    if parameter.grad is not None:
                        break
                else:
                    raise RuntimeError("MAPPO actor objective produced no gradients")
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    actor_parameters,
                    self.config.max_grad_norm,
                )
                critic_parameters = list(self.critic.parameters())
                critic_norm = torch.nn.utils.clip_grad_norm_(
                    critic_parameters, self.config.max_grad_norm,
                )
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                policy_updates_applied += 1
                logical_groups_applied += accepted_groups

                values_to_add = {
                    "grad_norm_actor": actor_norm,
                    "grad_norm_critic": critic_norm,
                }
                for name, value in values_to_add.items():
                    number = float(value.detach().float().cpu())
                    if not math.isfinite(number):
                        raise FloatingPointError(f"Non-finite MAPPO metric {name}: {number}")
                    aggregate[name].append(number)
                for critic_loss in window_critic_losses:
                    number = float(critic_loss.detach().float().cpu())
                    if not math.isfinite(number):
                        raise FloatingPointError(
                            f"Non-finite MAPPO metric critic_loss: {number}"
                        )
                    aggregate["critic_loss"].append(number)
            if stop_for_kl:
                break
        for role, values_by_name in role_optimizer.items():
            role_metrics[role].update({
                name: sum(values) / max(1, len(values))
                for name, values in values_by_name.items()
            })
            role_metrics[role]["optimizer_microbatches"] = float(
                len(values_by_name["actor_loss"])
            )
            role_enum = AgentRole(role)
            role_metrics[role]["configured_actor_role_weight"] = float(
                (getattr(self.config, "actor_role_weights", {}) or {}).get(
                    role, 1.0,
                )
            )
            role_metrics[role]["recovery_role_scale_applied"] = (
                float(getattr(
                    self.config, "reference_kl_recovery_role_scale", 0.25,
                ))
                if recovery_steps_applied.get(role_enum, 0) > 0 else 1.0
            )
            role_metrics[role]["recovery_steps_remaining"] = float(
                recovery_steps_applied.get(role_enum, 0)
            )
        controller_by_role = {}
        for role, totals in observed_by_role.items():
            controller = controllers.get(role)
            observed = totals["sum"] / max(1, totals["tokens"])
            if controller is not None and totals["tokens"]:
                controller_by_role[role.value] = controller.update(observed)
            elif controller is not None:
                controller_by_role[role.value] = {
                    "reference_kl_observed": 0.0,
                    "reference_kl_ema": float(controller.ema_kl or 0.0),
                    "reference_kl_beta": float(controller.beta),
                    "reference_kl_beta_next": float(controller.beta),
                    "reference_kl_controller_error": 0.0,
                    "reference_kl_emergency_count": float(controller.emergency_count),
                    "reference_kl_emergency_triggered": 0.0,
                }
            else:
                beta = self._reference_kl_beta(role)
                controller_by_role[role.value] = {
                    "reference_kl_observed": observed,
                    "reference_kl_ema": observed,
                    "reference_kl_beta": beta,
                    "reference_kl_beta_next": beta,
                    "reference_kl_controller_error": 0.0,
                    "reference_kl_emergency_count": 0.0,
                    "reference_kl_emergency_triggered": 0.0,
                }
        total_reference_tokens = sum(item["tokens"] for item in observed_by_role.values())
        observed_reference_kl = sum(item["sum"] for item in observed_by_role.values()) / max(
            1, total_reference_tokens,
        )
        emergency_roles = [
            role for role, values in controller_by_role.items()
            if values["reference_kl_emergency_triggered"]
        ]
        controller_metrics = {
            "reference_kl_observed": observed_reference_kl,
            "reference_kl_ema": max(
                values["reference_kl_ema"] for values in controller_by_role.values()
            ),
            "reference_kl_beta": sum(
                values["reference_kl_beta"] for values in controller_by_role.values()
            ) / len(controller_by_role),
            "reference_kl_beta_next": sum(
                values["reference_kl_beta_next"] for values in controller_by_role.values()
            ) / len(controller_by_role),
            "reference_kl_controller_error": max(
                values["reference_kl_controller_error"] for values in controller_by_role.values()
            ),
            "reference_kl_emergency_count": max(
                values["reference_kl_emergency_count"] for values in controller_by_role.values()
            ),
            "reference_kl_emergency_triggered": float(bool(emergency_roles)),
            "reference_kl_emergency_roles": emergency_roles,
            "reference_kl_by_role": controller_by_role,
            "reference_kl_recovery_steps_applied": {
                role.value: int(steps)
                for role, steps in recovery_steps_applied.items()
            },
        }
        for role in self.reference_kl_recovery_steps:
            if self.reference_kl_recovery_steps[role] > 0:
                self.reference_kl_recovery_steps[role] -= 1
        return {name: sum(values) / max(1, len(values)) for name, values in aggregate.items()} | {
            "ppo_early_stop": float(stop_for_kl),
            "policy_updates_applied": float(policy_updates_applied),
            "logical_groups_applied": float(logical_groups_applied),
            "gradient_accumulation_steps": float(accumulation_steps),
            "kl_rejected_updates": float(kl_rejected_updates),
            "entropy_coefficient": entropy_coefficient,
            "role_metrics": role_metrics,
            "gradient_diagnostics": gradient_diagnostics,
            **controller_metrics,
            **behavior_stats,
        }

    def train(self) -> None:
        from tqdm.auto import tqdm

        self._learner_memory_preflight()
        self._validate_initial_policy()
        total_epochs = max(1, math.ceil(self.config.num_train_epochs))
        metrics_path = self.run_dir / "train_metrics.jsonl"
        episodes_path = self.run_dir / "episodes.jsonl"
        for epoch in range(self.start_epoch, total_epochs):
            ordered = stratified_epoch_order(
                self.samples, seed=self.config.seed, epoch=epoch,
                batch_size=self.config.rollout_batch_size,
            )
            total_update_steps = math.ceil(
                len(ordered) / self.config.rollout_batch_size
            )
            validation_schedule = set(evenly_spaced_steps(
                total_update_steps,
                int(getattr(self.config, "validation_checks_per_epoch", 0)),
            ))

            def scheduled_validation_due(step: int) -> bool:
                if validation_schedule:
                    return step in validation_schedule
                return (
                    self.config.validation_steps > 0
                    and step % self.config.validation_steps == 0
                )

            start = self.sample_offset if epoch == self.start_epoch else 0
            with tqdm(
                total=len(ordered), initial=start,
                desc=f"MAPPO epoch {epoch + 1}/{total_epochs}",
                unit="sample", dynamic_ncols=True,
            ) as progress:
                for batch_start in range(start, len(ordered), self.config.rollout_batch_size):
                    batch_samples = ordered[batch_start:batch_start + self.config.rollout_batch_size]
                    batch_started = time.perf_counter()
                    rollout_started = time.perf_counter()
                    def rollout_complete(index: int, sample: Any, episode: Any) -> None:
                        progress.set_postfix(
                            stage="rollout",
                            step=self.global_step,
                            batch=f"{index + 1}/{len(batch_samples)}",
                            qid=sample.qid[:24],
                            workers=min(
                                self.config.train_rollout_workers,
                                len(batch_samples),
                            ),
                            refresh=True,
                        )
                        progress.update(1)
                    episodes = self._collect_episodes(
                        batch_samples,
                        workers=self.config.train_rollout_workers,
                        on_complete=rollout_complete,
                    )
                    rollout_seconds = time.perf_counter() - rollout_started
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
                    update_started = time.perf_counter()
                    update = self._update(transitions)
                    update_seconds = time.perf_counter() - update_started
                    self.global_step += 1
                    sync_seconds = 0.0
                    if (
                        self.config.vllm_sync_after_update
                        and hasattr(self.actor, "sync_generation")
                    ):
                        progress.set_postfix(stage="vllm-sync", step=self.global_step, refresh=True)
                        sync_started = time.perf_counter()
                        self.actor.sync_generation(
                            sync_root=self.run_dir / "vllm_sync",
                            step=self.global_step,
                            keep=self.config.vllm_sync_snapshots_to_keep,
                        )
                        sync_seconds = time.perf_counter() - sync_started
                        update["vllm_synced"] = 1.0
                        update["vllm_sync_step"] = float(self.global_step)
                    batch_seconds = time.perf_counter() - batch_started
                    metric = {
                        "global_step": self.global_step, "epoch": epoch,
                        "samples_consumed": batch_start + len(batch_samples),
                        "episodes": len(episodes), "transitions": len(transitions),
                        "rollout_workers": min(
                            self.config.train_rollout_workers, len(batch_samples),
                        ),
                        "rollout_seconds": rollout_seconds,
                        "rollout_seconds_per_sample": (
                            rollout_seconds / max(1, len(batch_samples))
                        ),
                        "ppo_update_seconds": update_seconds,
                        "vllm_sync_seconds": sync_seconds,
                        "batch_processing_seconds": batch_seconds,
                        "batch_seconds_per_sample": (
                            batch_seconds / max(1, len(batch_samples))
                        ),
                        "projected_epoch_hours_at_current_batch_rate": (
                            batch_seconds / max(1, len(batch_samples))
                            * len(ordered) / 3600.0
                        ),
                        "retrieval_batch_stats": self.collector.retrieval.batch_stats(),
                        "reward_mean": sum(x.global_reward for x in episodes) / len(episodes),
                        "answer_f1_mean": sum(x.answer_f1 for x in episodes) / len(episodes),
                        "evidence_coverage_mean": sum(x.evidence_coverage for x in episodes) / len(episodes),
                        "dataset_counts": {
                            dataset: sum(
                                episode.dataset == dataset for episode in episodes
                            )
                            for dataset in sorted({
                                episode.dataset for episode in episodes
                            })
                        },
                        "quality_by_dataset": self._quality_by_dataset(episodes),
                        "evidence_duplicates_filtered": sum(
                            x.evidence_duplicates_filtered for x in episodes
                        ),
                        "invalid_action_fraction": sum(not x.valid for x in transitions) / len(transitions),
                        "parse_failure_rate": protocol["parse_failure_rate"],
                        "missing_answer_tag_rate": protocol["missing_answer_tag_rate"],
                        "final_compliance_rate": protocol["final_compliance_rate"],
                        "protocol_window_parse_failure_rate": protocol_status["window_parse_failure_rate"],
                        **update,
                    }
                    _append(metrics_path, metric)
                    if bool(update["reference_kl_emergency_triggered"]):
                        emergency_roles = [
                            AgentRole(role) for role in update.get(
                                "reference_kl_emergency_roles", []
                            )
                        ]
                        recovery_steps = int(getattr(
                            self.config, "reference_kl_recovery_steps", 10,
                        ))
                        beta_multiplier = float(getattr(
                            self.config,
                            "reference_kl_recovery_beta_multiplier", 1.5,
                        ))
                        boosted_betas = {}
                        for role in emergency_roles:
                            self.reference_kl_recovery_steps[role] = max(
                                self.reference_kl_recovery_steps.get(role, 0),
                                recovery_steps,
                            )
                            boosted_betas[role.value] = (
                                self.reference_kl_controllers[role].boost_beta(
                                    beta_multiplier,
                                )
                            )
                        run_emergency_validation = bool(getattr(
                            self.config,
                            "reference_kl_emergency_validation", False,
                        ))
                        emergency_validation_kind = None
                        validation = None
                        if run_emergency_validation:
                            progress.set_postfix(
                                stage="kl-emergency-validation",
                                step=self.global_step, refresh=True,
                            )
                            emergency_validation_kind = (
                                "scheduled"
                                if scheduled_validation_due(self.global_step)
                                else "kl_emergency"
                            )
                            validation = self._validate(
                                kind=emergency_validation_kind,
                            )
                        regressed = bool(
                            validation is not None
                            and math.isfinite(self.best_observed_validation_score)
                            and validation["validation_score"]
                            < self.best_observed_validation_score
                            - self.config.validation_min_delta
                        )
                        protocol_failed = bool(
                            validation is not None
                            and not validation["checkpoint_eligible"]
                        )
                        emergency_stop = bool(
                            getattr(self.config, "reference_kl_emergency_stop", True)
                            and self.global_step >= int(getattr(
                                self.config, "early_stopping_min_steps", 0,
                            ))
                            and regressed
                        )
                        _append(self.run_dir / "protocol_events.jsonl", {
                            "event": "reference_kl_emergency",
                            "global_step": self.global_step,
                            "validation_forced": run_emergency_validation,
                            "validation_kind": emergency_validation_kind,
                            "quality_regressed": regressed,
                            "protocol_failed": protocol_failed,
                            "checkpoint_gate_only": protocol_failed,
                            "stopped": emergency_stop,
                            "recovery_steps": recovery_steps,
                            "recovery_role_scale": float(getattr(
                                self.config,
                                "reference_kl_recovery_role_scale", 0.25,
                            )),
                            "boosted_reference_kl_betas": boosted_betas,
                            "reference_kl_emergency_roles": update.get(
                                "reference_kl_emergency_roles", []
                            ),
                            **{
                                name: update[name] for name in (
                                    "reference_kl_observed", "reference_kl_ema",
                                    "reference_kl_beta", "reference_kl_beta_next",
                                    "reference_kl_emergency_count",
                                )
                            },
                        })
                        if emergency_stop:
                            self.stopped_early = True
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
                    if (
                        scheduled_validation_due(self.global_step)
                        and self.last_validation_step != self.global_step
                    ):
                        progress.set_postfix(stage="validation", step=self.global_step, refresh=True)
                        validation = self._validate(kind="scheduled")
                        if validation is not None:
                            progress.set_postfix(
                                stage="validated", step=self.global_step,
                                val_f1=f"{validation['answer_f1_mean']:.3f}",
                                val_parse=f"{validation['parse_failure_rate']:.1%}",
                                best=self.best_validation_step, refresh=True,
                            )
                        if bool(getattr(self.config, "save_on_validation", False)):
                            self._save(epoch, batch_start + len(batch_samples))
                    if (
                        self.config.save_steps > 0
                        and self.global_step % self.config.save_steps == 0
                    ):
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
            reference_kl_controller_state={
                role.value: controller.state_dict()
                for role, controller in self.reference_kl_controllers.items()
            },
            reference_kl_recovery_state={
                role.value: steps
                for role, steps in self.reference_kl_recovery_steps.items()
            },
            best_observed_validation_score=(
                self.best_observed_validation_score
                if math.isfinite(self.best_observed_validation_score) else None
            ),
            best_validation_score=(
                self.best_validation_score
                if math.isfinite(self.best_validation_score) else None
            ),
            best_validation_step=self.best_validation_step,
            baseline_validation_score=self.baseline_validation_score,
            best_early_stopping_score=(
                self.best_early_stopping_score
                if math.isfinite(self.best_early_stopping_score) else None
            ),
            bad_validation_count=self.bad_validation_count,
        )
        prune_checkpoints(self.run_dir, self.config.save_total_limit)

    def _final_save(self, epoch: int, sample_offset: int) -> None:
        if self.validation_samples and self.last_validation_step != self.global_step:
            self._validate(kind="final")
        self._save(epoch, sample_offset)
        summary = {
            "algorithm": "mappo", "global_step": self.global_step, "complete": True,
            "stopped_early": self.stopped_early,
            "train_samples": len(self.samples),
            "validation_samples": len(self.validation_samples),
            "baseline_validation_score": self.baseline_validation_score,
            "best_validation_step": self.best_validation_step,
            "checkpoint_selection_objective": (
                "macro_answer_f1 + macro_evidence_coverage + "
                "macro_format_compliance"
            ),
            "best_composite_actor": (
                str(self.run_dir / "best_composite_actor")
                if (self.run_dir / "best_composite_actor").is_dir() else None
            ),
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
            "best_observed_validation_score": (
                self.best_observed_validation_score
                if math.isfinite(self.best_observed_validation_score) else None
            ),
            "best_early_stopping_score": (
                self.best_early_stopping_score
                if math.isfinite(self.best_early_stopping_score) else None
            ),
            "bad_validation_count": self.bad_validation_count,
            "reference_kl_recovery_steps": {
                role.value: steps
                for role, steps in self.reference_kl_recovery_steps.items()
            },
            "reference_kl_controllers": {
                role.value: controller.state_dict()
                for role, controller in self.reference_kl_controllers.items()
            },
        }
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
