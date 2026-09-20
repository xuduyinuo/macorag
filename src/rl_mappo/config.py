from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class MAPPOConfig:
    model_path: str = "model/Qwen2.5-7B-Instruct"
    sft_adapter_path: str = ""
    rl_data_root: str = "data/rl_train_2000_stratified_v2"
    retrieval_root: str = "data/rl_train_2000_stratified_v2_e5_faiss"
    output_root: str = "outputs/mappo_Qwen2.5-7B-Instruct"
    prompt_template_version: str = "macorag-shared-pointer-v1"
    resume_from_checkpoint: str = ""
    max_samples: int | None = 500
    max_total_samples: int | None = 3000
    validation_ratio: float = 0.1
    validation_samples_per_dataset: int | None = None
    musique_rare_hop_oversample_factor: float = 1.0
    validation_steps: int = 25
    validation_temperature: float = 0.0
    validation_score_answer_weight: float = 1.0
    validation_score_evidence_weight: float = 1.0
    validation_score_parse_penalty: float = 1.0
    validation_min_delta: float = 0.001
    early_stopping_patience: int = 3
    early_stopping_min_steps: int = 0
    protocol_window_size: int = 100
    max_protocol_parse_failure_rate: float = 0.02
    max_validation_missing_answer_tag_rate: float = 0.01
    min_validation_final_compliance_rate: float = 0.98
    seed: int = 42
    max_rounds: int = 4
    rollout_batch_size: int = 4
    num_train_epochs: float = 1.0
    max_prompt_length: int = 1024
    max_completion_length: int = 128
    answer_max_completion_length: int = 192
    prompt_max_evidence_items: int = 6
    prompt_max_history_items: int = 3
    prompt_evidence_text_chars: int = 160
    prompt_observation_text_chars: int = 200
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 5
    use_vllm_generation: bool = True
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8002
    vllm_gpu_indices: str = "1"
    vllm_served_model_name: str = "mappo_base"
    vllm_lora_name: str = "mappo_policy"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.85
    vllm_max_model_len: int = 4096
    vllm_max_num_seqs: int = 8
    vllm_dtype: str = "bfloat16"
    vllm_max_lora_rank: int = 64
    vllm_timeout_seconds: float = 180.0
    vllm_generate_attempts: int = 3
    vllm_retry_backoff_seconds: float = 1.0
    vllm_sync_after_update: bool = True
    vllm_sync_snapshots_to_keep: int = 2
    # vLLM samples actions with a BF16 inference engine, while the trainable
    # actor may use a 4-bit base. Re-score sampled actions with the trainable
    # actor before PPO so old/new log-probabilities share one numerical path.
    ppo_old_logprob_source: str = "local_actor"
    retrieval_backend: str = "e5_faiss"
    retrieval_embedding_model: str = "intfloat/e5-base-v2"
    retrieval_device: str = "cpu"
    retrieval_max_length: int = 512
    retrieval_batch_size: int = 32
    retrieval_top_k: int = 5
    actor_learning_rate: float = 1.0e-6
    critic_learning_rate: float = 3.0e-4
    weight_decay: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_clip_epsilon: float = 0.2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    entropy_final_coef: float = 0.0
    entropy_anneal_start_step: int = 0
    entropy_anneal_end_step: int = 0
    reference_kl_beta: float = 0.0
    reference_kl_mode: str = "fixed"
    reference_kl_target: float = 0.03
    reference_kl_beta_min: float = 0.0
    reference_kl_beta_max: float = 0.2
    reference_kl_ema_decay: float = 0.9
    reference_kl_deadband_low: float = 0.8
    reference_kl_deadband_high: float = 1.2
    reference_kl_controller_rate: float = 0.1
    reference_kl_emergency_threshold: float = 0.12
    reference_kl_emergency_patience: int = 5
    reference_kl_emergency_stop: bool = True
    reference_kl_emergency_validation: bool = False
    reference_kl_recovery_steps: int = 10
    reference_kl_recovery_role_scale: float = 0.25
    reference_kl_recovery_beta_multiplier: float = 1.5
    # Optional per-role controller overrides. Keys are role names and values
    # may override beta/target/bounds/EMA/deadband/rate/emergency settings.
    reference_kl_role_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_grad_norm: float = 1.0
    ppo_epochs: int = 1
    minibatch_size: int = 1
    actor_minibatch_mode: str = "random"
    actor_role_weights: dict[str, float] = field(default_factory=dict)
    gradient_diagnostics_steps: int = 0
    gradient_diagnostics_max_groups: int = 1
    target_kl: float = 0.02
    normalize_advantages: bool = True
    advantage_normalization_scope: str = "role"
    terminal_reward_weight: float = 1.0
    eta_query: float = 0.2
    eta_evidence: float = 0.2
    omega_answer: float = 1.5
    omega_evidence: float = 1.0
    answer_local_reward_weight: float = 1.0
    format_reward_weight: float = 0.1
    invalid_action_penalty: float = -1.0
    final_answer_invalid_penalty: float = -3.0
    non_final_wait_reward: float = 0.2
    final_answer_bonus: float = 1.0
    gate_terminal_evidence_on_valid_answer: bool = True
    force_final_answer_decoding: bool = False
    force_evidence_guided_decoding: bool = False
    evidence_min_selected_passages: int = 1
    evidence_rationale_max_chars: int = 64
    evidence_max_completion_length: int = 192
    critic_hash_buckets: int = 32768
    critic_embedding_dim: int = 128
    critic_hidden_dim: int = 256
    critic_text_max_tokens: int = 512
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: str = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    bf16: bool = True
    fp16: bool = False
    load_4bit: bool = False
    gradient_checkpointing: bool = True
    attn_implementation: str = "flash_attention_2"
    save_steps: int = 200
    save_total_limit: int = 5
    logging_steps: int = 1
    check_only: bool = False
    device: str = "cuda:0"

    def validate(self) -> None:
        positive = (
            "max_rounds", "rollout_batch_size", "max_prompt_length",
            "max_completion_length", "retrieval_top_k", "ppo_epochs",
            "answer_max_completion_length",
            "prompt_max_evidence_items", "prompt_max_history_items",
            "prompt_evidence_text_chars", "prompt_observation_text_chars",
            "minibatch_size", "critic_hash_buckets", "critic_embedding_dim",
            "critic_hidden_dim", "critic_text_max_tokens", "save_steps",
            "save_total_limit", "vllm_port", "vllm_tensor_parallel_size",
            "vllm_max_model_len", "vllm_max_num_seqs", "vllm_max_lora_rank",
            "vllm_generate_attempts", "vllm_sync_snapshots_to_keep",
            "validation_steps", "protocol_window_size",
            "reference_kl_emergency_patience",
            "reference_kl_recovery_steps",
            "evidence_rationale_max_chars",
            "evidence_max_completion_length",
            "gradient_diagnostics_max_groups",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("gamma", "gae_lambda"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("actor_learning_rate", "critic_learning_rate", "target_kl"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if self.bf16 and self.fp16:
            raise ValueError("bf16 and fp16 cannot both be enabled")
        if self.ppo_old_logprob_source not in {"local_actor", "vllm"}:
            raise ValueError("ppo_old_logprob_source must be local_actor or vllm")
        if self.advantage_normalization_scope not in {"global", "role"}:
            raise ValueError("advantage_normalization_scope must be global or role")
        if self.actor_minibatch_mode not in {"random", "role_balanced"}:
            raise ValueError("actor_minibatch_mode must be random or role_balanced")
        if self.reference_kl_mode not in {"fixed", "adaptive"}:
            raise ValueError("reference_kl_mode must be fixed or adaptive")
        valid_roles = {"query_retriever", "evidence_updater", "answer_generator"}
        if not isinstance(self.actor_role_weights, dict):
            raise ValueError("actor_role_weights must be a mapping")
        unknown_role_weights = set(self.actor_role_weights) - valid_roles
        if unknown_role_weights:
            raise ValueError(
                f"unknown actor role weights: {sorted(unknown_role_weights)}"
            )
        if any(float(value) <= 0.0 for value in self.actor_role_weights.values()):
            raise ValueError("actor role weights must be positive")
        valid_override_keys = {
            "beta", "target", "beta_min", "beta_max", "ema_decay",
            "deadband_low", "deadband_high", "controller_rate",
            "emergency_threshold", "emergency_patience", "mode",
        }
        if not isinstance(self.reference_kl_role_overrides, dict):
            raise ValueError("reference_kl_role_overrides must be a mapping")
        for role, overrides in self.reference_kl_role_overrides.items():
            if role not in valid_roles or not isinstance(overrides, dict):
                raise ValueError(f"invalid reference KL role override: {role}")
            unknown = set(overrides) - valid_override_keys
            if unknown:
                raise ValueError(
                    f"unknown reference KL override keys for {role}: {sorted(unknown)}"
                )
        if self.num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive")
        if self.max_samples is not None and self.max_samples < 0:
            raise ValueError("max_samples must be non-negative or null")
        if self.max_total_samples is not None and self.max_total_samples < 0:
            raise ValueError("max_total_samples must be non-negative or null")
        if (
            self.validation_samples_per_dataset is not None
            and self.validation_samples_per_dataset < 0
        ):
            raise ValueError(
                "validation_samples_per_dataset must be non-negative or null"
            )
        if self.musique_rare_hop_oversample_factor < 1.0:
            raise ValueError("musique_rare_hop_oversample_factor must be at least 1")
        if not 0.0 <= self.validation_ratio < 1.0:
            raise ValueError("validation_ratio must be in [0, 1)")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping_min_steps < 0:
            raise ValueError("early_stopping_min_steps must be non-negative")
        if self.gradient_diagnostics_steps < 0:
            raise ValueError("gradient_diagnostics_steps must be non-negative")
        if self.entropy_anneal_start_step < 0 or self.entropy_anneal_end_step < 0:
            raise ValueError("entropy anneal steps must be non-negative")
        if self.entropy_anneal_end_step < self.entropy_anneal_start_step:
            raise ValueError("entropy_anneal_end_step must be >= entropy_anneal_start_step")
        for name in (
            "validation_temperature", "validation_score_answer_weight",
            "validation_score_evidence_weight", "validation_score_parse_penalty",
            "validation_min_delta",
            "max_protocol_parse_failure_rate", "max_validation_missing_answer_tag_rate",
            "min_validation_final_compliance_rate",
            "entropy_coef", "entropy_final_coef", "reference_kl_beta",
            "reference_kl_target", "reference_kl_beta_min", "reference_kl_beta_max",
            "reference_kl_controller_rate", "reference_kl_emergency_threshold",
            "reference_kl_recovery_role_scale",
            "non_final_wait_reward", "final_answer_bonus",
        ):
            value = float(getattr(self, name))
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.reference_kl_recovery_role_scale > 1.0:
            raise ValueError("reference_kl_recovery_role_scale must be at most 1")
        if self.reference_kl_recovery_beta_multiplier < 1.0:
            raise ValueError("reference_kl_recovery_beta_multiplier must be at least 1")
        for name in (
            "max_protocol_parse_failure_rate", "max_validation_missing_answer_tag_rate",
            "min_validation_final_compliance_rate",
        ):
            if float(getattr(self, name)) > 1.0:
                raise ValueError(f"{name} must be at most 1")
        if self.retrieval_backend not in {"e5_faiss", "bm25"}:
            raise ValueError("retrieval_backend must be e5_faiss or bm25")
        if not 0.0 < self.vllm_gpu_memory_utilization <= 1.0:
            raise ValueError("vllm_gpu_memory_utilization must be in (0, 1]")
        if self.vllm_timeout_seconds <= 0 or self.vllm_retry_backoff_seconds < 0:
            raise ValueError("vLLM timeout must be positive and retry backoff non-negative")
        if self.use_vllm_generation and not self.vllm_lora_name.strip():
            raise ValueError("vllm_lora_name is required when vLLM generation is enabled")
        if self.force_final_answer_decoding and not self.use_vllm_generation:
            raise ValueError("force_final_answer_decoding currently requires vLLM generation")
        if self.force_evidence_guided_decoding and not self.use_vllm_generation:
            raise ValueError("force_evidence_guided_decoding currently requires vLLM generation")
        if not 0 <= self.evidence_min_selected_passages <= self.retrieval_top_k:
            raise ValueError(
                "evidence_min_selected_passages must be between 0 and retrieval_top_k"
            )
        if not 0.0 <= self.reference_kl_ema_decay < 1.0:
            raise ValueError("reference_kl_ema_decay must be in [0, 1)")
        if not 0.0 < self.reference_kl_deadband_low <= 1.0:
            raise ValueError("reference_kl_deadband_low must be in (0, 1]")
        if self.reference_kl_deadband_high < 1.0:
            raise ValueError("reference_kl_deadband_high must be at least 1")
        if self.reference_kl_beta_min > self.reference_kl_beta_max:
            raise ValueError("reference_kl_beta_min must not exceed reference_kl_beta_max")
        if not self.reference_kl_beta_min <= self.reference_kl_beta <= self.reference_kl_beta_max:
            raise ValueError("reference_kl_beta must be within its configured bounds")
        if self.reference_kl_mode == "adaptive" and self.reference_kl_beta <= 0.0:
            raise ValueError("adaptive reference KL requires a positive initial beta")
        if self.reference_kl_beta > 0 and not self.sft_adapter_path:
            raise ValueError("sft_adapter_path is required when reference_kl_beta > 0")
        if not self.sft_adapter_path and not self.resume_from_checkpoint:
            raise ValueError("sft_adapter_path is required for MAPPO initialization")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to read MAPPO config") from exc
    if not path.is_file():
        raise SystemExit(f"MAPPO config not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit("MAPPO config must be a YAML mapping")
    allowed = {item.name for item in fields(MAPPOConfig)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise SystemExit("Unknown MAPPO config keys: " + ", ".join(unknown))
    return payload


def parse_args(argv: list[str] | None = None) -> MAPPOConfig:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument(
        "--config",
        default=str(Path(__file__).with_name("train_mappo.yml")),
    )
    bootstrap.add_argument("--check-only", action="store_true")
    known, remaining = bootstrap.parse_known_args(argv)
    values = MAPPOConfig().to_dict()
    values.update(_load_yaml(Path(known.config)))
    if known.check_only:
        values["check_only"] = True

    parser = argparse.ArgumentParser(description="Self-contained MACORAG MAPPO trainer")
    parser.add_argument("--config", default=known.config)
    parser.add_argument("--check-only", action="store_true", default=values["check_only"])
    parser.add_argument("--resume-from-checkpoint", default=values["resume_from_checkpoint"])
    parser.add_argument("--max-samples", type=int, default=values["max_samples"])
    parser.add_argument("--device", default=values["device"])
    parsed = parser.parse_args(argv)
    values.update({
        "check_only": parsed.check_only,
        "resume_from_checkpoint": parsed.resume_from_checkpoint,
        "max_samples": parsed.max_samples,
        "device": parsed.device,
    })
    config = MAPPOConfig(**values)
    config.validate()
    return config
