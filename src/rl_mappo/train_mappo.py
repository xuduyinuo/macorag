#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

from .config import parse_args
from .data import load_samples, sample_stratum, split_train_validation
from .models import CentralizedCritic, load_actor
from .retrieval import RetrievalEnvironment
from .rollout import RolloutCollector
from .trainer import MAPPOTrainer
from .vllm_client import VLLMClient


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    loaded_samples = load_samples(
        config.rl_data_root, max_per_dataset=config.max_samples,
        max_total=config.max_total_samples, seed=config.seed,
        musique_rare_hop_oversample_factor=config.musique_rare_hop_oversample_factor,
    )
    samples, validation_samples = split_train_validation(
        loaded_samples, validation_ratio=config.validation_ratio,
        seed=config.seed,
        validation_samples_per_dataset=config.validation_samples_per_dataset,
    )
    retrieval = RetrievalEnvironment(
        root=config.retrieval_root, backend=config.retrieval_backend,
        embedding_model=config.retrieval_embedding_model,
        device=config.retrieval_device, max_length=config.retrieval_max_length,
        batch_size=config.retrieval_batch_size, top_k=config.retrieval_top_k,
    )
    datasets = {sample.dataset for sample in loaded_samples}
    retrieval.validate(datasets)
    adapter = Path(config.resume_from_checkpoint or config.sft_adapter_path)
    if not adapter.exists():
        raise SystemExit(f"MAPPO initialization path not found: {adapter}")
    report = {
        "status": "ok", "algorithm": "mappo", "samples": len(samples),
        "validation_samples": len(validation_samples),
        "datasets": sorted(datasets), "model_path": config.model_path,
        "adapter_path": str(adapter), "retrieval_backend": config.retrieval_backend,
        "actor_load_4bit": config.load_4bit,
        "ppo_old_logprob_source": config.ppo_old_logprob_source,
        "completion_budgets": {
            "default": config.max_completion_length,
            "answer": config.answer_max_completion_length,
            "evidence": config.evidence_max_completion_length,
        },
        "actor_role_weights": config.actor_role_weights,
        "musique_rare_hop_oversample_factor": config.musique_rare_hop_oversample_factor,
        "reference_kl_beta": config.reference_kl_beta,
        "reference_kl_recovery": {
            "steps": config.reference_kl_recovery_steps,
            "role_scale": config.reference_kl_recovery_role_scale,
            "beta_multiplier": config.reference_kl_recovery_beta_multiplier,
            "emergency_stop": config.reference_kl_emergency_stop,
            "emergency_validation": config.reference_kl_emergency_validation,
        },
        "entropy_schedule": {
            "initial": config.entropy_coef,
            "final": config.entropy_final_coef,
            "start_step": config.entropy_anneal_start_step,
            "end_step": config.entropy_anneal_end_step,
        },
        "force_final_answer_decoding": config.force_final_answer_decoding,
        "checkpoint_selection": "protocol-gated macro_answer_f1",
        "early_stopping": {
            "scope": "scheduled_validation_only",
            "validation_steps": config.validation_steps,
            "patience": config.early_stopping_patience,
            "min_steps": config.early_stopping_min_steps,
        },
        "generation_backend": "vllm" if config.use_vllm_generation else "huggingface",
        "vllm_url": f"http://{config.vllm_host}:{config.vllm_port}" if config.use_vllm_generation else None,
    }
    vllm_client = None
    if config.use_vllm_generation:
        vllm_client = VLLMClient(
            host=config.vllm_host, port=config.vllm_port,
            model_name=config.vllm_served_model_name,
            lora_name=config.vllm_lora_name,
            timeout=config.vllm_timeout_seconds,
            attempts=config.vllm_generate_attempts,
            backoff=config.vllm_retry_backoff_seconds,
            initial_adapter_path=(adapter / "actor" if (adapter / "actor").is_dir() else adapter),
        )
        report.update(vllm_client.check_server())
    if config.check_only:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    import numpy as np
    import torch
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    run_dir = Path(config.output_root) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    actor = load_actor(config, device, vllm_client=vllm_client)
    critic = CentralizedCritic(
        hash_buckets=config.critic_hash_buckets,
        embedding_dim=config.critic_embedding_dim,
        hidden_dim=config.critic_hidden_dim,
        text_max_tokens=config.critic_text_max_tokens,
        device=device,
    )
    collector = RolloutCollector(actor=actor, critic=critic, retrieval=retrieval, config=config)
    report["evidence_completion_budget"] = collector.evidence_completion_budget
    (run_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "data_split.json").write_text(
        json.dumps({
            "seed": config.seed,
            "train": [
                {"dataset": item.dataset, "qid": item.qid, "stratum": sample_stratum(item)}
                for item in samples
            ],
            "validation": [
                {"dataset": item.dataset, "qid": item.qid, "stratum": sample_stratum(item)}
                for item in validation_samples
            ],
        }, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report | {"run_dir": str(run_dir)}, ensure_ascii=False), flush=True)
    MAPPOTrainer(
        actor=actor, critic=critic, collector=collector,
        samples=samples, validation_samples=validation_samples,
        config=config, run_dir=run_dir,
    ).train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
