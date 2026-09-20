#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

from .config import parse_args
from .data import (
    shuffled_training_samples, split_manifest, stratified_validation_samples,
)
from .models import CentralizedCritic, load_actor
from .retrieval import RetrievalEnvironment
from .protocol import validate_prompt_contract
from .rollout import RolloutCollector
from .trainer import MAPPOTrainer
from .vllm_client import VLLMClient


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    samples = shuffled_training_samples(
        config.train_file, seed=config.seed, limit=config.max_samples,
    )
    validation_samples = stratified_validation_samples(
        config.validation_file,
        seed=config.seed,
        limit=config.validation_max_samples,
    )
    loaded_samples = [*samples, *validation_samples]
    retrieval = RetrievalEnvironment(
        corpus_path=config.corpus_path, offsets_path=config.corpus_offsets_path,
        index_path=config.index_path, manifest_path=config.index_manifest_path,
        model_path=config.retrieval_model_path,
        device=config.retrieval_device, max_length=config.retrieval_max_length,
        batch_size=config.retrieval_batch_size,
        batch_wait_ms=config.retrieval_batch_wait_ms,
        top_k=config.retrieval_top_k, mmap=config.faiss_mmap,
        use_fp16=config.retrieval_use_fp16,
        max_passage_chars=config.retrieval_max_passage_chars,
    )
    datasets = {sample.dataset for sample in loaded_samples}
    retrieval.validate(datasets)
    adapter = Path(config.resume_from_checkpoint or config.sft_adapter_path)
    if not adapter.exists():
        raise SystemExit(f"MAPPO initialization path not found: {adapter}")
    contract = validate_prompt_contract(
        config.prompt_config_path,
        Path(config.sft_adapter_path) if config.sft_adapter_path else adapter,
        config.expected_prompt_contract_version,
    )
    report = {
        "status": "ok", "algorithm": "mappo", "samples": len(samples),
        "validation_samples": len(validation_samples),
        "datasets": sorted(datasets), "model_path": config.model_path,
        "adapter_path": str(adapter), "retrieval_backend": config.retrieval_backend,
        "retrieval_contract": {
            "model_path": config.retrieval_model_path,
            "pooling": config.retrieval_pooling_method,
            "instruction": config.retrieval_instruction,
            "query_prefix": "query: ",
            "query_max_length": config.retrieval_max_length,
            "use_fp16": config.retrieval_use_fp16,
            "faiss_gpu": False,
            "faiss_mmap": config.faiss_mmap,
            "top_k": config.retrieval_top_k,
            "max_passage_chars": config.retrieval_max_passage_chars,
            "preserve_faiss_rank": True,
        },
        "actor_load_4bit": config.load_4bit,
        "ppo_old_logprob_source": config.ppo_old_logprob_source,
        "completion_budgets": {
            "default": config.max_completion_length,
            "answer": config.answer_max_completion_length,
            "evidence": config.evidence_max_completion_length,
        },
        "actor_role_weights": config.actor_role_weights,
        "train_file": str(Path(config.train_file).resolve()),
        "validation_file": str(Path(config.validation_file).resolve()),
        "training_shuffle": {"enabled": True, "seed": config.seed, "before_limit": True},
        "parallel_rollouts": {
            "train_workers": config.train_rollout_workers,
            "validation_workers": config.validation_rollout_workers,
            "rollout_batch_size": config.rollout_batch_size,
            "vllm_max_num_seqs": config.vllm_max_num_seqs,
            "retrieval_batch_size": config.retrieval_batch_size,
            "retrieval_batch_wait_ms": config.retrieval_batch_wait_ms,
            "within_trajectory_sequential": True,
        },
        "prompt_contract_version": contract.version,
        "prompt_contract_fingerprint": contract.fingerprint,
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
    if config.use_vllm_generation and not config.check_only:
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
    initialization_started = time.perf_counter()
    stage_started = time.perf_counter()
    print("[rl-v2:init] Loading evaluation-aligned E5/FAISS retrieval before timed validation", flush=True)
    retrieval.initialize()
    initialization = {
        "retrieval_seconds": time.perf_counter() - stage_started,
    }
    stage_started = time.perf_counter()
    actor = load_actor(config, device, vllm_client=vllm_client)
    initialization["actor_seconds"] = time.perf_counter() - stage_started
    stage_started = time.perf_counter()
    critic = CentralizedCritic(
        hash_buckets=config.critic_hash_buckets,
        embedding_dim=config.critic_embedding_dim,
        hidden_dim=config.critic_hidden_dim,
        text_max_tokens=config.critic_text_max_tokens,
        device=device,
    )
    initialization["critic_seconds"] = time.perf_counter() - stage_started
    initialization["total_seconds"] = time.perf_counter() - initialization_started
    initialization["excluded_from_validation_timing"] = True
    (run_dir / "initialization_metrics.json").write_text(
        json.dumps(initialization, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    collector = RolloutCollector(actor=actor, critic=critic, retrieval=retrieval, config=config)
    report["evidence_completion_budget"] = collector.evidence_completion_budget
    (run_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (run_dir / "data_split.json").write_text(
        json.dumps({
            "shuffle": {"enabled": True, "seed": config.seed, "before_limit": True},
            "train": split_manifest(samples, config.train_file),
            "validation": split_manifest(validation_samples, config.validation_file),
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
