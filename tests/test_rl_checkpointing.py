from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rl_training.config import parse_args
from rl_training.checkpointing import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_COMPLETE_FILE,
    CHECKPOINT_MANIFEST_FILE,
    OPTIMIZER_STATE_FILE,
    TRAINER_STATE_FILE,
    SCHEDULER_STATE_FILE,
    fingerprint_config,
    fingerprint_dataset,
    load_full_checkpoint_metadata,
    prune_full_checkpoints,
    restore_full_training_state,
    save_full_checkpoint,
)


class _ToyAdapter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def save_pretrained(self, path: Path, **kwargs: object) -> None:
        assert kwargs == {"selected_adapters": ["default"]}
        path.mkdir(parents=True, exist_ok=True)
        (path / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        torch.save(self.state_dict(), path / "adapter_model.bin")


class _ToyTokenizer:
    def save_pretrained(self, path: Path) -> None:
        (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")


def _constant_scheduler(optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)


def _scheduler_save_kwargs(scheduler, *, updates: int = 0) -> dict[str, object]:
    return {
        "scheduler": scheduler,
        "successful_optimizer_updates": updates,
        "scheduler_total_updates": 1000,
        "scheduler_warmup_updates": 30,
        "optimization_contract": {"max_grad_norm": 1.0, "logprob_source": "hf_rescore_v1"},
    }


def _make_complete_checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    (checkpoint / CHECKPOINT_MANIFEST_FILE).write_text(
        json.dumps({"schema_version": CHECKPOINT_SCHEMA_VERSION, "global_step": step}) + "\n",
        encoding="utf-8",
    )
    (checkpoint / CHECKPOINT_COMPLETE_FILE).write_text("complete\n", encoding="utf-8")
    return checkpoint


def test_save_full_checkpoint_is_atomic_and_contains_complete_training_state(tmp_path: Path) -> None:
    model = _ToyAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    (model.weight.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = _constant_scheduler(optimizer)

    random.seed(17)
    np.random.seed(23)
    torch.manual_seed(29)
    checkpoint = save_full_checkpoint(
        raw_policy_model=model,
        tokenizer=_ToyTokenizer(),
        optimizer=optimizer,
        **_scheduler_save_kwargs(scheduler),
        output_dir=tmp_path,
        step=7,
        epoch=2,
        samples_consumed=11,
        generation_counter=31,
        gradient_accumulation_steps=1,
        dataset_fingerprint="dataset-sha",
        config_fingerprint="config-sha",
        torch_module=torch,
        save_total_limit=3,
        milestone_steps=1000,
    )

    assert checkpoint == tmp_path / "checkpoint-7"
    assert not (tmp_path / ".checkpoint-7.tmp").exists()
    assert {
        "adapter_config.json",
        "adapter_model.bin",
        "tokenizer_config.json",
        OPTIMIZER_STATE_FILE,
        SCHEDULER_STATE_FILE,
        TRAINER_STATE_FILE,
        "rng_state_rank0.pt",
        CHECKPOINT_MANIFEST_FILE,
        CHECKPOINT_COMPLETE_FILE,
    } <= {path.name for path in checkpoint.iterdir()}

    trainer_state = torch.load(checkpoint / TRAINER_STATE_FILE, map_location="cpu")
    assert trainer_state == {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": 2,
        "samples_consumed": 11,
        "global_step": 7,
        "generation_counter": 31,
        "gradient_accumulation_steps": 1,
        "world_size": 1,
        "dataset_fingerprint": "dataset-sha",
        "config_fingerprint": "config-sha",
        "successful_optimizer_updates": 0,
        "scheduler_total_updates": 1000,
        "scheduler_warmup_updates": 30,
        "optimization_contract": {"max_grad_norm": 1.0, "logprob_source": "hf_rescore_v1"},
    }
    optimizer_state = torch.load(checkpoint / OPTIMIZER_STATE_FILE, map_location="cpu")
    assert optimizer_state["state"]
    manifest = json.loads((checkpoint / CHECKPOINT_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["global_step"] == 7
    assert manifest["expected_files"] == sorted(manifest["expected_files"])


def test_prune_full_checkpoints_keeps_latest_and_milestones(tmp_path: Path) -> None:
    for step in (100, 200, 1000, 1700, 1800, 1900, 2000):
        _make_complete_checkpoint(tmp_path, step)
    incomplete = tmp_path / "checkpoint-1950"
    incomplete.mkdir()

    removed = prune_full_checkpoints(tmp_path, keep_last=3, milestone_steps=1000)

    assert {path.name for path in removed} == {
        "checkpoint-100",
        "checkpoint-200",
        "checkpoint-1700",
    }
    assert {path.name for path in tmp_path.glob("checkpoint-*")} == {
        "checkpoint-1000",
        "checkpoint-1800",
        "checkpoint-1900",
        "checkpoint-1950",
        "checkpoint-2000",
    }


@pytest.mark.parametrize("keep_last,milestone_steps", [(-1, 1000), (3, -1)])
def test_prune_full_checkpoints_rejects_negative_limits(
    tmp_path: Path,
    keep_last: int,
    milestone_steps: int,
) -> None:
    with pytest.raises(ValueError):
        prune_full_checkpoints(
            tmp_path,
            keep_last=keep_last,
            milestone_steps=milestone_steps,
        )


def test_restore_full_training_state_round_trips_optimizer_and_rng(tmp_path: Path) -> None:
    model = _ToyAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    (model.weight.square().sum()).backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler = _constant_scheduler(optimizer)
    expected_optimizer = deepcopy(optimizer.state_dict())

    random.seed(101)
    np.random.seed(103)
    torch.manual_seed(107)
    checkpoint = save_full_checkpoint(
        raw_policy_model=model,
        tokenizer=_ToyTokenizer(),
        optimizer=optimizer,
        **_scheduler_save_kwargs(scheduler),
        output_dir=tmp_path,
        step=9,
        epoch=1,
        samples_consumed=9,
        generation_counter=27,
        gradient_accumulation_steps=1,
        dataset_fingerprint="dataset-sha",
        config_fingerprint="config-sha",
        torch_module=torch,
        save_total_limit=3,
        milestone_steps=1000,
    )
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(3)

    optimizer.state.clear()
    new_scheduler = _constant_scheduler(optimizer)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)

    restored = restore_full_training_state(
        checkpoint,
        optimizer=optimizer,
        scheduler=new_scheduler,
        torch_module=torch,
        expected_dataset_fingerprint="dataset-sha",
        expected_config_fingerprint="config-sha",
        expected_gradient_accumulation_steps=1,
        expected_world_size=1,
    )

    assert restored["global_step"] == 9
    assert restored["generation_counter"] == 27
    actual_optimizer = optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    assert actual_optimizer["state"].keys() == expected_optimizer["state"].keys()
    for parameter_id, expected_state in expected_optimizer["state"].items():
        for name, expected_value in expected_state.items():
            actual_value = actual_optimizer["state"][parameter_id][name]
            if torch.is_tensor(expected_value):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(3), expected_torch)


def test_load_full_checkpoint_metadata_rejects_incomplete_checkpoint(tmp_path: Path) -> None:
    checkpoint = _make_complete_checkpoint(tmp_path, 4)
    (checkpoint / CHECKPOINT_COMPLETE_FILE).unlink()

    with pytest.raises(RuntimeError, match="not complete"):
        load_full_checkpoint_metadata(checkpoint)


@pytest.mark.parametrize(
    "keyword,value,message",
    [
        ("expected_dataset_fingerprint", "other", "dataset fingerprint"),
        ("expected_config_fingerprint", "other", "config fingerprint"),
        ("expected_gradient_accumulation_steps", 2, "gradient accumulation"),
        ("expected_world_size", 2, "world size"),
    ],
)
def test_restore_full_training_state_rejects_identity_mismatch(
    tmp_path: Path,
    keyword: str,
    value: object,
    message: str,
) -> None:
    model = _ToyAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = _constant_scheduler(optimizer)
    checkpoint = save_full_checkpoint(
        raw_policy_model=model,
        tokenizer=_ToyTokenizer(),
        optimizer=optimizer,
        **_scheduler_save_kwargs(scheduler),
        output_dir=tmp_path,
        step=3,
        epoch=1,
        samples_consumed=3,
        generation_counter=8,
        gradient_accumulation_steps=1,
        dataset_fingerprint="dataset-sha",
        config_fingerprint="config-sha",
        torch_module=torch,
        save_total_limit=3,
        milestone_steps=1000,
    )
    arguments = {
        "expected_dataset_fingerprint": "dataset-sha",
        "expected_config_fingerprint": "config-sha",
        "expected_gradient_accumulation_steps": 1,
        "expected_world_size": 1,
    }
    arguments[keyword] = value

    with pytest.raises(RuntimeError, match=message):
        restore_full_training_state(
            checkpoint,
            optimizer=optimizer,
            scheduler=scheduler,
            torch_module=torch,
            **arguments,
        )


def test_checkpoint_fingerprints_are_stable_and_dataset_order_sensitive() -> None:
    class Sample:
        def __init__(self, dataset: str, qid: str, question: str = "question") -> None:
            self.dataset = dataset
            self.qid = qid
            self.question = question
            self.answer = "answer"
            self.answer_aliases = ["alias"]
            self.supporting_facts = [{"title": "title", "text": "fact"}]
            self.context_doc_ids = ["doc"]

    class Args:
        model_path = "base"
        sft_adapter_path = "sft"
        group_size = 4
        max_rounds = 4
        max_prompt_length = 2048
        max_completion_length = 192
        temperature = 0.8
        top_p = 0.95
        top_k = 5
        gradient_accumulation_steps = 1
        learning_rate = 1e-5
        weight_decay = 0.0
        clip_epsilon = 0.2
        kl_beta = 0.02
        query_global_reward_weight = 1 / 3
        evidence_global_reward_weight = 3 / 7
        answer_global_reward_weight = 7 / 3
        advantage_epsilon = 1e-8
        retrieval_backend = "e5_faiss"
        retrieval_root = "index"
        retrieval_embedding_model = "intfloat/e5-base-v2"
        retrieval_top_k = 5

    first = [Sample("2wiki", "a"), Sample("hotpotqa", "b")]
    second = list(reversed(first))

    assert fingerprint_dataset(first) == fingerprint_dataset(first)
    assert fingerprint_dataset(first) != fingerprint_dataset(second)
    assert fingerprint_dataset(first) != fingerprint_dataset(
        [Sample("2wiki", "a", question="changed"), Sample("hotpotqa", "b")]
    )
    assert fingerprint_config(Args()) == fingerprint_config(Args())


def test_parse_args_loads_full_checkpoint_retention_fields(tmp_path: Path) -> None:
    config = tmp_path / "train.yml"
    config.write_text(
        "save_total_limit: 4\nsave_milestone_steps: 500\n",
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config)])

    assert args.save_total_limit == 4
    assert args.save_milestone_steps == 500


def test_config_fingerprint_tracks_attention_implementation() -> None:
    flash = SimpleNamespace(attn_implementation="flash_attention_2")
    sdpa = SimpleNamespace(attn_implementation="sdpa")

    assert fingerprint_config(flash) != fingerprint_config(sdpa)


def test_resolve_resume_state_reads_cursor_from_full_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rl_training import train_grpo_macorag as training

    checkpoint = tmp_path / "checkpoint-17"
    checkpoint.mkdir()
    manifest = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "epoch": 2,
        "samples_consumed": 7,
        "global_step": 17,
        "generation_counter": 123,
        "gradient_accumulation_steps": 1,
        "world_size": 1,
        "dataset_fingerprint": "dataset-sha",
        "config_fingerprint": "config-sha",
        "expected_files": [CHECKPOINT_COMPLETE_FILE, CHECKPOINT_MANIFEST_FILE],
    }
    (checkpoint / CHECKPOINT_MANIFEST_FILE).write_text(
        json.dumps(manifest) + "\n",
        encoding="utf-8",
    )
    (checkpoint / CHECKPOINT_COMPLETE_FILE).write_text("complete\n", encoding="utf-8")
    monkeypatch.setattr(training, "_validate_resume_checkpoint", lambda path: None)
    args = SimpleNamespace(
        resume_from_checkpoint=str(checkpoint),
        resume_epoch=1,
        resume_samples_consumed=0,
        resume_global_step=0,
    )

    state = training._resolve_resume_state(args, rank_epoch_size=20, total_epochs=3)

    assert state.epoch == 2
    assert state.samples_consumed == 7
    assert state.global_step == 17
    assert state.generation_counter == 123
    assert state.is_full_checkpoint is True
    assert state.optimizer_state_restored is False


def test_resume_meta_reports_full_state_restoration_flags(tmp_path: Path) -> None:
    from rl_training.train_grpo_macorag import ResumeState, _resume_meta_payload

    state = ResumeState(
        checkpoint_path=tmp_path / "checkpoint-17",
        epoch=2,
        samples_consumed=7,
        global_step=17,
        generation_counter=123,
        is_full_checkpoint=True,
        optimizer_state_restored=True,
        rng_state_restored=True,
    )

    payload = _resume_meta_payload(state)

    assert payload["checkpoint_schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert payload["optimizer_state_restored"] is True
    assert payload["rng_state_restored"] is True
    assert payload["generation_counter"] == 123


def test_due_checkpoint_saves_on_safe_skipped_update_boundary() -> None:
    from rl_training.train_grpo_macorag import _checkpoint_save_decision

    should_save, pending = _checkpoint_save_decision(
        global_step=100,
        save_steps=100,
        optimizer_safe_boundary=True,
        pending=False,
    )
    assert (should_save, pending) == (True, False)


def test_due_checkpoint_remains_pending_while_gradients_are_accumulated() -> None:
    from rl_training.train_grpo_macorag import _checkpoint_save_decision

    should_save, pending = _checkpoint_save_decision(
        global_step=100,
        save_steps=100,
        optimizer_safe_boundary=False,
        pending=False,
    )
    assert (should_save, pending) == (False, True)


def test_pending_checkpoint_saves_at_next_safe_boundary() -> None:
    from rl_training.train_grpo_macorag import _checkpoint_save_decision

    should_save, pending = _checkpoint_save_decision(
        global_step=101,
        save_steps=100,
        optimizer_safe_boundary=True,
        pending=True,
    )
    assert (should_save, pending) == (True, False)


def test_finite_final_partial_accumulation_steps_clears_syncs_then_saves() -> None:
    from rl_training.train_grpo_macorag import _flush_pending_gradients_if_finite
    from rl_training.train_grpo_macorag import _run_final_checkpoint_actions

    events: list[str] = []

    class Optimizer:
        def step(self) -> None:
            events.append("step")

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            events.append("zero")

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    flush_results: list[bool] = []

    def flush() -> None:
        flush_results.append(
            _flush_pending_gradients_if_finite(
                raw_policy_model=model,
                optimizer=Optimizer(),
                torch=torch,
                sync_weights=lambda: events.append("sync"),
            )
        )

    _run_final_checkpoint_actions(
        has_pending_gradients=True,
        checkpoint_pending=True,
        flush_pending_gradients=flush,
        save_pending_checkpoint=lambda: events.append("save"),
    )

    assert flush_results == [True]
    assert events == ["step", "zero", "sync", "save"]


def test_nonfinite_final_partial_accumulation_clears_then_saves_without_step_or_sync() -> None:
    from rl_training.train_grpo_macorag import _flush_pending_gradients_if_finite
    from rl_training.train_grpo_macorag import _run_final_checkpoint_actions

    events: list[str] = []

    class Optimizer:
        def step(self) -> None:
            events.append("step")

        def zero_grad(self, *, set_to_none: bool) -> None:
            assert set_to_none is True
            events.append("zero")

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, torch.nan)
    flush_results: list[bool] = []

    def flush() -> None:
        flush_results.append(
            _flush_pending_gradients_if_finite(
                raw_policy_model=model,
                optimizer=Optimizer(),
                torch=torch,
                sync_weights=lambda: events.append("sync"),
            )
        )

    _run_final_checkpoint_actions(
        has_pending_gradients=True,
        checkpoint_pending=True,
        flush_pending_gradients=flush,
        save_pending_checkpoint=lambda: events.append("save"),
    )

    assert flush_results == [False]
    assert events == ["zero", "save"]


def test_interrupted_resume_matches_uninterrupted_toy_training(tmp_path: Path) -> None:
    def update(model: _ToyAdapter, optimizer: torch.optim.Optimizer) -> None:
        target = torch.rand(1)
        loss = (model.weight - target).square().sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    random.seed(211)
    np.random.seed(223)
    torch.manual_seed(227)
    continuous_model = _ToyAdapter()
    continuous_optimizer = torch.optim.AdamW(continuous_model.parameters(), lr=0.01)
    update(continuous_model, continuous_optimizer)
    continuous_scheduler = _constant_scheduler(continuous_optimizer)
    checkpoint = save_full_checkpoint(
        raw_policy_model=continuous_model,
        tokenizer=_ToyTokenizer(),
        optimizer=continuous_optimizer,
        **_scheduler_save_kwargs(continuous_scheduler),
        output_dir=tmp_path,
        step=1,
        epoch=1,
        samples_consumed=1,
        generation_counter=9,
        gradient_accumulation_steps=1,
        dataset_fingerprint="dataset-sha",
        config_fingerprint="config-sha",
        torch_module=torch,
        save_total_limit=3,
        milestone_steps=1000,
    )
    update(continuous_model, continuous_optimizer)
    expected_weight = continuous_model.weight.detach().clone()
    expected_optimizer = deepcopy(continuous_optimizer.state_dict())

    resumed_model = _ToyAdapter()
    resumed_model.load_state_dict(
        torch.load(checkpoint / "adapter_model.bin", map_location="cpu", weights_only=False)
    )
    resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=0.01)
    resumed_scheduler = _constant_scheduler(resumed_optimizer)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    restored = restore_full_training_state(
        checkpoint,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
        torch_module=torch,
        expected_dataset_fingerprint="dataset-sha",
        expected_config_fingerprint="config-sha",
        expected_gradient_accumulation_steps=1,
        expected_world_size=1,
    )
    update(resumed_model, resumed_optimizer)

    assert restored["samples_consumed"] == 1
    assert torch.equal(resumed_model.weight, expected_weight)
    actual_optimizer = resumed_optimizer.state_dict()
    assert actual_optimizer["param_groups"] == expected_optimizer["param_groups"]
    for parameter_id, expected_state in expected_optimizer["state"].items():
        for name, expected_value in expected_state.items():
            actual_value = actual_optimizer["state"][parameter_id][name]
            if torch.is_tensor(expected_value):
                assert torch.equal(actual_value, expected_value)
            else:
                assert actual_value == expected_value


def test_full_checkpoint_resume_script_uses_checkpoint_metadata_only() -> None:
    script = Path("scripts/run_train_grpo_resume.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "RESUME_CHECKPOINT" in script
    assert "checkpoint_manifest.json" in script
    assert "COMPLETE" in script
    assert '--resume-from-checkpoint "${RESUME_CHECKPOINT}"' in script
    assert "--resume-epoch" not in script
    assert "--resume-samples-consumed" not in script
    assert "--resume-global-step" not in script
    assert "RUN_UNTIL_STEP" in script
    assert '--run-until-step "${RUN_UNTIL_STEP}"' in script


def test_run_until_step_is_operational_not_critical() -> None:
    first = SimpleNamespace(max_steps=1000, run_until_step=300)
    second = SimpleNamespace(max_steps=1000, run_until_step=1000)
    changed_horizon = SimpleNamespace(max_steps=999, run_until_step=300)

    assert fingerprint_config(first) == fingerprint_config(second)
    assert fingerprint_config(first) != fingerprint_config(changed_horizon)


def test_config_fingerprint_tracks_vllm_sync_contract() -> None:
    every_step = SimpleNamespace(
        vllm_sync_mode="lora",
        vllm_sync_after_step=True,
        vllm_sync_every_steps=1,
    )
    stale_rollouts = SimpleNamespace(
        vllm_sync_mode="lora",
        vllm_sync_after_step=True,
        vllm_sync_every_steps=4,
    )

    assert fingerprint_config(every_step) != fingerprint_config(stale_rollouts)


def test_full_checkpoint_round_trips_scheduler_and_update_count(tmp_path: Path) -> None:
    model = _ToyAdapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    optimizer.step()
    scheduler.step()
    checkpoint = save_full_checkpoint(
        raw_policy_model=model,
        tokenizer=_ToyTokenizer(),
        optimizer=optimizer,
        scheduler=scheduler,
        successful_optimizer_updates=1,
        scheduler_total_updates=1000,
        scheduler_warmup_updates=30,
        optimization_contract={"max_grad_norm": 1.0, "logprob_source": "hf_rescore_v1"},
        prompt_contract_metadata={
            "prompt_contract_version": "v1",
            "prompt_contract_fingerprint": "prompt-sha",
        },
        output_dir=tmp_path,
        step=1,
        epoch=1,
        samples_consumed=1,
        generation_counter=1,
        gradient_accumulation_steps=1,
        dataset_fingerprint="dataset-sha",
        config_fingerprint="config-sha",
        torch_module=torch,
        save_total_limit=3,
        milestone_steps=1000,
    )
    new_model = _ToyAdapter()
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=0.01)
    new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lambda step: 1.0)
    restored = restore_full_training_state(
        checkpoint,
        optimizer=new_optimizer,
        scheduler=new_scheduler,
        torch_module=torch,
        expected_dataset_fingerprint="dataset-sha",
        expected_config_fingerprint="config-sha",
        expected_gradient_accumulation_steps=1,
        expected_world_size=1,
    )

    assert (checkpoint / SCHEDULER_STATE_FILE).is_file()
    assert restored["successful_optimizer_updates"] == 1
    assert new_scheduler.state_dict() == scheduler.state_dict()
    assert json.loads((checkpoint / "prompt_contract.json").read_text(encoding="utf-8")) == {
        "prompt_contract_fingerprint": "prompt-sha",
        "prompt_contract_version": "v1",
    }
