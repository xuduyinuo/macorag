from pathlib import Path

import pytest

from rl_training import owned_grpo_run as runner


def test_full_run_rejects_partial_target(monkeypatch, tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("output_root: out\n")
    parsed = type("Args", (), {"run_until_step": 200, "max_steps": 3000})()
    monkeypatch.setattr(runner, "parse_args", lambda args: parsed)
    with pytest.raises(ValueError, match="run_until_step"):
        runner.main(["--config", str(config)])


def test_full_run_rejects_overlapping_gpus(monkeypatch, tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("output_root: out\n")
    parsed = type("Args", (), {
        "run_until_step": 3000, "max_steps": 3000,
        "use_vllm_generation": True, "vllm_sync_mode": "lora",
        "gpu_indices": "0", "vllm_gpu_indices": "0",
    })()
    monkeypatch.setattr(runner, "parse_args", lambda args: parsed)
    with pytest.raises(ValueError, match="disjoint"):
        runner.main(["--config", str(config)])


def test_full_run_rejects_adapter_for_another_base(monkeypatch, tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("output_root: out\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text('{"base_model_name_or_path": "model/other"}')
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    (adapter / "prompt_contract.json").write_text("{}")
    parsed = type("Args", (), {
        "run_until_step": 3000, "max_steps": 3000,
        "use_vllm_generation": True, "vllm_sync_mode": "lora",
        "gpu_indices": "0", "vllm_gpu_indices": "1",
        "sft_adapter_path": str(adapter), "model_path": str(tmp_path / "model/qwen"),
    })()
    monkeypatch.setattr(runner, "parse_args", lambda args: parsed)
    with pytest.raises(ValueError, match="does not match"):
        runner.main(["--config", str(config)])
