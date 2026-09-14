"""Run one GRPO config with an owned vLLM server and guaranteed cleanup."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path

import yaml

from .answer_reward_ablation import REPO_ROOT, _command, _environment, _server
from .config import parse_args


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint", default="")
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    parsed = parse_args(["--config", str(config_path)])
    if parsed.run_until_step != parsed.max_steps:
        raise ValueError("Owned full run requires run_until_step == max_steps")
    if not parsed.use_vllm_generation or parsed.vllm_sync_mode != "lora":
        raise ValueError("Owned run requires LoRA vLLM generation")
    if set(parsed.gpu_indices.split(",")) & set(parsed.vllm_gpu_indices.split(",")):
        raise ValueError("Training and vLLM GPUs must be disjoint")
    adapter = Path(parsed.sft_adapter_path).resolve()
    for name in ("adapter_config.json", "adapter_model.safetensors", "prompt_contract.json"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(f"Incomplete SFT adapter: missing {adapter / name}")
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_base = Path(str(adapter_config.get("base_model_name_or_path") or "")).resolve()
    configured_base = Path(parsed.model_path).resolve()
    if adapter_base != configured_base:
        raise ValueError(f"SFT adapter base {adapter_base} does not match configured model {configured_base}")
    output_root = Path(parsed.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    plan = {"server_start_timeout": 900}
    server_log = output_root / "owned-vllm-server.log"
    training_log = output_root / "owned-training.log"
    with (output_root / ".launch.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with _server(plan, config_path, adapter, server_log):
            command = ["bash", str(REPO_ROOT / "scripts/run_train_grpo.sh")]
            if args.resume_from_checkpoint:
                command += ["--resume-from-checkpoint", args.resume_from_checkpoint]
            _command(command, env=_environment(config_path), log=training_log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
