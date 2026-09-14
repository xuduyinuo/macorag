from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any


def _save_policy_adapter(model: Any, path: Path) -> None:
    kwargs: dict[str, Any] = {}
    policy_adapter = getattr(model, "_mappo_policy_adapter_name", None)
    if policy_adapter is not None:
        kwargs["selected_adapters"] = [policy_adapter]
    model.save_pretrained(path, **kwargs)


def _copy_prompt_contract(path: Path, config: Any) -> None:
    initialization_paths = [
        Path(value)
        for value in (
            getattr(config, "resume_from_checkpoint", ""),
            getattr(config, "sft_adapter_path", ""),
        )
        if value
    ]
    candidates = [
        candidate
        for root in initialization_paths
        for candidate in (root / "prompt_contract.json", root / "actor" / "prompt_contract.json")
    ]
    source = next((candidate for candidate in candidates if candidate.is_file()), None)
    if source is None:
        searched = ", ".join(str(candidate) for candidate in candidates) or "<no initialization path>"
        raise FileNotFoundError(f"MAPPO prompt_contract.json not found; searched: {searched}")
    shutil.copy2(source, path / "prompt_contract.json")


def save_checkpoint(
    path: Path,
    *,
    actor: Any,
    critic: Any,
    actor_optimizer: Any,
    critic_optimizer: Any,
    global_step: int,
    epoch: int,
    sample_offset: int,
    config: Any,
) -> None:
    import numpy as np
    import torch
    path.mkdir(parents=True, exist_ok=True)
    actor_dir = path / "actor"
    _save_policy_adapter(actor.model, actor_dir)
    actor.tokenizer.save_pretrained(actor_dir)
    _copy_prompt_contract(actor_dir, config)
    torch.save(critic.state_dict(), path / "critic.pt")
    torch.save({
        "actor_optimizer": actor_optimizer.state_dict(),
        "critic_optimizer": critic_optimizer.state_dict(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }, path / "training_state.pt")
    state = {
        "schema_version": 1,
        "algorithm": "mappo",
        "global_step": global_step,
        "epoch": epoch,
        "sample_offset": sample_offset,
        "config": config.to_dict(),
        "generation_counter": int(getattr(actor, "generation_counter", 0)),
    }
    (path / "trainer_state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (path / "COMPLETE").write_text("ok\n", encoding="utf-8")


def save_actor_export(
    path: Path, *, actor: Any, metadata: dict[str, Any], config: Any,
) -> None:
    """Export the current policy adapter in the same simple shape used for evaluation."""
    path.mkdir(parents=True, exist_ok=True)
    _save_policy_adapter(actor.model, path)
    actor.tokenizer.save_pretrained(path)
    _copy_prompt_contract(path, config)
    (path / "validation_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def restore_checkpoint(path: Path, *, critic: Any, actor_optimizer: Any, critic_optimizer: Any) -> dict[str, int]:
    import numpy as np
    import torch
    if not (path / "COMPLETE").is_file():
        raise RuntimeError(f"Incomplete MAPPO checkpoint: {path}")
    critic.load_state_dict(torch.load(path / "critic.pt", map_location=critic.device, weights_only=True))
    payload = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    actor_optimizer.load_state_dict(payload["actor_optimizer"])
    critic_optimizer.load_state_dict(payload["critic_optimizer"])
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    state = json.loads((path / "trainer_state.json").read_text(encoding="utf-8"))
    if state.get("algorithm") != "mappo":
        raise RuntimeError(f"Checkpoint is not MAPPO: {path}")
    return {
        name: int(state.get(name, 0))
        for name in ("global_step", "epoch", "sample_offset", "generation_counter")
    }


def prune_checkpoints(run_dir: Path, keep: int) -> None:
    checkpoints = sorted(
        (path for path in run_dir.glob("checkpoint-*" ) if (path / "COMPLETE").is_file()),
        key=lambda path: int(path.name.rsplit("-", 1)[-1]),
    )
    for path in checkpoints[:-keep]:
        import shutil
        shutil.rmtree(path)
