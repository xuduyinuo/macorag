from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_PROMPT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "prompts.yml"


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise SystemExit("PyYAML is required to load prompt config.") from exc
    if not path.exists():
        raise SystemExit(f"Prompt config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid prompt config format at {path}: expected a mapping.")
    return payload


def load_system_prompt(path: str | Path | None = None) -> str:
    prompt_path = Path(path) if path is not None else DEFAULT_PROMPT_CONFIG_PATH
    payload = _load_yaml(prompt_path)
    system_prompt = str(payload.get("system_prompt") or "").strip()
    if not system_prompt:
        raise SystemExit(f"Prompt config missing non-empty system_prompt: {prompt_path}")
    return system_prompt


DEFAULT_SYSTEM_PROMPT = load_system_prompt()
