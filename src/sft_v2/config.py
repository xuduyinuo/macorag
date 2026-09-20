from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TeacherConfig:
    source_path: Path
    output_dir: Path
    corpus_path: Path
    index_path: Path
    index_manifest_path: Path
    corpus_offsets_path: Path
    retrieval_model_path: str
    prompt_path: Path
    dataset_aliases: dict[str, str] = field(default_factory=dict)
    candidate_limits_by_dataset: dict[str, int] = field(default_factory=dict)
    accepted_target_total: int = 2200
    train_target_total: int | None = None
    validation_target_total: int = 200
    retrieval_device: str = "cpu"
    retrieval_batch_size: int = 8
    retrieval_batch_wait_ms: int = 1000
    retrieval_max_length: int = 512
    retrieval_top_k: int = 5
    faiss_mmap: bool = True
    api_base: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    teacher_model: str = "deepseek-flash"
    thinking: str = "disabled"
    temperature: float = 0.2
    max_tokens: int = 1200
    request_timeout_seconds: int = 180
    request_retries: int = 4
    role_validation_retries: int = 2
    retry_base_seconds: float = 2.0
    max_rounds: int = 4
    sample_workers: int = 8
    seed: int = 42
    shuffle_source_examples: bool = True
    shuffle_retrieved_passages: bool = True
    resume: bool = True
    retry_failed: bool = True
    skip_filtered_on_resume: bool = True

    @property
    def resolved_train_target_total(self) -> int:
        if self.train_target_total is None:
            return self.accepted_target_total - self.validation_target_total
        return int(self.train_target_total)

    @property
    def reserve_target_total(self) -> int:
        return (
            self.accepted_target_total
            - self.resolved_train_target_total
            - self.validation_target_total
        )


def _require(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required config value: {key}")
    return value


def load_config(path: str | Path) -> TeacherConfig:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required for SFT-v2 configuration") from exc

    config_path = Path(path).expanduser().resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    path_fields = {
        name: Path(str(_require(payload, name))).expanduser().resolve()
        for name in (
            "source_path",
            "output_dir",
            "corpus_path",
            "index_path",
            "index_manifest_path",
            "corpus_offsets_path",
            "prompt_path",
        )
    }
    known = {field.name for field in TeacherConfig.__dataclass_fields__.values()}
    unknown = sorted(set(payload) - known - {"schema_version"})
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")
    values = {key: value for key, value in payload.items() if key in known}
    values.update(path_fields)
    config = TeacherConfig(**values)
    validate_config(config)
    return config


def validate_config(config: TeacherConfig) -> None:
    if config.teacher_model != "deepseek-flash":
        raise ValueError("SFT-v2 teacher_model must be the current DeepSeek V4.1 Flash ID: deepseek-flash")
    if config.thinking not in {"enabled", "disabled"}:
        raise ValueError("thinking must be enabled or disabled")
    for name in ("retrieval_batch_size", "retrieval_max_length", "retrieval_top_k", "max_rounds", "sample_workers"):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} must be positive")
    if int(config.retrieval_batch_wait_ms) < 0:
        raise ValueError("retrieval_batch_wait_ms must be non-negative")
    if config.request_retries <= 0 or config.request_timeout_seconds <= 0:
        raise ValueError("request retry and timeout values must be positive")
    if int(config.role_validation_retries) < 0:
        raise ValueError("role_validation_retries must be non-negative")
    if not config.dataset_aliases:
        raise ValueError("dataset_aliases must not be empty")
    for name in ("shuffle_source_examples", "shuffle_retrieved_passages"):
        if not isinstance(getattr(config, name), bool):
            raise ValueError(f"{name} must be a boolean")
    invalid_limits = {key: value for key, value in config.candidate_limits_by_dataset.items() if int(value) <= 0}
    if invalid_limits:
        raise ValueError(f"candidate limits must be positive: {invalid_limits}")
    candidate_total = sum(int(value) for value in config.candidate_limits_by_dataset.values())
    if not 0 < config.accepted_target_total <= candidate_total:
        raise ValueError("accepted_target_total must be positive and no larger than the candidate pool")
    if not 0 < config.validation_target_total < config.accepted_target_total:
        raise ValueError("validation_target_total must be between 1 and accepted_target_total - 1")
    if config.resolved_train_target_total <= 0:
        raise ValueError("train_target_total must be positive")
    if config.reserve_target_total < 0:
        raise ValueError(
            "train_target_total + validation_target_total must not exceed accepted_target_total"
        )
