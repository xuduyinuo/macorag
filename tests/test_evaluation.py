from __future__ import annotations

from pathlib import Path

import pytest

from evaluation.config import parse_args


def test_parse_eval_config_loads_yaml_and_cli_overrides(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text(
        "\n".join(
            [
                'model_path: "model/base"',
                'adapter_path: "outputs/grpo/adapter"',
                'data_root: "data/eval_1000"',
                'retrieval_root: "data/eval_1000_retrieval"',
                'output_dir: "outputs/eval"',
                'judge_model: "qwen-plus"',
                'judge_api_key_env: "DASHSCOPE_API_KEY"',
                "max_samples: 20",
                "max_rounds: 2",
                "retrieval_top_k: 4",
                'gpu_indices: "1"',
                "skip_judge: false",
            ]
        ),
        encoding="utf-8",
    )

    args = parse_args(["--config", str(config), "--max-samples", "3", "--skip-judge"])

    assert args.model_path == "model/base"
    assert args.adapter_path == "outputs/grpo/adapter"
    assert args.data_root == "data/eval_1000"
    assert args.retrieval_root == "data/eval_1000_retrieval"
    assert args.output_dir == "outputs/eval"
    assert args.judge_model == "qwen-plus"
    assert args.judge_api_key_env == "DASHSCOPE_API_KEY"
    assert args.max_samples == 3
    assert args.max_rounds == 2
    assert args.retrieval_top_k == 4
    assert args.gpu_indices == "1"
    assert args.skip_judge is True


def test_parse_eval_config_rejects_unknown_yaml_keys(tmp_path: Path) -> None:
    config = tmp_path / "evaluate_rag_model.yml"
    config.write_text("unknown_key: 1\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="Unknown evaluation config keys"):
        parse_args(["--config", str(config)])


def test_parse_eval_config_rejects_missing_explicit_config_with_equals(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_config.yml"

    with pytest.raises(SystemExit, match="Evaluation config not found"):
        parse_args([f"--config={missing}"])
