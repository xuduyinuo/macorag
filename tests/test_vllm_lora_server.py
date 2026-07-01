from __future__ import annotations

from rl_training.vllm_lora_server import parse_server_args


def test_parse_server_args_requires_lora_identity() -> None:
    args = parse_server_args(
        [
            "--model",
            "model/Qwen2.5-7B-Instruct",
            "--lora-name",
            "macorag_train",
            "--lora-int-id",
            "1",
            "--lora-adapter-path",
            "outputs/adapter",
        ]
    )

    assert args.model == "model/Qwen2.5-7B-Instruct"
    assert args.lora_name == "macorag_train"
    assert args.lora_int_id == 1
    assert args.lora_adapter_path == "outputs/adapter"
