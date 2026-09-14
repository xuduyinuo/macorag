from __future__ import annotations

import os
from pathlib import Path
import subprocess

import yaml


ROOT = Path(__file__).resolve().parents[1]


def _make_adapter(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "prompt_contract.json").write_text("{}", encoding="utf-8")
    return path


def _yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "config" / name).read_text(encoding="utf-8"))


def test_v2_retraining_configs_freeze_shared_contract() -> None:
    teacher = _yaml("generate_teacher_sft.yml")
    sft = _yaml("train_sft.yml")
    grpo = _yaml("train_grpo.yml")
    evaluation = _yaml("eval_macorag.yml")

    assert teacher["output_dir"] == "data/sft/teacher_qwen_plus_trajectory_train_v2"
    assert teacher["retrieval_root"] == "data/trajectory_train_e5_faiss"
    assert teacher["retrieval_backend"] == "e5_faiss"
    assert teacher["embedding_model"] == "intfloat/e5-base-v2"
    assert teacher["target_valid_per_dataset"] == 1000
    assert teacher["validate_retrieval_contract"] is True

    assert sft["model_path"] == "model/Qwen2.5-7B-Instruct"
    assert sft["data_root"] == teacher["output_dir"]
    assert sft["require_teacher_provenance"] is True
    assert grpo["sft_adapter_path"] == "/path/to/sft_adapter"
    assert grpo["rl_data_root"] == "data/rl_train_2000_stratified_v2"
    assert grpo["retrieval_root"] == "data/rl_train_2000_stratified_v2_e5_faiss"
    assert grpo["max_samples"] == 2000
    assert grpo["data_sampling_strategy"] == "proportional_stratified"
    assert grpo["data_sampling_seed"] == 20260826
    assert grpo["bf16"] is True
    assert grpo["fp16"] is False
    assert grpo["attn_implementation"] == "flash_attention_2"
    assert evaluation["data_root"] == "data/eval_1000_stratified_v2"
    assert evaluation["retrieval_root"] == "data/eval_1000_stratified_v2_e5_faiss"

    for config in (teacher, sft, grpo, evaluation):
        assert config["max_rounds"] == 4
        assert config.get("retrieval_top_k", 5) == 5
        assert config["prompt_config_path"] == "config/prompts.yml"


def test_grpo_launchers_require_configured_sft_adapter_path(tmp_path: Path) -> None:
    config = _yaml("train_grpo.yml")
    config["sft_adapter_path"] = ""
    config_path = tmp_path / "train_grpo.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = os.environ.copy()
    env.pop("SFT_ADAPTER_PATH", None)
    env["MACORAG_LAUNCH_DRY_RUN"] = "1"
    env["CONFIG_PATH"] = str(config_path)

    for script in ("run_train_grpo.sh", "run_grpo_vllm_server.sh"):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / script)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        assert result.returncode == 2
        assert "sft_adapter_path" in result.stderr


def test_grpo_launchers_use_configured_sft_adapter_path(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path / "chosen-sft")
    config = _yaml("train_grpo.yml")
    config["sft_adapter_path"] = str(adapter)
    config_path = tmp_path / "train_grpo.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = os.environ.copy()
    env["MACORAG_LAUNCH_DRY_RUN"] = "1"
    env["CONFIG_PATH"] = str(config_path)
    env.pop("SFT_ADAPTER_PATH", None)

    for script in ("run_train_grpo.sh", "run_grpo_vllm_server.sh"):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / script)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        assert str(adapter) in result.stdout


def test_grpo_training_launcher_uses_selected_python_for_distributed_run() -> None:
    content = (ROOT / "scripts" / "run_train_grpo.sh").read_text(encoding="utf-8")

    assert '"${PYTHON:-python}" -m torch.distributed.run' in content
    assert "\ntorchrun " not in content


def test_eval_vllm_launcher_requires_and_uses_configured_adapter_path(tmp_path: Path) -> None:
    config = _yaml("eval_macorag.yml")
    config["adapter_path"] = ""
    config_path = tmp_path / "eval_macorag.yml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    env = os.environ.copy()
    env.pop("ADAPTER_PATH", None)
    env["MACORAG_LAUNCH_DRY_RUN"] = "1"
    env["CONFIG_PATH"] = str(config_path)
    missing = subprocess.run(
        ["bash", str(ROOT / "scripts/eval_vllm_server.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert missing.returncode == 2
    assert "adapter_path" in missing.stderr

    adapter = _make_adapter(tmp_path / "chosen-eval")
    config["adapter_path"] = str(adapter)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    selected = subprocess.run(
        ["bash", str(ROOT / "scripts/eval_vllm_server.sh")],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert str(adapter) in selected.stdout


def test_adapter_launchers_do_not_auto_discover_latest_runs() -> None:
    for script in ("run_train_grpo.sh", "run_grpo_vllm_server.sh", "eval_vllm_server.sh"):
        content = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "AUTO_FROM_" not in content
        assert "glob(" not in content


def test_full_trajectory_e5_build_config_is_separate() -> None:
    config = _yaml("retrieval_teacher.yml")
    assert config["data_root"] == "data/trajectory_train"
    assert config["retrieval_root"] == "data/trajectory_train_e5_faiss"
    assert config["embedding_model"] == "intfloat/e5-base-v2"
    assert config["retrieval_top_k"] == 5


def test_launchers_are_syntax_valid_and_support_shared_dry_run() -> None:
    scripts = (
        "build_retrieval.sh",
        "build_teacher_retrieval.sh",
        "generate_teacher_sft.sh",
        "run_train_sft.sh",
        "run_train_grpo.sh",
        "run_grpo_vllm_server.sh",
        "eval_macorag.sh",
        "eval_vllm_server.sh",
        "validate_pipeline.sh",
    )
    for script in scripts:
        subprocess.run(["bash", "-n", str(ROOT / "scripts" / script)], check=True)
        content = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert "CONFIG_PATH" in content
        assert "MACORAG_LAUNCH_DRY_RUN" in content


def test_fixed_grpo_evaluation_launcher_requires_stable_identity(tmp_path: Path) -> None:
    script = ROOT / "scripts" / "evaluate_grpo_fixed.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    base_env = {
        "PATH": "/usr/bin:/bin",
        "MACORAG_LAUNCH_DRY_RUN": "1",
    }
    missing = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=base_env,
        text=True,
        capture_output=True,
    )
    assert missing.returncode != 0
    assert "ADAPTER_LABEL" in missing.stderr

    output_dir = tmp_path / "sft-eval"
    adapter = _make_adapter(tmp_path / "fixed-sft")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env={
            **base_env,
            "ADAPTER_LABEL": "sft",
            "ADAPTER_PATH": str(adapter),
            "OUTPUT_DIR": str(output_dir),
            "PYTHON": "/data/conda/envs/macorag/bin/python",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert "config/eval_grpo_fixed.yml" in completed.stdout
    assert str(output_dir) not in completed.stderr


def test_teacher_retrieval_launcher_defaults_to_full_trajectory_config() -> None:
    script = ROOT / "scripts" / "build_teacher_retrieval.sh"
    completed = subprocess.run(
        ["bash", str(script)],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "MACORAG_LAUNCH_DRY_RUN": "1"},
    )

    assert "config/retrieval_teacher.yml" in completed.stdout


def test_superseded_pipeline_entrypoints_are_removed() -> None:
    version = "stratified_" + "v2"
    build_prefix = "build_" + "retrieval"
    obsolete = (
        f"config/train_grpo_{version}.yml",
        f"config/eval_macorag_{version}.yml",
        "config/extract_datasets.yml",
        f"config/extract_{version.removesuffix('_v2')}_train_v2.yml",
        f"config/extract_{version.removesuffix('_v2')}_eval_v2.yml",
        f"config/{build_prefix}.yml",
        f"config/{build_prefix}_eval_e5.yml",
        f"config/{build_prefix}_eval_{version}_e5.yml",
        f"config/{build_prefix}_train_e5.yml",
        f"config/{build_prefix}_train_{version}_e5.yml",
        f"config/{build_prefix}_trajectory_train_e5.yml",
        "scripts/run_train_grpo_resume_" + "3600.sh",
        "scripts/validate_retraining_" + "v2.sh",
    )

    assert not [path for path in obsolete if (ROOT / path).exists()]


def test_flash_attention_installer_freezes_source_build_contract() -> None:
    script = ROOT / "scripts" / "install_flash_attn.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    content = script.read_text(encoding="utf-8")

    required_fragments = (
        "flash_attn-2.8.3.post1.tar.gz",
        "55d5103ed846da8b56e0797acf4bde07dee4b1c7e8907fcfc6699c203030c348",
        'CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.9}"',
        "FLASH_ATTENTION_FORCE_BUILD=TRUE",
        "FLASH_ATTENTION_FORCE_CXX11_ABI=FALSE",
        "FLASH_ATTN_CUDA_ARCHS=80",
        "MAX_JOBS",
        "NVCC_THREADS",
        "--no-build-isolation",
        "--no-deps",
        "--no-cache-dir",
        "torch.__version__ == '2.6.0+cu124'",
        "torch._C._GLIBCXX_USE_CXX11_ABI is False",
        "flash_attn_func",
        "torch.bfloat16",
        "backward()",
    )
    for fragment in required_fragments:
        assert fragment in content

    assert "/data/conda/pip-cache" not in content
