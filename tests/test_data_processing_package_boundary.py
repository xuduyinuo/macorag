from __future__ import annotations

import importlib
from pathlib import Path


def test_data_processing_owns_foundational_modules():
    for module_name in (
        "io_utils",
        "schemas",
        "retrieval",
        "retrieval_cli",
        "generate_teacher_sft",
    ):
        importlib.import_module(f"data_processing.{module_name}")


def test_data_processing_absorbs_legacy_data_modules():
    source_root = Path(__file__).resolve().parents[1] / "src"

    assert not (source_root / "retrieval_env").exists()
    assert not (source_root / "sft_data_generation").exists()


def test_data_processing_source_has_no_macorag_imports():
    source_root = Path(__file__).resolve().parents[1] / "src" / "data_processing"
    for source_path in source_root.glob("*.py"):
        source = source_path.read_text(encoding="utf-8")
        assert "macorag." not in source
