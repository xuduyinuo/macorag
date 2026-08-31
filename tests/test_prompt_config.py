from __future__ import annotations

from pathlib import Path

import pytest

from prompt_config import (
    EXPECTED_PROMPT_CONTRACT_VERSION,
    load_prompt_contract,
    load_system_prompt,
    system_prompt_for,
)


ROLES = ("query_retriever", "evidence_updater", "answer_generator")


def test_repository_prompt_contract_is_complete_and_versioned() -> None:
    contract = load_prompt_contract()

    assert contract.version == EXPECTED_PROMPT_CONTRACT_VERSION == "macorag-rag-v2"
    assert set(contract.system_prompts) == set(ROLES)
    assert all(contract.system_prompts[role].strip() for role in ROLES)
    assert len(contract.fingerprint) == 64
    assert contract.source_path.name == "prompts.yml"


def test_prompt_contract_fingerprint_is_semantic_and_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text(
        """prompt_contract_version: macorag-rag-v2
system_prompts:
  query_retriever: query
  evidence_updater: evidence
  answer_generator: answer
instructions:
  answer:
    normal: continue
    final: finish
""",
        encoding="utf-8",
    )
    second.write_text(
        """instructions: {answer: {final: finish, normal: continue}}
system_prompts: {answer_generator: answer, evidence_updater: evidence, query_retriever: query}
prompt_contract_version: macorag-rag-v2
""",
        encoding="utf-8",
    )

    assert load_prompt_contract(first).fingerprint == load_prompt_contract(second).fingerprint


@pytest.mark.parametrize(
    "payload, message",
    [
        ("prompt_contract_version: old\nsystem_prompts: {}\ninstructions: {}\n", "Unsupported prompt contract"),
        (
            "prompt_contract_version: macorag-rag-v2\nsystem_prompts: {query_retriever: q}\ninstructions: {}\n",
            "missing system prompts",
        ),
    ],
)
def test_prompt_contract_rejects_invalid_contract(tmp_path: Path, payload: str, message: str) -> None:
    path = tmp_path / "prompts.yml"
    path.write_text(payload, encoding="utf-8")

    with pytest.raises(SystemExit, match=message):
        load_prompt_contract(path)


def test_compatibility_system_prompt_is_answer_role_prompt() -> None:
    contract = load_prompt_contract()

    assert load_system_prompt() == contract.system_prompts["answer_generator"]
    assert system_prompt_for("query_retriever", contract) == contract.system_prompts["query_retriever"]


def test_system_prompt_for_rejects_unknown_role() -> None:
    with pytest.raises(ValueError, match="Unknown prompt role"):
        system_prompt_for("planner")
