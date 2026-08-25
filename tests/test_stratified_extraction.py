from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from data_processing.extract_stratified_datasets import load_extraction_config, main
from data_processing.io_utils import write_jsonl
from data_processing.stratified_extraction import (
    SelectionResult,
    derive_seed,
    eligibility_error,
    extract_pair,
    normalize_question,
    required_doc_ids,
    scope_corpus,
    select_rows,
    sha256_file,
    stratum_key,
    validate_dataset_output,
    write_dataset_output,
)


def make_row(
    dataset: str,
    split: str,
    qid: str,
    stratum: str | None,
    *,
    level: str | None = None,
    question: str | None = None,
) -> dict:
    row = {
        "qid": qid,
        "dataset": dataset,
        "split": split,
        "question": question or f"Question for {qid}?",
        "answer": f"answer-{qid}",
        "answer_aliases": [],
        "question_type": stratum,
        "hop_count": 2,
        "supporting_facts": [
            {
                "doc_id": f"{dataset}:support:{qid}",
                "title": f"Support {qid}",
                "text": f"Evidence for {qid}.",
            }
        ],
        "evidence_chain": [],
        "context_doc_ids": [f"{dataset}:support:{qid}", f"{dataset}:distractor:{qid}"],
        "usable_for_sft": True,
        "usable_for_retrieval_eval": True,
        "quality_flags": [],
        "metadata": {},
    }
    if level is not None:
        row["metadata"]["level"] = level
    return row


def test_normalize_question_is_case_punctuation_and_space_stable() -> None:
    assert normalize_question("  Who's   Alice?  ") == "who s alice"


def test_training_requires_both_flags_and_empty_quality_flags() -> None:
    row = make_row("2wiki", "train", "q1", "inference")
    assert eligibility_error(row, split="train") is None

    row["usable_for_retrieval_eval"] = False
    assert eligibility_error(row, split="train") == "not_usable_for_retrieval_eval"

    row["usable_for_retrieval_eval"] = True
    row["quality_flags"] = ["missing_supporting_fact_text"]
    assert eligibility_error(row, split="train") == "quality_flags"


def test_evaluation_requires_dev_and_retrieval_usability_only() -> None:
    row = make_row("hotpotqa", "dev", "q1", "bridge", level="hard")
    row["usable_for_sft"] = False
    assert eligibility_error(row, split="dev") is None

    row["split"] = "train"
    assert eligibility_error(row, split="dev") == "wrong_split"


def test_eligibility_rejects_invalid_required_fields_in_stable_order() -> None:
    row = make_row("2wiki", "train", "q1", "inference")
    row["qid"] = ""
    row["answer"] = ""
    assert eligibility_error(row, split="train") == "missing_qid"

    row["qid"] = "q1"
    assert eligibility_error(row, split="train") == "missing_answer"

    row["answer"] = "answer"
    row["supporting_facts"] = None
    assert eligibility_error(row, split="train") == "invalid_supporting_facts"


def test_dataset_specific_stratum_keys() -> None:
    assert stratum_key("2wiki", make_row("2wiki", "train", "w", "inference")) == "inference"
    assert (
        stratum_key("hotpotqa", make_row("hotpotqa", "train", "h", "bridge", level="hard"))
        == "hard/bridge"
    )
    assert stratum_key("musique", make_row("musique", "train", "4hop2__1_2", None)) == "4hop2"

    with pytest.raises(ValueError, match="Unsupported dataset"):
        stratum_key("unknown", make_row("unknown", "train", "q", None))


def test_derived_seed_is_stable_and_partitioned() -> None:
    first = derive_seed(20260826, "2wiki", "train", "inference")
    assert first == derive_seed(20260826, "2wiki", "train", "inference")
    assert first != derive_seed(20260826, "2wiki", "dev", "inference")


def _write_selection_source(path: Path) -> None:
    rows = [
        make_row("2wiki", "train", f"c{i}", "comparison") for i in range(8)
    ] + [make_row("2wiki", "train", f"i{i}", "inference") for i in range(6)]
    rows.append(
        make_row(
            "2wiki",
            "train",
            "duplicate-c0",
            "comparison",
            question=rows[0]["question"].upper(),
        )
    )
    invalid = make_row("2wiki", "train", "invalid", "comparison")
    invalid["quality_flags"] = ["bad"]
    rows.append(invalid)
    write_jsonl(path, rows)


def test_select_rows_fulfills_quota_deduplicates_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "2wiki_train.jsonl"
    _write_selection_source(source)
    kwargs = {
        "source_path": source,
        "dataset": "2wiki",
        "split": "train",
        "quotas": {"comparison": 3, "inference": 2},
        "seed": 20260826,
    }

    first = select_rows(**kwargs)
    second = select_rows(**kwargs)

    assert first == second
    assert first.quota_actual == {"comparison": 3, "inference": 2}
    assert len(first.rows) == 5
    assert len(first.source_indices) == 5
    assert len(set(first.qids)) == 5
    assert len({normalize_question(row["question"]) for row in first.rows}) == 5
    assert first.excluded_by_reason == {"duplicate_question": 1, "quality_flags": 1}


def test_select_rows_uses_seed_to_change_selection(tmp_path: Path) -> None:
    source = tmp_path / "2wiki_train.jsonl"
    _write_selection_source(source)
    common = {
        "source_path": source,
        "dataset": "2wiki",
        "split": "train",
        "quotas": {"comparison": 3, "inference": 2},
    }
    selections = {
        tuple(select_rows(**common, seed=seed).qids)
        for seed in range(1, 8)
    }
    assert len(selections) > 1


def test_select_rows_rejects_underfilled_stratum(tmp_path: Path) -> None:
    source = tmp_path / "2wiki_train.jsonl"
    write_jsonl(source, [make_row("2wiki", "train", "q", "inference")])

    with pytest.raises(ValueError, match=r"inference.*required=2.*available=1"):
        select_rows(
            source_path=source,
            dataset="2wiki",
            split="train",
            quotas={"inference": 2},
            seed=20260826,
        )


def test_test_fixture_rows_are_json_serializable() -> None:
    json.dumps(make_row("2wiki", "train", "q", "comparison"))


def make_doc(doc_id: str) -> dict:
    return {
        "doc_id": doc_id,
        "dataset": doc_id.split(":", 1)[0],
        "title": f"Title {doc_id}",
        "text": f"Text for {doc_id}",
        "sentences": [f"Text for {doc_id}"],
        "metadata": {},
    }


def test_required_doc_ids_unions_context_and_support() -> None:
    row = make_row("2wiki", "train", "q", "inference")
    row["context_doc_ids"] = ["d1", "d2", "d1"]
    row["supporting_facts"] = [
        {"doc_id": "d3", "text": "gold"},
        {"doc_id": "", "text": "ignored"},
    ]
    assert required_doc_ids([row]) == {"d1", "d2", "d3"}


def test_scope_corpus_is_exact_and_preserves_source_order(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(corpus, [make_doc("d2"), make_doc("unused"), make_doc("d1")])

    scoped = scope_corpus(corpus, {"d1", "d2"})

    assert [row["doc_id"] for row in scoped] == ["d2", "d1"]


def test_scope_corpus_rejects_missing_required_doc(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(corpus, [make_doc("d1")])

    with pytest.raises(ValueError, match="missing required corpus docs: missing"):
        scope_corpus(corpus, {"d1", "missing"})


def test_scope_corpus_rejects_duplicate_required_doc(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    write_jsonl(corpus, [make_doc("d1"), make_doc("d1")])

    with pytest.raises(ValueError, match="duplicate corpus doc_id: d1"):
        scope_corpus(corpus, {"d1"})


def test_sha256_file_streams_stable_digest(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    path.write_bytes(b"macorag\n")
    assert sha256_file(path) == "f0b320aa25a7cb197a7f0b4f53f5f68d2f3368ffb01051a9347b2828c1cb5038"


def _selected_output_fixture(tmp_path: Path) -> tuple[SelectionResult, Path, Path]:
    rows = [
        make_row("2wiki", "train", "i1", "inference"),
        make_row("2wiki", "train", "i2", "inference"),
    ]
    examples = tmp_path / "repo" / "data" / "processed" / "2wiki" / "2wiki_train.jsonl"
    corpus = examples.parent / "corpus.jsonl"
    write_jsonl(examples, rows)
    doc_ids = sorted(required_doc_ids(rows))
    write_jsonl(corpus, [make_doc(doc_id) for doc_id in doc_ids] + [make_doc("2wiki:unused")])
    selection = SelectionResult(
        rows=rows,
        source_indices=[4, 9],
        qids=["i1", "i2"],
        quota_actual={"inference": 2},
        eligible_count=8,
        excluded_by_reason={"quality_flags": 1},
    )
    return selection, examples, corpus


def test_write_and_validate_dataset_output_records_contract(tmp_path: Path) -> None:
    selection, examples, corpus = _selected_output_fixture(tmp_path)
    output_root = tmp_path / "repo" / "data" / "out"

    summary = write_dataset_output(
        output_root=output_root,
        repo_root=tmp_path / "repo",
        dataset="2wiki",
        split="train",
        selection=selection,
        source_examples=examples,
        source_corpus=corpus,
        quotas={"inference": 2},
        seed=20260826,
    )

    assert summary["output_examples"] == "2wiki/2wiki_train.jsonl"
    assert summary["output_corpus"] == "2wiki/corpus.jsonl"
    assert summary["source_examples"] == "data/processed/2wiki/2wiki_train.jsonl"
    assert summary["actual_quota"] == {"inference": 2}
    assert summary["required_corpus_count"] == 4
    assert len(summary["output_sha256"]["examples"]) == 64

    audit = validate_dataset_output(
        output_root / "2wiki",
        dataset="2wiki",
        split="train",
        quotas={"inference": 2},
    )
    assert audit["example_count"] == 2
    assert audit["unique_qid_count"] == 2
    assert audit["unique_question_count"] == 2
    assert audit["required_corpus_count"] == audit["corpus_count"] == 4
    assert audit["actual_quota"] == {"inference": 2}


def test_validator_detects_tampered_output(tmp_path: Path) -> None:
    selection, examples, corpus = _selected_output_fixture(tmp_path)
    output_root = tmp_path / "repo" / "data" / "out"
    write_dataset_output(
        output_root=output_root,
        repo_root=tmp_path / "repo",
        dataset="2wiki",
        split="train",
        selection=selection,
        source_examples=examples,
        source_corpus=corpus,
        quotas={"inference": 2},
        seed=20260826,
    )
    with (output_root / "2wiki" / "2wiki_train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(selection.rows[0]))
        handle.write("\n")

    with pytest.raises(ValueError, match="example SHA256 mismatch"):
        validate_dataset_output(
            output_root / "2wiki",
            dataset="2wiki",
            split="train",
            quotas={"inference": 2},
        )


def _dataset_stratum(dataset: str) -> tuple[str | None, str | None, str]:
    if dataset == "2wiki":
        return "inference", None, "inference"
    if dataset == "hotpotqa":
        return "bridge", "hard", "hard/bridge"
    return None, None, "2hop"


def _write_pair_sources(repo_root: Path, *, overlap_question: bool = False) -> None:
    source_root = repo_root / "data" / "processed"
    for dataset in ("2wiki", "hotpotqa", "musique"):
        stratum, level, _ = _dataset_stratum(dataset)
        corpus_rows = []
        for split in ("train", "dev"):
            qid = f"{dataset}-{split}"
            if dataset == "musique":
                qid = f"2hop__{split}_1"
            question = f"{dataset} {split} question?"
            if overlap_question and dataset == "2wiki":
                question = "Same normalized question?" if split == "train" else "same normalized question!"
            row = make_row(dataset, split, qid, stratum, level=level, question=question)
            path = source_root / dataset / f"{dataset}_{split}.jsonl"
            write_jsonl(path, [row])
            corpus_rows.extend(make_doc(doc_id) for doc_id in required_doc_ids([row]))
        unique_docs = {row["doc_id"]: row for row in corpus_rows}
        write_jsonl(source_root / dataset / "corpus.jsonl", unique_docs.values())


def _pair_configs(tmp_path: Path, *, overlap_question: bool = False) -> tuple[dict, dict]:
    repo_root = tmp_path / "repo"
    _write_pair_sources(repo_root, overlap_question=overlap_question)
    datasets = {
        dataset: {"quotas": {_dataset_stratum(dataset)[2]: 1}}
        for dataset in ("2wiki", "hotpotqa", "musique")
    }
    common = {
        "schema_version": 1,
        "repo_root": repo_root,
        "source_root": repo_root / "data" / "processed",
        "seed": 20260826,
        "expected_total": 1,
        "datasets": datasets,
    }
    train = {**common, "split": "train", "output_root": repo_root / "data" / "train_out"}
    evaluation = {**common, "split": "dev", "output_root": repo_root / "data" / "eval_out"}
    return train, evaluation


def test_extract_pair_publishes_both_after_zero_overlap_audit(tmp_path: Path) -> None:
    train, evaluation = _pair_configs(tmp_path)

    manifest = extract_pair(train, evaluation)

    assert manifest["overlap_audit"] == {"qid_count": 0, "normalized_question_count": 0}
    assert train["output_root"].is_dir()
    assert evaluation["output_root"].is_dir()
    assert not list((tmp_path / "repo" / "data").glob("*.staging-*"))
    assert (train["output_root"] / "extraction_manifest.json").is_file()
    assert (evaluation["output_root"] / "extraction_manifest.json").is_file()


def test_extract_pair_refuses_existing_target(tmp_path: Path) -> None:
    train, evaluation = _pair_configs(tmp_path)
    train["output_root"].mkdir(parents=True)

    with pytest.raises(FileExistsError, match="target already exists"):
        extract_pair(train, evaluation)


def test_extract_pair_never_publishes_overlapping_questions(tmp_path: Path) -> None:
    train, evaluation = _pair_configs(tmp_path, overlap_question=True)

    with pytest.raises(ValueError, match="normalized-question overlap"):
        extract_pair(train, evaluation)

    assert not train["output_root"].exists()
    assert not evaluation["output_root"].exists()


def test_extract_pair_removes_staging_after_handled_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train, evaluation = _pair_configs(tmp_path)

    def fail_validation(*args, **kwargs):
        raise ValueError("forced validation failure")

    monkeypatch.setattr(
        "data_processing.stratified_extraction.validate_dataset_output", fail_validation
    )
    with pytest.raises(ValueError, match="forced validation failure"):
        extract_pair(train, evaluation)

    assert not list((tmp_path / "repo" / "data").glob("*.staging-*"))
    assert not train["output_root"].exists()
    assert not evaluation["output_root"].exists()


def _write_config(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_pair_config_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "repo"
    _write_pair_sources(repo_root)
    datasets = {
        dataset: {"quotas": {_dataset_stratum(dataset)[2]: 1}}
        for dataset in ("2wiki", "hotpotqa", "musique")
    }
    config_dir = repo_root / "config"
    train_path = config_dir / "train.yml"
    eval_path = config_dir / "eval.yml"
    _write_config(
        train_path,
        {
            "schema_version": 1,
            "source_root": "data/processed",
            "output_root": "data/train_out",
            "split": "train",
            "seed": 20260826,
            "expected_total": 1,
            "datasets": datasets,
        },
    )
    _write_config(
        eval_path,
        {
            "schema_version": 1,
            "source_root": "data/processed",
            "output_root": "data/eval_out",
            "split": "dev",
            "seed": 20260826,
            "expected_total": 1,
            "datasets": datasets,
        },
    )
    return repo_root, train_path, eval_path


def test_config_rejects_wrong_quota_total(tmp_path: Path) -> None:
    repo_root, train_path, _ = _write_pair_config_files(tmp_path)
    payload = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    payload["expected_total"] = 2
    _write_config(train_path, payload)

    with pytest.raises(ValueError, match="quota total 1 != expected_total 2"):
        load_extraction_config(train_path, repo_root=repo_root)


def test_cli_dry_run_prints_contract_without_writing(tmp_path: Path, capsys) -> None:
    repo_root, train_path, eval_path = _write_pair_config_files(tmp_path)

    rc = main(
        [
            "--train-config",
            str(train_path),
            "--eval-config",
            str(eval_path),
            "--repo-root",
            str(repo_root),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["train"]["total_quota"] == 3
    assert payload["evaluation"]["total_quota"] == 3
    assert not (repo_root / "data" / "train_out").exists()


def test_cli_audits_existing_pair(tmp_path: Path, capsys) -> None:
    repo_root, train_path, eval_path = _write_pair_config_files(tmp_path)
    train = load_extraction_config(train_path, repo_root=repo_root)
    evaluation = load_extraction_config(eval_path, repo_root=repo_root)
    extract_pair(train, evaluation)

    rc = main(
        [
            "--train-config",
            str(train_path),
            "--eval-config",
            str(eval_path),
            "--repo-root",
            str(repo_root),
            "--audit-existing",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["train_total"] == 3
    assert payload["evaluation_total"] == 3
    assert payload["overlap_audit"] == {"qid_count": 0, "normalized_question_count": 0}


def test_production_configs_match_approved_contract() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    train = load_extraction_config(
        repo_root / "config" / "extract_stratified_train_v2.yml", repo_root=repo_root
    )
    evaluation = load_extraction_config(
        repo_root / "config" / "extract_stratified_eval_v2.yml", repo_root=repo_root
    )

    assert train["seed"] == evaluation["seed"] == 20260826
    assert (train["split"], evaluation["split"]) == ("train", "dev")
    assert (train["expected_total"], evaluation["expected_total"]) == (2000, 1000)
    assert train["datasets"]["2wiki"]["quotas"] == {
        "compositional": 831,
        "comparison": 486,
        "bridge_comparison": 440,
        "inference": 243,
    }
    assert evaluation["datasets"]["hotpotqa"]["quotas"] == {
        "hard/bridge": 798,
        "hard/comparison": 202,
    }
    assert evaluation["datasets"]["musique"]["quotas"] == {
        "2hop": 518,
        "3hop1": 235,
        "3hop2": 79,
        "4hop1": 102,
        "4hop2": 27,
        "4hop3": 39,
    }


def test_opt_in_downstream_configs_use_only_v2_roots() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    train_retrieval = yaml.safe_load(
        (repo_root / "config" / "build_retrieval_train_stratified_v2_e5.yml").read_text()
    )
    eval_retrieval = yaml.safe_load(
        (repo_root / "config" / "build_retrieval_eval_stratified_v2_e5.yml").read_text()
    )
    train_runtime = yaml.safe_load(
        (repo_root / "config" / "train_grpo_stratified_v2.yml").read_text()
    )
    eval_runtime = yaml.safe_load(
        (repo_root / "config" / "eval_macorag_stratified_v2.yml").read_text()
    )

    assert train_retrieval["data_root"] == "data/rl_train_2000_stratified_v2"
    assert train_retrieval["retrieval_root"] == "data/rl_train_2000_stratified_v2_e5_faiss"
    assert eval_retrieval["data_root"] == "data/eval_1000_stratified_v2"
    assert eval_retrieval["retrieval_root"] == "data/eval_1000_stratified_v2_e5_faiss"
    assert train_runtime["rl_data_root"] == "data/rl_train_2000_stratified_v2"
    assert train_runtime["retrieval_root"] == "data/rl_train_2000_stratified_v2_e5_faiss"
    assert train_runtime["max_samples"] == 2000
    assert eval_runtime["data_root"] == "data/eval_1000_stratified_v2"
    assert eval_runtime["retrieval_root"] == "data/eval_1000_stratified_v2_e5_faiss"
    assert eval_runtime["max_samples"] is None
