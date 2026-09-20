from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
from typing import Any

from tqdm.auto import tqdm

from .config import TeacherConfig, load_config
from .deepseek_client import DeepSeekClient
from .prompts import load_prompt_contract
from .retrieval import UnifiedE5FaissRetriever
from .source import SourceExample, iter_jsonl, load_source_examples, source_summary
from .trajectory import TrajectoryGenerator


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _append_jsonl(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    handle.flush()


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _deterministic_example_order(
    examples: list[SourceExample], *, seed: int, enabled: bool
) -> list[SourceExample]:
    """Return a reproducibly shuffled global candidate stream.

    Hash ordering avoids mutable RNG state, so resume and concurrent generation
    see exactly the same candidate order for the same seed.
    """

    if not enabled:
        return list(examples)
    return sorted(
        examples,
        key=lambda item: hashlib.sha256(
            f"sft-v2-source\0{seed}\0{item.qid}".encode("utf-8")
        ).digest(),
    )


def _load_ordered_examples(
    config: TeacherConfig, *, limit: int | None
) -> list[SourceExample]:
    # Apply per-dataset candidate-pool limits first, then shuffle the complete
    # unified pool. ``--limit`` is intentionally applied after shuffling so a
    # smoke run does not always draw from the first dataset in the source file.
    examples = load_source_examples(
        config.source_path,
        aliases=config.dataset_aliases,
        limits=config.candidate_limits_by_dataset,
        total_limit=None,
    )
    examples = _deterministic_example_order(
        examples,
        seed=config.seed,
        enabled=config.shuffle_source_examples,
    )
    return examples[:limit] if limit is not None else examples


def _validate_assets(config: TeacherConfig) -> dict[str, Any]:
    required = [
        config.source_path,
        config.corpus_path,
        config.index_path,
        config.index_manifest_path,
        config.prompt_path,
        Path(config.retrieval_model_path),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing SFT-v2 assets: " + ", ".join(missing))
    manifest = json.loads(config.index_manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("retrieval_method")) != "e5":
        raise ValueError("Index manifest retrieval_method must be e5")
    if str(manifest.get("metric")) != "inner_product":
        raise ValueError("Index manifest metric must be inner_product")
    if str(manifest.get("pooling_method")) != "mean":
        raise ValueError("Index manifest pooling_method must be mean")
    if str(manifest.get("faiss_type")) != "Flat":
        raise ValueError("Index manifest faiss_type must be Flat")
    if manifest.get("l2_normalized") is False:
        raise ValueError("Index manifest reports l2_normalized=false, incompatible with E5 inner-product retrieval")
    if int(manifest.get("max_length", -1)) != config.retrieval_max_length:
        raise ValueError("Configured retrieval_max_length differs from the index manifest")
    if int(manifest.get("ntotal", 0)) <= 0 or int(manifest.get("dimension", 0)) <= 0:
        raise ValueError("Index manifest must contain positive ntotal and dimension")
    declared_paths = {
        "corpus_path": config.corpus_path,
        "index_path": config.index_path,
        "model_path": Path(config.retrieval_model_path),
    }
    mismatched_paths = [
        key
        for key, actual in declared_paths.items()
        if manifest.get(key) and Path(str(manifest[key])).resolve() != actual.resolve()
    ]
    if mismatched_paths:
        raise ValueError("Configured assets differ from index manifest: " + ", ".join(mismatched_paths))
    if manifest.get("corpus_bytes") and config.corpus_path.stat().st_size != int(manifest["corpus_bytes"]):
        raise ValueError("Corpus byte size differs from index manifest")
    if manifest.get("index_bytes") and config.index_path.stat().st_size != int(manifest["index_bytes"]):
        raise ValueError("Index byte size differs from index manifest")
    return manifest


def _load_statuses(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    statuses: dict[str, str] = {}
    for row in iter_jsonl(path):
        statuses[str(row.get("qid"))] = str(row.get("status"))
    return statuses


def _should_process(
    example: SourceExample,
    statuses: dict[str, str],
    accepted_qids: set[str],
    config: TeacherConfig,
) -> bool:
    if example.qid in accepted_qids:
        return False
    status = statuses.get(example.qid)
    if not config.resume or status is None or status == "accepted":
        # The accepted pool is authoritative. A ledger-only accepted record can
        # result from an interrupted two-file append and must be regenerated.
        return True
    if status == "filtered":
        return not config.skip_filtered_on_resume
    if status == "failed":
        return config.retry_failed
    return True


def _read_accepted_pool(config: TeacherConfig, *, prompt_fingerprint: str) -> list[dict[str, Any]]:
    path = config.output_dir / "accepted_sft.jsonl"
    rows = list(iter_jsonl(path)) if config.resume and path.is_file() else []
    if len(rows) > config.accepted_target_total:
        raise RuntimeError(
            f"Accepted pool {path} exceeds target: {len(rows)} > {config.accepted_target_total}"
        )
    seen: set[str] = set()
    for row in rows:
        qid = str(row.get("qid") or "")
        if not qid or qid in seen:
            raise RuntimeError(f"Accepted pool has a missing or duplicate qid: {qid!r}")
        if str(row.get("prompt_contract_fingerprint")) != prompt_fingerprint:
            raise RuntimeError(
                "Accepted pool prompt contract differs from the current SFT-v2 prompt contract"
            )
        seen.add(qid)
    return rows


def _accepted_behavior_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rounds = [len(row.get("trajectory") or []) for row in rows]
    answer_decisions = [
        turn.get("answer") or {}
        for row in rows
        for turn in (row.get("trajectory") or [])
    ]
    return {
        "one_round_trajectories": sum(value == 1 for value in rounds),
        "multi_round_trajectories": sum(value > 1 for value in rounds),
        "average_rounds": sum(rounds) / max(1, len(rounds)),
        "intermediate_can_answer_false_decisions": sum(
            decision.get("can_answer") is False for decision in answer_decisions
        ),
        "total_answer_decisions": len(answer_decisions),
    }


def _materialize_train_validation_reserve_splits(
    config: TeacherConfig,
    accepted_rows: list[dict[str, Any]],
    *,
    source_order: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if len(accepted_rows) != config.accepted_target_total:
        raise RuntimeError(
            f"Cannot split accepted pool: expected {config.accepted_target_total}, got {len(accepted_rows)}"
        )
    ranked = sorted(
        accepted_rows,
        key=lambda row: hashlib.sha256(
            f"{config.seed}\0{row['qid']}".encode("utf-8")
        ).digest(),
    )
    validation_qids = {
        str(row["qid"])
        for row in ranked[: config.validation_target_total]
    }
    train_start = config.validation_target_total
    train_end = train_start + config.resolved_train_target_total
    train_qids = {
        str(row["qid"])
        for row in ranked[train_start:train_end]
    }
    validation_rows = [
        {**row, "sft_split": "validation"}
        for row in accepted_rows
        if str(row["qid"]) in validation_qids
    ]
    train_rows = [
        {**row, "sft_split": "train"}
        for row in accepted_rows
        if str(row["qid"]) in train_qids
    ]
    reserve_rows = [
        {**row, "sft_split": "reserve"}
        for row in accepted_rows
        if str(row["qid"]) not in validation_qids
        and str(row["qid"]) not in train_qids
    ]
    train_rows.sort(key=lambda row: source_order[str(row["qid"])])
    validation_rows.sort(key=lambda row: source_order[str(row["qid"])])
    reserve_rows.sort(key=lambda row: source_order[str(row["qid"])])
    expected_validation = config.validation_target_total
    expected_total = config.accepted_target_total
    if (
        len(validation_rows) != expected_validation
        or len(train_rows) != config.resolved_train_target_total
        or len(reserve_rows) != config.reserve_target_total
        or len(train_rows) + len(validation_rows) + len(reserve_rows) != expected_total
    ):
        raise RuntimeError("Materialized SFT-v2 split counts do not match the configured contract")
    _write_jsonl_atomic(config.output_dir / "train_sft.jsonl", train_rows)
    _write_jsonl_atomic(config.output_dir / "validation_sft.jsonl", validation_rows)
    _write_jsonl_atomic(config.output_dir / "reserve_sft.jsonl", reserve_rows)
    return train_rows, validation_rows, reserve_rows


def _safe_generate(generator: TrajectoryGenerator, example: SourceExample) -> dict[str, Any]:
    try:
        return generator.generate(example)
    except BaseException as exc:
        return {
            "status": "failed",
            "qid": example.qid,
            "dataset": example.dataset,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def generate(config: TeacherConfig, *, limit: int | None = None, dry_run: bool = False) -> dict[str, Any]:
    print("[sft-v2:init] Validating assets and loading the shuffled source pool", file=sys.stderr, flush=True)
    manifest = _validate_assets(config)
    prompts = load_prompt_contract(config.prompt_path)
    examples = _load_ordered_examples(config, limit=limit)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = config.output_dir / "generation_ledger.jsonl"
    statuses = _load_statuses(ledger_path) if config.resume else {}
    accepted_rows = _read_accepted_pool(config, prompt_fingerprint=prompts.fingerprint)
    accepted_qids = {str(row["qid"]) for row in accepted_rows}
    source_order = {item.qid: index for index, item in enumerate(examples)}

    run_config = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_path": str(config.source_path),
        "source_summary": source_summary(examples),
        "corpus_path": str(config.corpus_path),
        "index_path": str(config.index_path),
        "index_manifest_path": str(config.index_manifest_path),
        "index_manifest": manifest,
        "index_manifest_fingerprint": _canonical_hash(manifest),
        "retrieval_model_path": config.retrieval_model_path,
        "retrieval_top_k": config.retrieval_top_k,
        "retrieval_query_batch_size": config.retrieval_batch_size,
        "retrieval_batch_wait_ms": config.retrieval_batch_wait_ms,
        "teacher_model": config.teacher_model,
        "thinking": config.thinking,
        "max_rounds": config.max_rounds,
        "sample_workers": config.sample_workers,
        "role_validation_retries": config.role_validation_retries,
        "prompt_path": str(prompts.source_path),
        "prompt_contract_version": prompts.version,
        "prompt_contract_fingerprint": prompts.fingerprint,
        "candidate_limits_by_dataset": config.candidate_limits_by_dataset,
        "source_order": (
            "deterministic_global_hash_shuffle"
            if config.shuffle_source_examples
            else "source_file_order"
        ),
        "shuffle_source_examples": config.shuffle_source_examples,
        "retrieval_candidate_order": (
            "deterministic_per_query_hash_shuffle_then_reindex"
            if config.shuffle_retrieved_passages
            else "faiss_score_order"
        ),
        "shuffle_retrieved_passages": config.shuffle_retrieved_passages,
        "shuffle_seed": config.seed,
        "accepted_target_total": config.accepted_target_total,
        "train_target_total": config.resolved_train_target_total,
        "validation_target_total": config.validation_target_total,
        "reserve_target_total": config.reserve_target_total,
        "acceptance_policy": "first_qualified_from_shuffled_unified_pool_without_dataset_quota",
        "resume": config.resume,
        "dry_run": dry_run,
    }
    _write_json(config.output_dir / "run_config.json", run_config)
    print(
        f"[sft-v2:init] Source ready: {len(examples):,} candidates; "
        f"already accepted={len(accepted_rows):,}/{config.accepted_target_total:,}",
        file=sys.stderr,
        flush=True,
    )

    initially_complete = len(accepted_rows) == config.accepted_target_total
    if initially_complete:
        train_rows, validation_rows, reserve_rows = _materialize_train_validation_reserve_splits(
            config,
            accepted_rows,
            source_order=source_order,
        )
        accepted_by_dataset = Counter(str(row.get("dataset")) for row in accepted_rows)
        summary = {
            "status": "complete",
            "source": source_summary(examples),
            "processed_this_run": 0,
            "accepted_total": len(accepted_rows),
            "accepted_by_dataset": dict(sorted(accepted_by_dataset.items())),
            "accepted_behavior": _accepted_behavior_stats(accepted_rows),
            "train_trajectories": len(train_rows),
            "validation_trajectories": len(validation_rows),
            "reserve_trajectories": len(reserve_rows),
            "targets_met": True,
        }
        _write_json(config.output_dir / "summary.json", summary)
        return summary

    retriever = UnifiedE5FaissRetriever(
        corpus_path=config.corpus_path,
        offsets_path=config.corpus_offsets_path,
        index_path=config.index_path,
        manifest_path=config.index_manifest_path,
        model_path=config.retrieval_model_path,
        device=config.retrieval_device,
        max_length=config.retrieval_max_length,
        batch_size=config.retrieval_batch_size,
        batch_wait_ms=config.retrieval_batch_wait_ms,
        top_k=config.retrieval_top_k,
        mmap=config.faiss_mmap,
    )
    client = None if dry_run else DeepSeekClient(config)
    output_mode = "a" if config.resume else "w"
    accepted_path = config.output_dir / "accepted_sft.jsonl"
    counts: Counter[str] = Counter()
    dataset_counts: dict[str, Counter[str]] = {
        dataset: Counter() for dataset in sorted({item.dataset for item in examples})
    }
    processed_this_run = 0
    candidates = [
        item for item in examples
        if _should_process(item, statuses, accepted_qids, config)
    ]
    accepted_dataset_counts = Counter(str(row.get("dataset")) for row in accepted_rows)
    stage_counts: Counter[str] = Counter()
    stage_lock = threading.Lock()
    display_lock = threading.Lock()
    progress_holder: dict[str, Any] = {}
    last_progress_refresh = [0.0]

    def render_progress(*, force: bool = False) -> None:
        progress = progress_holder.get("progress")
        if progress is None:
            return
        now = time.monotonic()
        with display_lock:
            if not force and now - last_progress_refresh[0] < 1.0:
                return
            last_progress_refresh[0] = now
            with stage_lock:
                stages = dict(stage_counts)
            progress.set_postfix(
                accepted=f"{len(accepted_rows)}/{config.accepted_target_total}",
                failed=counts.get("failed", 0),
                filtered=counts.get("filtered", 0),
                stages=(
                    f"q{stages.get('query', 0)}/api{stages.get('api', 0)}/"
                    f"r{stages.get('retrieval', 0)}/e{stages.get('evidence', 0)}/"
                    f"a{stages.get('answer', 0)}"
                ),
                retries=f"q{stages.get('query_retry', 0)}/a{stages.get('answer_retry', 0)}",
                r_batch=f"{retriever.batch_stats()['average_batch_size']:.1f}",
            )
            progress.refresh()

    def record_stage(stage: str) -> None:
        with stage_lock:
            stage_counts[stage] += 1
        render_progress()

    generator = TrajectoryGenerator(
        config=config,
        prompts=prompts,
        client=client,
        retriever=retriever,
        dry_run=dry_run,
        progress_callback=record_stage,
    )
    next_candidate = 0
    with (
        ledger_path.open(output_mode, encoding="utf-8") as ledger,
        accepted_path.open(output_mode, encoding="utf-8") as accepted_handle,
        ThreadPoolExecutor(max_workers=config.sample_workers) as executor,
        tqdm(
            total=len(candidates),
            desc="SFT-v2 candidate scan",
            unit="sample",
            dynamic_ncols=True,
        ) as progress,
    ):
        progress_holder["progress"] = progress
        render_progress(force=True)
        while len(accepted_rows) < config.accepted_target_total and next_candidate < len(candidates):
            remaining = config.accepted_target_total - len(accepted_rows)
            batch = candidates[next_candidate : next_candidate + min(config.sample_workers, remaining)]
            next_candidate += len(batch)
            # Batch width never exceeds the remaining global quota, so an
            # all-success batch still cannot overshoot 2,200 accepted samples.
            future_to_example = {
                executor.submit(_safe_generate, generator, item): item
                for item in batch
            }
            for future in as_completed(future_to_example):
                example = future_to_example[future]
                result = future.result()
                processed_this_run += 1
                status = str(result.get("status"))
                counts[status] += 1
                dataset_counts[example.dataset][status] += 1
                if status == "accepted":
                    sample = result["sample"]
                    _append_jsonl(accepted_handle, sample)
                    accepted_rows.append(sample)
                    accepted_qids.add(example.qid)
                    accepted_dataset_counts[example.dataset] += 1
                ledger_row = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "qid": example.qid,
                    "dataset": example.dataset,
                    **{key: value for key, value in result.items() if key != "sample"},
                }
                _append_jsonl(ledger, ledger_row)
                progress.update(1)
                render_progress(force=True)

        progress_holder.clear()

    retriever.close()

    targets_met = len(accepted_rows) == config.accepted_target_total
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    reserve_rows: list[dict[str, Any]] = []
    if targets_met:
        train_rows, validation_rows, reserve_rows = _materialize_train_validation_reserve_splits(
            config,
            accepted_rows,
            source_order=source_order,
        )
    accepted_by_dataset = Counter(str(row.get("dataset")) for row in accepted_rows)
    summary = {
        "status": "complete" if targets_met else "incomplete",
        "source": source_summary(examples),
        "previously_recorded": len(statuses),
        "processed_this_run": processed_this_run,
        "counts": dict(sorted(counts.items())),
        "dataset_counts": {key: dict(sorted(value.items())) for key, value in sorted(dataset_counts.items())},
        "accepted_total": len(accepted_rows),
        "accepted_by_dataset": dict(sorted(accepted_by_dataset.items())),
        "accepted_behavior": _accepted_behavior_stats(accepted_rows),
        "accepted_target_total": config.accepted_target_total,
        "train_trajectories": len(train_rows),
        "validation_trajectories": len(validation_rows),
        "reserve_trajectories": len(reserve_rows),
        "outputs": {
            "accepted_pool": str(accepted_path),
            "train": str(config.output_dir / "train_sft.jsonl") if targets_met else None,
            "validation": str(config.output_dir / "validation_sft.jsonl") if targets_met else None,
            "reserve": str(config.output_dir / "reserve_sft.jsonl") if targets_met else None,
        },
        "ledger": str(ledger_path),
        "targets_met": targets_met,
        "retrieval_batching": retriever.batch_stats(),
        "stage_counts": dict(sorted(stage_counts.items())),
    }
    _write_json(config.output_dir / "summary.json", summary)
    return summary


def check_only(config: TeacherConfig, *, limit: int | None = None) -> dict[str, Any]:
    manifest = _validate_assets(config)
    prompts = load_prompt_contract(config.prompt_path)
    examples = _load_ordered_examples(config, limit=limit)
    return {
        "config_valid": True,
        "source": source_summary(examples),
        "prompt_contract_version": prompts.version,
        "prompt_contract_fingerprint": prompts.fingerprint,
        "accepted_target_total": config.accepted_target_total,
        "train_target_total": config.resolved_train_target_total,
        "validation_target_total": config.validation_target_total,
        "reserve_target_total": config.reserve_target_total,
        "acceptance_policy": "first_qualified_from_shuffled_unified_pool_without_dataset_quota",
        "source_order": (
            "deterministic_global_hash_shuffle"
            if config.shuffle_source_examples
            else "source_file_order"
        ),
        "retrieval_candidate_order": (
            "deterministic_per_query_hash_shuffle_then_reindex"
            if config.shuffle_retrieved_passages
            else "faiss_score_order"
        ),
        "retrieval_query_batch_size": config.retrieval_batch_size,
        "retrieval_batch_wait_ms": config.retrieval_batch_wait_ms,
        "sample_workers": config.sample_workers,
        "role_validation_retries": config.role_validation_retries,
        "shuffle_seed": config.seed,
        "index": {
            "ntotal": manifest.get("ntotal"),
            "dimension": manifest.get("dimension"),
            "metric": manifest.get("metric"),
            "pooling_method": manifest.get("pooling_method"),
            "max_length": manifest.get("max_length"),
        },
        "note": "check-only does not load the 64 GB FAISS index or build corpus offsets",
    }


def build_parser() -> argparse.ArgumentParser:
    default = Path(__file__).resolve().parent / "config" / "teacher_trajectory.yml"
    parser = argparse.ArgumentParser(description="Generate SFT-v2 three-agent trajectories with DeepSeek V4.1 Flash")
    parser.add_argument("--config", default=str(default))
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    config = load_config(args.config)
    result = check_only(config, limit=args.limit) if args.check_only else generate(config, limit=args.limit, dry_run=args.dry_run)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if args.dry_run or result.get("targets_met", True) else 2


if __name__ == "__main__":
    raise SystemExit(main())
