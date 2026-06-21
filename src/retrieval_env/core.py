from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any

from data_processing.io_utils import read_jsonl, write_json, write_jsonl


PROCESSED_DATASETS = ("2wiki", "hotpotqa", "musique")
RETRIEVAL_DEFAULT_SPLITS = ("train", "dev")
RETRIEVAL_DEFAULT_ROOT = "data/retrieval_env"


@dataclass(frozen=True)
class RetrievalResult:
    dataset: str
    query: str
    passages: list[str]
    scores: list[float]


def _dataset_dir(processed_root: str | Path, dataset: str) -> Path:
    return Path(processed_root) / dataset


def _retrieval_dataset_dir(retrieval_root: str | Path, dataset: str) -> Path:
    return Path(retrieval_root) / dataset


def _candidate_example_paths(dataset_dir: Path, dataset: str, split: str) -> list[Path]:
    candidates = [
        dataset_dir / f"examples.{split}.jsonl",
        dataset_dir / f"{dataset}_{split}.jsonl",
    ]
    # Backward-compatible fallback if someone renames split file as just `<split>.jsonl`.
    candidates.append(dataset_dir / f"{split}.jsonl")
    return candidates


def _resolve_example_path(dataset_dir: Path, dataset: str, split: str) -> Path:
    for path in _candidate_example_paths(dataset_dir, dataset, split):
        if path.exists():
            return path
    expected = [str(path) for path in _candidate_example_paths(dataset_dir, dataset, split)]
    raise FileNotFoundError(
        f"Could not find split file for {dataset}/{split}. Tried: {', '.join(expected)}"
    )


def _chunk_text_from_row(row: dict[str, Any]) -> str:
    chunk_text = str(row.get("text", "")).strip()
    if chunk_text:
        return chunk_text

    sentences = row.get("sentences", [])
    if isinstance(sentences, list):
        sent_texts = [
            str(item.get("text", "")).strip()
            for item in sentences
            if isinstance(item, dict) and item.get("text") is not None
        ]
        if sent_texts:
            return "\n".join(sent_texts)
    return ""


def build_linearrag_assets(
    *,
    processed_root: str | Path,
    retrieval_root: str | Path,
    datasets: list[str],
    splits: list[str],
) -> dict[str, dict[str, Any]]:
    """Build LinearRAG-style `questions.json` and `chunks.json` from processed data."""
    summary: dict[str, dict[str, Any]] = {}
    for dataset in datasets:
        source_dir = _dataset_dir(processed_root, dataset)
        target_dir = _retrieval_dataset_dir(retrieval_root, dataset)
        target_dir.mkdir(parents=True, exist_ok=True)

        corpus_records = list(read_jsonl(source_dir / "corpus.jsonl"))
        chunks: list[str] = []
        chunk_metadata: list[dict[str, Any]] = []
        for chunk_idx, row in enumerate(corpus_records):
            chunk_text = _chunk_text_from_row(row)
            title = str(row.get("title") or "").strip()
            chunk = f"{title}\n{chunk_text}" if title else chunk_text
            chunks.append(chunk)
            chunk_metadata.append(
                {
                    "chunk_id": chunk_idx,
                    "doc_id": row.get("doc_id"),
                    "dataset": dataset,
                    "title": row.get("title", ""),
                    "source": row.get("source", ""),
                }
            )

        questions: list[dict[str, Any]] = []
        for split in splits:
            example_path = _resolve_example_path(source_dir, dataset, split)
            for example in read_jsonl(example_path):
                questions.append(
                    {
                        "qid": example.get("qid"),
                        "question": example.get("question", ""),
                        "answer": example.get("answer"),
                        "split": split,
                        "dataset": dataset,
                        "question_type": example.get("question_type"),
                        "hop_count": example.get("hop_count"),
                    }
                )

        write_json(target_dir / "questions.json", questions)
        write_json(target_dir / "chunks.json", chunks)
        write_jsonl(target_dir / "chunk_metadata.jsonl", chunk_metadata)

        summary[dataset] = {
            "questions": len(questions),
            "chunks": len(chunks),
            "chunk_metadata": str(target_dir / "chunk_metadata.jsonl"),
            "example_files_used": [
                str(_resolve_example_path(source_dir, dataset, split)) for split in splits
            ],
        }

    return summary


@contextmanager
def _with_linearrag_pythonpath() -> Any:
    root = Path(__file__).resolve().parents[1] / "LinearRAG"
    if not root.exists():
        raise FileNotFoundError(
            f"LinearRAG directory not found: {root}. "
            "Expected local copy at /data/xudu/macorag/LinearRAG"
        )

    original = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = original


def _load_linearrag_modules() -> tuple[Any, Any]:
    with _with_linearrag_pythonpath():
        try:
            from src.config import LinearRAGConfig
            from src.LinearRAG import LinearRAG
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Cannot import LinearRAG modules. Please install LinearRAG dependencies "
                "in this environment, e.g.:\n"
                "  PIP_NO_BUILD_ISOLATION=1 python -m pip install -r LinearRAG/requirements.txt"
            ) from exc

        return LinearRAGConfig, LinearRAG


def build_linear_rag_index(
    *,
    retrieval_root: str | Path,
    dataset: str,
    chunks: list[str],
    embedding_model: str,
    spacy_model: str,
    max_workers: int = 16,
    batch_size: int = 128,
    retrieval_top_k: int = 5,
    use_vectorized_retrieval: bool = False,
) -> Path:
    """Build passage/entity/sentence embeddings and graph cache for one dataset."""
    LinearRAGConfig, LinearRAG = _load_linearrag_modules()

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install sentence-transformers to build LinearRAG index: "
            "pip install sentence-transformers"
        ) from exc

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        device = "cpu"

    working_dir = Path(retrieval_root) / dataset
    model = SentenceTransformer(embedding_model, device=device)
    config = LinearRAGConfig(
        dataset_name=dataset,
        embedding_model=model,
        llm_model=None,
        spacy_model=spacy_model,
        working_dir=str(working_dir),
        batch_size=batch_size,
        max_workers=max_workers,
        retrieval_top_k=retrieval_top_k,
        use_vectorized_retrieval=use_vectorized_retrieval,
    )

    engine = LinearRAG(config)
    engine.index(chunks)
    return working_dir


def query_linear_rag(
    *,
    retrieval_root: str | Path,
    dataset: str,
    query: str,
    embedding_model: str,
    spacy_model: str,
    top_k: int = 5,
    max_workers: int = 16,
    batch_size: int = 128,
    use_vectorized_retrieval: bool = False,
) -> RetrievalResult:
    """Query one processed LinearRAG index and return passages + scores."""
    LinearRAGConfig, LinearRAG = _load_linearrag_modules()

    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Install sentence-transformers to run retrieval: "
            "pip install sentence-transformers"
        ) from exc

    dataset_root = _retrieval_dataset_dir(retrieval_root, dataset)
    required = [
        dataset_root / "passage_embedding.parquet",
        dataset_root / "entity_embedding.parquet",
        dataset_root / "sentence_embedding.parquet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "LinearRAG index files missing: "
            + ", ".join(missing)
            + ". Run build_linear_rag_index() first."
        )

    model = SentenceTransformer(embedding_model)
    config = LinearRAGConfig(
        dataset_name=dataset,
        embedding_model=model,
        llm_model=None,
        spacy_model=spacy_model,
        working_dir=str(dataset_root),
        batch_size=batch_size,
        max_workers=max_workers,
        retrieval_top_k=top_k,
        use_vectorized_retrieval=use_vectorized_retrieval,
    )
    engine = LinearRAG(config)
    results = engine.retrieve([{"question": query}])[0]
    return RetrievalResult(
        dataset=dataset,
        query=query,
        passages=list(results["sorted_passage"]),
        scores=[float(score) for score in results["sorted_passage_scores"]],
    )
