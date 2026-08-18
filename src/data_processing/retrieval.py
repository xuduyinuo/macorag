from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import sys
from pathlib import Path
from typing import Any, Optional, Union

from data_processing.io_utils import read_jsonl, write_json, write_jsonl

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable


PROCESSED_DATASETS = ("2wiki", "hotpotqa", "musique")
RETRIEVAL_DEFAULT_SPLITS = ("train", "dev")
RETRIEVAL_DEFAULT_ROOT = "data/retrieval_env"
SPACY_FALLBACK_MODELS = ("en_core_web_sm", "en_core_web_md")
REQUIRED_INDEX_FILES = (
    "passage_embedding.parquet",
    "entity_embedding.parquet",
    "sentence_embedding.parquet",
)


def _resolve_spacy_model(preferred_model: Optional[str]) -> str:
    """解析可用的 spaCy 模型；优先使用配置值，缺失时尝试轻量模型。"""
    try:
        import spacy
    except Exception as exc:
        raise RuntimeError("spaCy is required for retrieval building/querying. Install it with `python -m pip install spacy`.") from exc

    candidates: list[str] = []
    if preferred_model:
        candidates.append(preferred_model)
    candidates.extend(SPACY_FALLBACK_MODELS)

    for model_name in candidates:
        if not model_name:
            continue
        try:
            nlp = spacy.load(model_name)
            del nlp
            return model_name
        except OSError:
            continue

    installed = ", ".join(spacy.util.get_installed_models())
    fallback_hint = ", ".join(SPACY_FALLBACK_MODELS)
    raise RuntimeError(
        "No usable spaCy NER model found. "
        f"Preferred='{preferred_model}'. Installed models: [{installed if installed else 'none'}]. "
        f"Install one with: python -m spacy download {fallback_hint}"
    )


@dataclass(frozen=True)
class RetrievalResult:
    dataset: str
    query: str
    passages: list[str]
    scores: list[float]


class LinearRAGQueryEngine:
    def __init__(
        self,
        *,
        retrieval_root: Union[str, Path],
        dataset: str,
        embedding_model: str,
        spacy_model: str,
        top_k: int = 5,
        max_workers: int = 16,
        batch_size: int = 128,
        use_vectorized_retrieval: bool = False,
    ) -> None:
        LinearRAGConfig, LinearRAG = _load_linearrag_modules()

        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "Install sentence-transformers to run retrieval: "
                "pip install sentence-transformers"
            ) from exc

        working_dir, index_root = _resolve_retrieval_dataset_working_dir(retrieval_root, dataset)
        candidate_index_dirs = [
            _retrieval_dataset_dir(retrieval_root, dataset),
            _retrieval_dataset_dir(retrieval_root, dataset) / dataset,
        ]
        missing = [str(path) for path in _missing_index_files(index_root)]
        if missing:
            raise RuntimeError(
                "LinearRAG index files missing: "
                + ", ".join(missing)
                + ". Run build_linear_rag_index() first."
                + " Checked directories: "
                + ", ".join(str(path) for path in candidate_index_dirs)
            )

        try:
            model = SentenceTransformer(embedding_model, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load retrieval embedding model from local cache: "
                f"{embedding_model!r}. The query path runs offline to avoid Hugging Face/proxy failures during "
                "long RL jobs. Pre-download the model into the Hugging Face cache or set retrieval_embedding_model "
                "to a local model directory."
            ) from exc
        resolved_spacy_model = _resolve_spacy_model(spacy_model)
        config = LinearRAGConfig(
            dataset_name=dataset,
            embedding_model=model,
            llm_model=None,
            spacy_model=resolved_spacy_model,
            working_dir=str(working_dir),
            batch_size=batch_size,
            max_workers=max_workers,
            retrieval_top_k=top_k,
            use_vectorized_retrieval=use_vectorized_retrieval,
        )
        self.dataset = dataset
        self.engine = LinearRAG(config)

    def prepare(self) -> None:
        prepare = getattr(self.engine, "_prepare_retrieval_state", None)
        if callable(prepare):
            prepare()

    def query(self, query: str) -> RetrievalResult:
        return self.query_batch([query])[0]

    def query_batch(self, queries: list[str]) -> list[RetrievalResult]:
        if not queries:
            return []
        rows = self.engine.retrieve([{"question": query} for query in queries])
        if len(rows) != len(queries):
            raise RuntimeError(
                "LinearRAG returned a mismatched batch size: "
                f"expected {len(queries)}, got {len(rows)}."
            )
        return [
            RetrievalResult(
                dataset=self.dataset,
                query=query,
                passages=list(row["sorted_passage"]),
                scores=[float(score) for score in row["sorted_passage_scores"]],
            )
            for query, row in zip(queries, rows)
        ]


def _dataset_dir(processed_root: Union[str, Path], dataset: str) -> Path:
    return Path(processed_root) / dataset


def _retrieval_dataset_dir(retrieval_root: Union[str, Path], dataset: str) -> Path:
    return Path(retrieval_root) / dataset


def _resolve_retrieval_dataset_working_dir(retrieval_root: Union[str, Path], dataset: str) -> tuple[Path, Path]:
    """Return:
    - working_dir: directory passed to LinearRAGConfig. LinearRAG will append dataset name itself.
    - index_dir: directory that must contain `<split>_embedding.parquet`.
    """
    dataset_root = _retrieval_dataset_dir(retrieval_root, dataset)
    candidate_dirs = [
        dataset_root,
        dataset_root / dataset,
    ]

    for index_dir in candidate_dirs:
        if all((index_dir / filename).exists() for filename in REQUIRED_INDEX_FILES):
            working_dir = index_dir.parent
            return working_dir, index_dir

    return dataset_root.parent, dataset_root


def _missing_index_files(index_dir: Path) -> list[Path]:
    return [index_dir / filename for filename in REQUIRED_INDEX_FILES if not (index_dir / filename).exists()]


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
    processed_root: Union[str, Path],
    retrieval_root: Union[str, Path],
    datasets: list[str],
    splits: list[str],
) -> dict[str, dict[str, Any]]:
    """从标准处理数据构建 LinearRAG 需要的 questions/chunks 文件。"""
    summary: dict[str, dict[str, Any]] = {}
    for dataset in tqdm(datasets, desc="Building retrieval corpora", unit="dataset"):
        source_dir = _dataset_dir(processed_root, dataset)
        target_dir = _retrieval_dataset_dir(retrieval_root, dataset)
        target_dir.mkdir(parents=True, exist_ok=True)

        corpus_records = list(read_jsonl(source_dir / "corpus.jsonl"))
        chunks: list[str] = []
        chunk_metadata: list[dict[str, Any]] = []
        for chunk_idx, row in tqdm(
            enumerate(corpus_records),
            total=len(corpus_records),
            desc=f"{dataset} chunks",
            unit="chunk",
        ):
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
            for example in tqdm(
                read_jsonl(example_path),
                desc=f"{dataset} {split} questions",
                unit="question",
            ):
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
    # LinearRAG is vendored at repository root, alongside `src`.
    root = Path(__file__).resolve().parents[2] / "LinearRAG"
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
    retrieval_root: Union[str, Path],
    dataset: str,
    chunks: list[str],
    embedding_model: str,
    spacy_model: str,
    max_workers: int = 16,
    batch_size: int = 128,
    retrieval_top_k: int = 5,
    use_vectorized_retrieval: bool = False,
    co_locate_index: bool = False,
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

    resolved_spacy_model = _resolve_spacy_model(spacy_model)
    retrieval_root = Path(retrieval_root)
    if co_locate_index:
        working_dir = str(retrieval_root)
        index_dir = retrieval_root / dataset
    else:
        working_dir = str(retrieval_root / dataset)
        index_dir = Path(working_dir) / dataset
    model = SentenceTransformer(embedding_model, device=device)
    config = LinearRAGConfig(
        dataset_name=dataset,
        embedding_model=model,
        llm_model=None,
        spacy_model=resolved_spacy_model,
        working_dir=working_dir,
        batch_size=batch_size,
        max_workers=max_workers,
        retrieval_top_k=retrieval_top_k,
        use_vectorized_retrieval=use_vectorized_retrieval,
    )

    engine = LinearRAG(config)
    with tqdm(total=1, desc=f"Indexing {dataset}", unit="stage") as pbar:
        engine.index(chunks)
        pbar.update(1)
    return index_dir


def query_linear_rag(
    *,
    retrieval_root: Union[str, Path],
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
    engine = create_linear_rag_query_engine(
        retrieval_root=retrieval_root,
        dataset=dataset,
        embedding_model=embedding_model,
        spacy_model=spacy_model,
        top_k=top_k,
        max_workers=max_workers,
        batch_size=batch_size,
        use_vectorized_retrieval=use_vectorized_retrieval,
    )
    return engine.query(query)


def create_linear_rag_query_engine(
    *,
    retrieval_root: Union[str, Path],
    dataset: str,
    embedding_model: str,
    spacy_model: str,
    top_k: int = 5,
    max_workers: int = 16,
    batch_size: int = 128,
    use_vectorized_retrieval: bool = False,
) -> LinearRAGQueryEngine:
    """Create a reusable LinearRAG query engine for repeated queries to one dataset."""
    return LinearRAGQueryEngine(
        retrieval_root=retrieval_root,
        dataset=dataset,
        embedding_model=embedding_model,
        spacy_model=spacy_model,
        top_k=top_k,
        max_workers=max_workers,
        batch_size=batch_size,
        use_vectorized_retrieval=use_vectorized_retrieval,
    )
