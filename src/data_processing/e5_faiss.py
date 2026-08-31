from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from .io_utils import read_jsonl


DEFAULT_E5_MODEL = "intfloat/e5-base-v2"
DEFAULT_MAX_LENGTH = 512
INDEX_FILE = "e5_Flat.index"
CORPUS_FILE = "corpus.jsonl"
METADATA_FILE = "index_metadata.json"
SCHEMA_VERSION = 1
BUILD_STATE_SCHEMA_VERSION = 1
EMBEDDINGS_FILE = ".e5_embeddings.npy"
BUILD_STATE_FILE = ".e5_build_state.json"


def e5_index_fingerprint(metadata: dict[str, Any]) -> str:
    semantic = {
        key: value
        for key, value in metadata.items()
        if key not in {"index_contract_fingerprint", "build_status"}
    }
    canonical = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_faiss() -> Any:
    try:
        import faiss
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "FAISS is required for the e5_faiss retrieval backend. "
            "Install it with `python -m pip install faiss-cpu==1.9.0.post1`."
        ) from exc
    return faiss


def _passage_text(row: dict[str, Any]) -> str:
    title = str(row.get("title") or "").strip()
    text = str(row.get("text") or "").strip()
    return f"{json.dumps(title, ensure_ascii=False)}\n{text}" if title else text


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_corpus(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for passage_id, row in enumerate(rows):
            payload = dict(row)
            payload["passage_id"] = passage_id
            handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _resume_contract_error(detail: str) -> RuntimeError:
    return RuntimeError(
        "E5-FAISS resume contract mismatch: "
        f"{detail}. Remove .e5_embeddings.npy and .e5_build_state.json "
        "from this dataset output directory to restart it."
    )


def _passage_batches(
    encoder: Any,
    texts: list[str],
    *,
    start: int,
    batch_size: int,
    progress_desc: str,
) -> Iterator[tuple[int, np.ndarray]]:
    batch_method = getattr(encoder, "encode_passage_batches", None)
    if callable(batch_method):
        yield from batch_method(
            texts,
            start=start,
            show_progress=True,
            progress_desc=progress_desc,
        )
        return
    for batch_start in range(start, len(texts), batch_size):
        batch_end = min(batch_start + batch_size, len(texts))
        yield batch_end, encoder.encode_passages(texts[batch_start:batch_end])


@dataclass(frozen=True)
class E5FaissResult:
    dataset: str
    query: str
    passages: list[dict[str, Any]]
    scores: list[float]


class E5Encoder:
    def __init__(
        self,
        *,
        model_name: str,
        device: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = 32,
        tokenizer: Any = None,
        model: Any = None,
    ) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if tokenizer is None or model is None:
            try:
                from transformers import AutoModel, AutoTokenizer
            except ModuleNotFoundError as exc:
                raise RuntimeError("transformers is required for E5 retrieval") from exc
            local_model_path = Path(model_name)
            if not local_model_path.exists():
                try:
                    from huggingface_hub import snapshot_download

                    local_model_path = Path(
                        snapshot_download(repo_id=model_name, local_files_only=True)
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"E5 model is not available locally: {model_name}. "
                        "Download it before building or querying the index."
                    ) from exc
            tokenizer = AutoTokenizer.from_pretrained(
                local_model_path,
                local_files_only=True,
            )
            model = AutoModel.from_pretrained(local_model_path, local_files_only=True)
        self.model_name = str(model_name)
        self.device = str(device)
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.tokenizer = tokenizer
        self.model = model.eval().to(self.device)
        config = getattr(self.model, "config", None)
        self.dimension = int(getattr(config, "hidden_size", 0) or 0)

    @staticmethod
    def masked_mean_pool(last_hidden_state: Any, attention_mask: Any) -> Any:
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def _encode(self, texts: list[str], *, prefix: str) -> np.ndarray:
        import torch
        import torch.nn.functional as functional

        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        batches: list[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [f"{prefix}{text}" for text in texts[start : start + self.batch_size]]
            tokenized = self.tokenizer(
                batch,
                max_length=self.max_length,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in tokenized.items()}
            with torch.inference_mode():
                output = self.model(**inputs)
                pooled = self.masked_mean_pool(output.last_hidden_state, inputs["attention_mask"])
                normalized = functional.normalize(pooled, p=2, dim=-1)
            array = normalized.detach().cpu().to(torch.float32).numpy()
            batches.append(np.ascontiguousarray(array, dtype=np.float32))
        result = np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)
        self.dimension = int(result.shape[1])
        return result

    def encode_passage_batches(
        self,
        texts: list[str],
        *,
        start: int = 0,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ) -> Iterator[tuple[int, np.ndarray]]:
        if start < 0 or start > len(texts):
            raise ValueError(f"start must be between 0 and {len(texts)}; got {start}")
        progress: Any = None
        if show_progress:
            from tqdm.auto import tqdm

            progress = tqdm(
                total=len(texts),
                initial=start,
                desc=progress_desc or "E5 passages",
                unit="passage",
                dynamic_ncols=True,
            )
        try:
            for batch_start in range(start, len(texts), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(texts))
                vectors = self._encode(texts[batch_start:batch_end], prefix="passage: ")
                yield batch_end, vectors
                if progress is not None:
                    progress.update(batch_end - batch_start)
        finally:
            if progress is not None:
                progress.close()

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prefix="query: ")

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prefix="passage: ")


def build_e5_faiss_index(
    *,
    data_root: str | Path,
    output_root: str | Path,
    dataset: str,
    model_name: str = DEFAULT_E5_MODEL,
    device: str = "cpu",
    max_length: int = DEFAULT_MAX_LENGTH,
    batch_size: int = 64,
    encoder: E5Encoder | None = None,
    faiss_module: Any = None,
) -> dict[str, Any]:
    source_path = Path(data_root) / dataset / CORPUS_FILE
    if not source_path.is_file():
        raise FileNotFoundError(f"E5-FAISS source corpus not found: {source_path}")
    rows = list(read_jsonl(source_path))
    if not rows:
        raise ValueError(f"E5-FAISS source corpus is empty: {source_path}")

    source_corpus_sha256 = _sha256(source_path)
    target_dir = Path(output_root) / dataset
    target_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = target_dir / CORPUS_FILE
    index_path = target_dir / INDEX_FILE
    metadata_path = target_dir / METADATA_FILE
    final_paths = (corpus_path, index_path, metadata_path)
    if all(path.is_file() for path in final_paths):
        metadata = validate_e5_index_contract(
            retrieval_root=output_root,
            dataset=dataset,
            expected_model=model_name,
            expected_max_length=max_length,
            expected_corpus_count=len(rows),
            faiss_module=faiss_module,
        )
        if str(metadata.get("source_corpus_sha256", "")) == source_corpus_sha256:
            return {**metadata, "build_status": "skipped"}

    active_encoder = encoder or E5Encoder(
        model_name=model_name,
        device=device,
        max_length=max_length,
        batch_size=batch_size,
    )
    dimension = int(getattr(active_encoder, "dimension", 0) or 0)
    if dimension <= 0:
        raise RuntimeError("E5 passage encoder must expose a positive dimension")

    state_path = target_dir / BUILD_STATE_FILE
    embeddings_path = target_dir / EMBEDDINGS_FILE
    expected_state = {
        "schema_version": BUILD_STATE_SCHEMA_VERSION,
        "dataset": dataset,
        "source_corpus_sha256": source_corpus_sha256,
        "retriever_model": str(model_name),
        "max_length": int(max_length),
        "corpus_count": len(rows),
        "dimension": dimension,
    }
    if state_path.exists() != embeddings_path.exists():
        raise _resume_contract_error("only one temporary resume artifact exists")

    encoded_count = 0
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise _resume_contract_error(f"invalid state file: {exc}") from exc
        mismatches = [
            key
            for key, expected in expected_state.items()
            if state.get(key) != expected
        ]
        if mismatches:
            raise _resume_contract_error(
                "changed fields: " + ", ".join(sorted(mismatches))
            )
        encoded_count = int(state.get("encoded_count", -1))
        if encoded_count < 0 or encoded_count > len(rows):
            raise _resume_contract_error(f"invalid encoded_count={encoded_count}")
        try:
            vectors = np.load(embeddings_path, mmap_mode="r+")
        except Exception as exc:
            raise _resume_contract_error(f"invalid embedding file: {exc}") from exc
        if vectors.shape != (len(rows), dimension) or vectors.dtype != np.float32:
            raise _resume_contract_error(
                f"embedding shape/dtype is {vectors.shape}/{vectors.dtype}"
            )
    else:
        vectors = np.lib.format.open_memmap(
            embeddings_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(rows), dimension),
        )
        _write_json_atomic(state_path, {**expected_state, "encoded_count": 0})

    texts = [_passage_text(row) for row in rows]
    batch_iterator = _passage_batches(
        active_encoder,
        texts,
        start=encoded_count,
        batch_size=batch_size,
        progress_desc=f"E5 {dataset}",
    )
    previous_end = encoded_count
    for batch_end, batch_vectors in batch_iterator:
        batch_vectors = np.ascontiguousarray(batch_vectors, dtype=np.float32)
        expected_shape = (int(batch_end) - previous_end, dimension)
        if batch_end <= previous_end or batch_end > len(rows) or batch_vectors.shape != expected_shape:
            raise RuntimeError(
                "E5 passage encoder returned an invalid batch: "
                f"end={batch_end}, shape={tuple(batch_vectors.shape)}, expected={expected_shape}"
            )
        vectors[previous_end:batch_end] = batch_vectors
        vectors.flush()
        previous_end = int(batch_end)
        _write_json_atomic(
            state_path,
            {**expected_state, "encoded_count": previous_end},
        )
    if previous_end != len(rows):
        raise RuntimeError(
            f"E5 passage encoder stopped early: encoded {previous_end}/{len(rows)} passages"
        )

    faiss_api = faiss_module or _load_faiss()
    index = faiss_api.IndexFlatIP(dimension)
    index.add(np.asarray(vectors, dtype=np.float32))
    index_tmp_path = target_dir / f".{INDEX_FILE}.tmp"
    faiss_api.write_index(index, str(index_tmp_path))
    index_tmp_path.replace(index_path)

    corpus_tmp_path = target_dir / f".{CORPUS_FILE}.tmp"
    _write_corpus(corpus_tmp_path, rows)
    corpus_tmp_path.replace(corpus_path)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "retriever_model": str(model_name),
        "pooling_method": "mean",
        "l2_normalized": True,
        "similarity_metric": "inner_product",
        "passage_format": "json_title_newline_text",
        "max_length": int(max_length),
        "faiss_type": "IndexFlatIP",
        "dimension": dimension,
        "corpus_count": len(rows),
        "corpus_sha256": _sha256(corpus_path),
        "source_corpus_sha256": source_corpus_sha256,
    }
    metadata["index_contract_fingerprint"] = e5_index_fingerprint(metadata)
    _write_json_atomic(metadata_path, metadata)
    del vectors
    embeddings_path.unlink()
    state_path.unlink()
    return {**metadata, "build_status": "built"}


def validate_e5_index_contract(
    *,
    retrieval_root: str | Path,
    dataset: str,
    expected_model: str,
    expected_max_length: int = DEFAULT_MAX_LENGTH,
    expected_corpus_count: int | None = None,
    faiss_module: Any = None,
) -> dict[str, Any]:
    metadata = validate_e5_faiss_assets(
        retrieval_root=retrieval_root,
        dataset=dataset,
        expected_model=expected_model,
        faiss_module=faiss_module,
    )
    if int(metadata.get("max_length", -1)) != int(expected_max_length):
        raise RuntimeError(
            f"E5-FAISS max_length mismatch: expected {expected_max_length}, got {metadata.get('max_length')}"
        )
    if str(metadata.get("faiss_type")) != "IndexFlatIP":
        raise RuntimeError(f"E5-FAISS index type mismatch: expected IndexFlatIP, got {metadata.get('faiss_type')}")
    if expected_corpus_count is not None and int(metadata.get("corpus_count", -1)) != int(expected_corpus_count):
        raise RuntimeError(
            "E5-FAISS corpus contract mismatch: "
            f"expected {expected_corpus_count}, got {metadata.get('corpus_count')}"
        )
    fingerprint = e5_index_fingerprint(metadata)
    stored = metadata.get("index_contract_fingerprint")
    if stored is not None and str(stored) != fingerprint:
        raise RuntimeError("E5-FAISS index contract fingerprint mismatch")
    metadata["index_contract_fingerprint"] = fingerprint
    return metadata


def validate_e5_faiss_assets(
    *,
    retrieval_root: str | Path,
    dataset: str,
    expected_model: str | None = None,
    faiss_module: Any = None,
) -> dict[str, Any]:
    dataset_root = Path(retrieval_root) / dataset
    corpus_path = dataset_root / CORPUS_FILE
    index_path = dataset_root / INDEX_FILE
    metadata_path = dataset_root / METADATA_FILE
    missing = [path for path in (corpus_path, index_path, metadata_path) if not path.is_file()]
    if missing:
        raise RuntimeError(
            "E5-FAISS assets are incomplete for "
            f"{dataset}: {', '.join(str(path) for path in missing)}"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"E5-FAISS metadata schema mismatch for {dataset}")
    if str(metadata.get("dataset")) != dataset:
        raise RuntimeError(f"E5-FAISS metadata dataset mismatch for {dataset}")
    if expected_model and str(metadata.get("retriever_model")) != str(expected_model):
        raise RuntimeError(
            "E5-FAISS retriever model mismatch: "
            f"expected {expected_model}, got {metadata.get('retriever_model')}"
        )
    actual_hash = _sha256(corpus_path)
    if actual_hash != str(metadata.get("corpus_sha256")):
        raise RuntimeError(
            "E5-FAISS corpus SHA256 mismatch: "
            f"expected {metadata.get('corpus_sha256')}, got {actual_hash}"
        )
    corpus_count = sum(1 for _ in read_jsonl(corpus_path))
    expected_count = int(metadata.get("corpus_count", -1))
    if corpus_count != expected_count:
        raise RuntimeError(
            f"E5-FAISS corpus count mismatch: expected {expected_count}, got {corpus_count}"
        )
    faiss_api = faiss_module or _load_faiss()
    index = faiss_api.read_index(str(index_path))
    if int(index.ntotal) != expected_count:
        raise RuntimeError(
            f"E5-FAISS index count mismatch: expected {expected_count}, got {index.ntotal}"
        )
    expected_dimension = int(metadata.get("dimension", -1))
    if int(index.d) != expected_dimension:
        raise RuntimeError(
            f"E5-FAISS index dimension mismatch: expected {expected_dimension}, got {index.d}"
        )
    return dict(metadata)


class E5FaissQueryEngine:
    def __init__(
        self,
        *,
        retrieval_root: str | Path,
        dataset: str,
        model_name: str = DEFAULT_E5_MODEL,
        device: str = "cpu",
        top_k: int = 5,
        max_length: int = DEFAULT_MAX_LENGTH,
        batch_size: int = 32,
        encoder: E5Encoder | None = None,
        faiss_module: Any = None,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.retrieval_root = Path(retrieval_root)
        self.dataset = dataset
        self.top_k = int(top_k)
        self._faiss = faiss_module or _load_faiss()
        self.metadata = validate_e5_faiss_assets(
            retrieval_root=self.retrieval_root,
            dataset=dataset,
            expected_model=model_name,
            faiss_module=self._faiss,
        )
        dataset_root = self.retrieval_root / dataset
        self.corpus = list(read_jsonl(dataset_root / CORPUS_FILE))
        self.index = self._faiss.read_index(str(dataset_root / INDEX_FILE))
        self.encoder = encoder or E5Encoder(
            model_name=model_name,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
        )

    def query(self, query: str) -> E5FaissResult:
        return self.query_batch([query])[0]

    def query_batch(self, queries: list[str]) -> list[E5FaissResult]:
        if not queries:
            return []
        vectors = np.ascontiguousarray(self.encoder.encode_queries(queries), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != int(self.index.d):
            raise RuntimeError(
                "E5 query encoder dimension mismatch: "
                f"expected {self.index.d}, got {tuple(vectors.shape)}"
            )
        limit = min(self.top_k, int(self.index.ntotal))
        scores, indices = self.index.search(vectors, limit)
        results: list[E5FaissResult] = []
        for query, row_scores, row_indices in zip(queries, scores, indices):
            passages: list[dict[str, Any]] = []
            kept_scores: list[float] = []
            for score, passage_id in zip(row_scores.tolist(), row_indices.tolist()):
                if int(passage_id) < 0:
                    continue
                passage = dict(self.corpus[int(passage_id)])
                passage["passage_id"] = int(passage_id)
                passages.append(passage)
                kept_scores.append(float(score))
            results.append(
                E5FaissResult(
                    dataset=self.dataset,
                    query=query,
                    passages=passages,
                    scores=kept_scores,
                )
            )
        return results
