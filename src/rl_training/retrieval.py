from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from data_processing.retrieval import create_linear_rag_query_engine


class CachedLinearRAGRetrievalEnv:
    def __init__(
        self,
        *,
        retrieval_root: str | Path,
        embedding_model: str,
        spacy_model: str | None,
        top_k: int,
        max_workers: int,
        batch_size: int,
        use_vectorized_retrieval: bool,
    ) -> None:
        self.retrieval_root = Path(retrieval_root)
        self.embedding_model = embedding_model
        self.spacy_model = spacy_model
        self.top_k = top_k
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_vectorized_retrieval = use_vectorized_retrieval
        self._engines_by_dataset: dict[str, Any] = {}
        self._engine_locks_by_dataset: dict[str, threading.RLock] = {}
        self._registry_lock = threading.Lock()

    def _dataset_lock(self, dataset: str) -> threading.RLock:
        with self._registry_lock:
            lock = self._engine_locks_by_dataset.get(dataset)
            if lock is None:
                lock = threading.RLock()
                self._engine_locks_by_dataset[dataset] = lock
            return lock

    def _engine(self, dataset: str) -> Any:
        lock = self._dataset_lock(dataset)
        with lock:
            engine = self._engines_by_dataset.get(dataset)
            if engine is None:
                engine = create_linear_rag_query_engine(
                    retrieval_root=self.retrieval_root,
                    dataset=dataset,
                    embedding_model=self.embedding_model,
                    spacy_model=self.spacy_model,
                    top_k=self.top_k,
                    max_workers=self.max_workers,
                    batch_size=self.batch_size,
                    use_vectorized_retrieval=self.use_vectorized_retrieval,
                )
                self._engines_by_dataset[dataset] = engine
            return engine

    def prewarm(self, datasets: list[str] | tuple[str, ...] | set[str]) -> None:
        seen: set[str] = set()
        for dataset in datasets:
            if dataset in seen:
                continue
            engine = self._engine(dataset)
            prepare = getattr(engine, "prepare", None)
            if callable(prepare):
                prepare()
            seen.add(dataset)

    def query(self, dataset: str, query: str) -> dict[str, Any]:
        lock = self._dataset_lock(dataset)
        with lock:
            result = self._engine(dataset).query(query)
        return self._observation(result, query=query)

    @staticmethod
    def _observation(result: Any, *, query: str | None = None) -> dict[str, Any]:
        passages = []
        for passage_id, text in enumerate(result.passages):
            passages.append(
                {
                    "passage_id": passage_id,
                    "title": "",
                    "text": text,
                    "score": result.scores[passage_id] if passage_id < len(result.scores) else None,
                }
            )
        return {"query": query if query is not None else result.query, "passages": passages}

    def query_batch(self, dataset: str, queries: list[str]) -> list[dict[str, Any]]:
        if not queries:
            return []
        lock = self._dataset_lock(dataset)
        with lock:
            results = self._engine(dataset).query_batch(queries)
        if len(results) != len(queries):
            raise RuntimeError(
                "LinearRAG query environment returned a mismatched batch size: "
                f"expected {len(queries)}, got {len(results)}."
            )
        return [self._observation(result) for result in results]
