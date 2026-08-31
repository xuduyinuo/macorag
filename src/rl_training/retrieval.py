from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

from data_processing.e5_faiss import E5Encoder, E5FaissQueryEngine, validate_e5_faiss_assets
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
        query_cache_size: int = 0,
    ) -> None:
        self.retrieval_root = Path(retrieval_root)
        self.embedding_model = embedding_model
        self.spacy_model = spacy_model
        self.top_k = top_k
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.use_vectorized_retrieval = use_vectorized_retrieval
        self.query_cache_size = max(0, int(query_cache_size))
        self._engines_by_dataset: dict[str, Any] = {}
        self._engine_locks_by_dataset: dict[str, threading.RLock] = {}
        self._registry_lock = threading.Lock()
        self._cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._stats_lock = threading.Lock()
        self._cache_hits = 0
        self._cache_misses = 0
        self._time_retrieval_seconds = 0.0

    @staticmethod
    def _cache_key(dataset: str, query: str) -> tuple[str, str]:
        return dataset, " ".join(str(query).casefold().split())

    def _record_stats(self, *, hits: int = 0, misses: int = 0, seconds: float = 0.0) -> None:
        with self._stats_lock:
            self._cache_hits += hits
            self._cache_misses += misses
            self._time_retrieval_seconds += seconds

    def stats(self) -> dict[str, float | int]:
        with self._stats_lock:
            return {
                "cache_hits": self._cache_hits,
                "cache_misses": self._cache_misses,
                "time_retrieval_seconds": self._time_retrieval_seconds,
            }

    def _cached(self, key: tuple[str, str], *, query: str) -> dict[str, Any] | None:
        if self.query_cache_size <= 0 or key not in self._cache:
            return None
        observation = self._cache.pop(key)
        self._cache[key] = observation
        result = copy.deepcopy(observation)
        result["query"] = query
        return result

    def _store(self, key: tuple[str, str], observation: dict[str, Any]) -> None:
        if self.query_cache_size <= 0:
            return
        self._cache.pop(key, None)
        self._cache[key] = copy.deepcopy(observation)
        while len(self._cache) > self.query_cache_size:
            self._cache.popitem(last=False)

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
            key = self._cache_key(dataset, query)
            cached = self._cached(key, query=query)
            if cached is not None:
                self._record_stats(hits=1)
                return cached
            start = time.perf_counter()
            result = self._engine(dataset).query(query)
            elapsed = time.perf_counter() - start
            observation = self._observation(result, query=query)
            self._store(key, observation)
            self._record_stats(misses=1, seconds=elapsed)
            return copy.deepcopy(observation)

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
            observations: list[dict[str, Any] | None] = [None] * len(queries)
            miss_positions: dict[tuple[str, str], list[int]] = {}
            miss_queries: dict[tuple[str, str], str] = {}
            hits = 0
            for index, query in enumerate(queries):
                key = self._cache_key(dataset, query)
                cached = self._cached(key, query=query)
                if cached is not None:
                    observations[index] = cached
                    hits += 1
                elif key in miss_positions and self.query_cache_size > 0:
                    miss_positions[key].append(index)
                    hits += 1
                else:
                    miss_positions[key] = [index]
                    miss_queries[key] = query

            unique_queries = list(miss_queries.values())
            elapsed = 0.0
            if unique_queries:
                engine = self._engine(dataset)
                start = time.perf_counter()
                batch_query = getattr(engine, "query_batch", None)
                if callable(batch_query):
                    results = batch_query(unique_queries)
                else:
                    results = [engine.query(query) for query in unique_queries]
                elapsed = time.perf_counter() - start
                if len(results) != len(unique_queries):
                    raise RuntimeError(
                        "LinearRAG query environment returned a mismatched batch size: "
                        f"expected {len(unique_queries)}, got {len(results)}."
                    )
                for key, result in zip(miss_queries, results):
                    base_observation = self._observation(result, query=miss_queries[key])
                    self._store(key, base_observation)
                    for index in miss_positions[key]:
                        observation = copy.deepcopy(base_observation)
                        observation["query"] = queries[index]
                        observations[index] = observation
            self._record_stats(hits=hits, misses=len(unique_queries), seconds=elapsed)
            return [copy.deepcopy(item) for item in observations if item is not None]


class CachedE5FaissRetrievalEnv(CachedLinearRAGRetrievalEnv):
    def __init__(
        self,
        *,
        retrieval_root: str | Path,
        embedding_model: str,
        device: str,
        top_k: int,
        max_length: int,
        batch_size: int,
        query_cache_size: int = 0,
        engine_factory: Any = E5FaissQueryEngine,
        encoder_factory: Any = E5Encoder,
    ) -> None:
        super().__init__(
            retrieval_root=retrieval_root,
            embedding_model=embedding_model,
            spacy_model=None,
            top_k=top_k,
            max_workers=1,
            batch_size=batch_size,
            use_vectorized_retrieval=True,
            query_cache_size=query_cache_size,
        )
        self.device = str(device)
        self.max_length = int(max_length)
        self._e5_engine_factory = engine_factory
        self._e5_encoder_factory = encoder_factory
        self._shared_encoder: Any | None = None
        self._e5_query_lock = threading.RLock()

    def _encoder(self) -> Any:
        with self._registry_lock:
            if self._shared_encoder is None:
                self._shared_encoder = self._e5_encoder_factory(
                    model_name=self.embedding_model,
                    device=self.device,
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                )
            return self._shared_encoder

    def _engine(self, dataset: str) -> Any:
        lock = self._dataset_lock(dataset)
        with lock:
            engine = self._engines_by_dataset.get(dataset)
            if engine is None:
                engine = self._e5_engine_factory(
                    retrieval_root=self.retrieval_root,
                    dataset=dataset,
                    model_name=self.embedding_model,
                    device=self.device,
                    top_k=self.top_k,
                    max_length=self.max_length,
                    batch_size=self.batch_size,
                    encoder=self._encoder(),
                )
                self._engines_by_dataset[dataset] = engine
            return engine

    def query(self, dataset: str, query: str) -> dict[str, Any]:
        with self._e5_query_lock:
            return super().query(dataset, query)

    def query_batch(self, dataset: str, queries: list[str]) -> list[dict[str, Any]]:
        with self._e5_query_lock:
            return super().query_batch(dataset, queries)

    @staticmethod
    def _observation(result: Any, *, query: str | None = None) -> dict[str, Any]:
        passages: list[dict[str, Any]] = []
        for passage_id, value in enumerate(result.passages):
            passage = value if isinstance(value, dict) else {"text": str(value)}
            passages.append(
                {
                    "passage_id": passage_id,
                    "title": str(passage.get("title") or ""),
                    "text": str(passage.get("text") or ""),
                    "score": result.scores[passage_id]
                    if passage_id < len(result.scores)
                    else None,
                }
            )
        return {"query": query if query is not None else result.query, "passages": passages}


def create_retrieval_env(
    *,
    backend: str,
    retrieval_root: str | Path,
    embedding_model: str,
    device: str,
    top_k: int,
    max_length: int,
    batch_size: int,
    query_cache_size: int = 0,
    spacy_model: str | None = None,
    max_workers: int = 4,
    use_vectorized_retrieval: bool = True,
    e5_engine_factory: Any = E5FaissQueryEngine,
    e5_encoder_factory: Any = E5Encoder,
) -> Any:
    normalized = str(backend).strip().casefold()
    if normalized == "e5_faiss":
        return CachedE5FaissRetrievalEnv(
            retrieval_root=retrieval_root,
            embedding_model=embedding_model,
            device=device,
            top_k=top_k,
            max_length=max_length,
            batch_size=batch_size,
            query_cache_size=query_cache_size,
            engine_factory=e5_engine_factory,
            encoder_factory=e5_encoder_factory,
        )
    if normalized == "linear_rag":
        return CachedLinearRAGRetrievalEnv(
            retrieval_root=retrieval_root,
            embedding_model=embedding_model,
            spacy_model=spacy_model,
            top_k=top_k,
            max_workers=max_workers,
            batch_size=batch_size,
            use_vectorized_retrieval=use_vectorized_retrieval,
            query_cache_size=query_cache_size,
        )
    raise ValueError(
        f"Unknown retrieval backend {backend!r}; expected one of: e5_faiss, linear_rag"
    )


def validate_retrieval_assets(
    *,
    backend: str,
    retrieval_root: str | Path,
    datasets: list[str] | set[str] | tuple[str, ...],
    embedding_model: str,
) -> None:
    normalized = str(backend).strip().casefold()
    dataset_names = sorted({str(item).strip() for item in datasets if str(item).strip()})
    if normalized == "e5_faiss":
        for dataset in dataset_names:
            validate_e5_faiss_assets(
                retrieval_root=retrieval_root,
                dataset=dataset,
                expected_model=embedding_model,
            )
        return
    if normalized == "linear_rag":
        required_files = (
            "passage_embedding.parquet",
            "entity_embedding.parquet",
            "sentence_embedding.parquet",
            "LinearRAG.graphml",
        )
        missing = [
            Path(retrieval_root) / dataset / filename
            for dataset in dataset_names
            for filename in required_files
            if not (Path(retrieval_root) / dataset / filename).exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing retrieval assets: " + ", ".join(str(path) for path in missing)
            )
        return
    raise ValueError(
        f"Unknown retrieval backend {backend!r}; expected one of: e5_faiss, linear_rag"
    )
