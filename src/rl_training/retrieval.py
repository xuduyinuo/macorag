from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
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
