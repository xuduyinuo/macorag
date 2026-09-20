from __future__ import annotations

from array import array
from concurrent.futures import Future
from dataclasses import dataclass
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import time
from typing import Any

from tqdm.auto import tqdm


@dataclass(frozen=True)
class RetrievedPassage:
    pointer: str
    corpus_id: str
    title: str
    text: str
    score: float
    corpus_row: int

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "passage_id": self.pointer,
            "corpus_id": self.corpus_id,
            "title": self.title,
            "text": self.text,
            "score": self.score,
        }


def split_contents(contents: Any) -> tuple[str, str]:
    lines = str(contents or "").splitlines()
    if not lines:
        return "", ""
    title = lines[0].strip().strip('"')
    text = "\n".join(lines[1:]).strip()
    if not text:
        return "", title
    return title, text


class JsonlOffsetStore:
    """Random-access JSONL reader backed by a compact uint64 offset sidecar."""

    def __init__(self, corpus_path: str | Path, offsets_path: str | Path) -> None:
        self.corpus_path = Path(corpus_path)
        self.offsets_path = Path(offsets_path)
        self.metadata_path = self.offsets_path.with_suffix(self.offsets_path.suffix + ".json")
        self._fd: int | None = None
        self._offsets: Any = None

    def _expected_metadata(self) -> dict[str, Any]:
        stat = self.corpus_path.stat()
        return {"schema_version": 1, "corpus_size": stat.st_size, "corpus_mtime_ns": stat.st_mtime_ns}

    def ensure_offsets(self) -> None:
        expected = self._expected_metadata()
        if self.offsets_path.is_file() and self.metadata_path.is_file():
            try:
                metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                metadata = {}
            if all(metadata.get(key) == value for key, value in expected.items()):
                print(
                    f"[rl-v2:init] Reusing corpus offsets: {self.offsets_path}",
                    file=sys.stderr,
                    flush=True,
                )
                return
        print(
            f"[rl-v2:init] Building corpus offsets from {self.corpus_path}",
            file=sys.stderr,
            flush=True,
        )
        self.offsets_path.parent.mkdir(parents=True, exist_ok=True)
        temp_offsets = self.offsets_path.with_name("." + self.offsets_path.name + ".tmp")
        temp_metadata = self.metadata_path.with_name("." + self.metadata_path.name + ".tmp")
        offsets = array("Q")
        position = 0
        row_count = 0
        with (
            self.corpus_path.open("rb") as source,
            temp_offsets.open("wb") as target,
            tqdm(
                total=expected["corpus_size"],
                desc="RL-v2 corpus offsets",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                dynamic_ncols=True,
            ) as progress,
        ):
            offsets.append(0)
            for line in source:
                line_size = len(line)
                position += line_size
                offsets.append(position)
                row_count += 1
                progress.update(line_size)
                if len(offsets) >= 1_000_000:
                    offsets.tofile(target)
                    offsets = array("Q")
            if offsets:
                offsets.tofile(target)
            target.flush()
            os.fsync(target.fileno())
        temp_metadata.write_text(
            json.dumps({**expected, "row_count": row_count}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_offsets.replace(self.offsets_path)
        temp_metadata.replace(self.metadata_path)
        print(
            f"[rl-v2:init] Corpus offsets ready: {self.offsets_path} "
            f"({row_count:,} rows)",
            file=sys.stderr,
            flush=True,
        )

    def open(self) -> None:
        self.ensure_offsets()
        import numpy as np

        self._offsets = np.memmap(self.offsets_path, dtype="<u8", mode="r")
        self._fd = os.open(self.corpus_path, os.O_RDONLY)

    @property
    def row_count(self) -> int:
        if self._offsets is None:
            raise RuntimeError("Corpus store is not open")
        return max(0, len(self._offsets) - 1)

    def get(self, row_index: int) -> dict[str, Any]:
        if self._offsets is None or self._fd is None:
            raise RuntimeError("Corpus store is not open")
        if row_index < 0 or row_index >= self.row_count:
            raise IndexError(row_index)
        start = int(self._offsets[row_index])
        end = int(self._offsets[row_index + 1])
        # pread is position-independent, so concurrent document reads do not
        # need to serialize around a shared seek pointer.
        return json.loads(os.pread(self._fd, end - start, start))

    def close(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None


@dataclass
class _QueuedSearch:
    query: str
    future: Future[list[RetrievedPassage]]


class E5QueryEncoder:
    def __init__(
        self, model_path: str, *, device: str, max_length: int,
        batch_size: int, use_fp16: bool,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True).eval().to(device)
        if use_fp16:
            self.model = self.model.half()

    def encode(self, queries: list[str]) -> Any:
        import numpy as np
        import torch.nn.functional as functional

        batches = []
        for start in range(0, len(queries), self.batch_size):
            values = [f"query: {item}" for item in queries[start : start + self.batch_size]]
            tokens = self.tokenizer(
                values,
                max_length=self.max_length,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in tokens.items()}
            with self.torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                pooled = functional.normalize(pooled, p=2, dim=-1)
            batches.append(pooled.float().cpu().numpy())
        return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)


class UnifiedE5FaissRetriever:
    def __init__(
        self,
        *,
        corpus_path: str | Path,
        offsets_path: str | Path,
        index_path: str | Path,
        manifest_path: str | Path,
        model_path: str,
        device: str,
        max_length: int,
        batch_size: int,
        batch_wait_ms: int = 1000,
        top_k: int,
        mmap: bool = True,
        use_fp16: bool = False,
    ) -> None:
        import faiss

        self.top_k = top_k
        self.retrieval_batch_size = batch_size
        self.batch_wait_seconds = float(batch_wait_ms) / 1000.0
        self._search_queue: Queue[_QueuedSearch | None] = Queue()
        self._closed = False
        self.retrieval_queries = 0
        self.retrieval_batches = 0
        self.max_observed_batch_size = 0
        self.manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        index_size = Path(index_path).stat().st_size
        print(
            f"[rl-v2:init] Loading FAISS index: {index_path} "
            f"({index_size / (1024 ** 3):.1f} GiB, mmap={mmap})",
            file=sys.stderr,
            flush=True,
        )
        flags = 0
        if mmap:
            flags = int(getattr(faiss, "IO_FLAG_MMAP", 0)) | int(getattr(faiss, "IO_FLAG_READ_ONLY", 0))
        if flags:
            try:
                self.index = faiss.read_index(str(index_path), flags)
            except RuntimeError:
                # Some FAISS builds/index types do not support mmap flags.
                self.index = faiss.read_index(str(index_path))
        else:
            self.index = faiss.read_index(str(index_path))
        print(
            f"[rl-v2:init] FAISS index ready: ntotal={int(self.index.ntotal):,}, "
            f"dimension={int(self.index.d)}",
            file=sys.stderr,
            flush=True,
        )
        self.corpus = JsonlOffsetStore(corpus_path, offsets_path)
        self.corpus.open()
        expected_rows = int(self.manifest.get("ntotal", -1))
        if int(self.index.ntotal) != expected_rows or self.corpus.row_count != expected_rows:
            raise RuntimeError(
                "Unified retrieval contract mismatch: "
                f"index={self.index.ntotal}, corpus={self.corpus.row_count}, manifest={expected_rows}"
            )
        expected_dim = int(self.manifest.get("dimension", -1))
        if int(self.index.d) != expected_dim:
            raise RuntimeError(f"Index dimension mismatch: index={self.index.d}, manifest={expected_dim}")
        print(
            f"[rl-v2:init] Loading E5 query encoder on {device}: {model_path}",
            file=sys.stderr,
            flush=True,
        )
        self.encoder = E5QueryEncoder(
            model_path,
            device=device,
            max_length=max_length,
            batch_size=batch_size,
            use_fp16=use_fp16,
        )
        self._batch_worker = threading.Thread(
            target=self._batch_worker_loop,
            name="rl-v2-retrieval-batcher",
            daemon=True,
        )
        self._batch_worker.start()
        print("[rl-v2:init] Unified retriever ready", file=sys.stderr, flush=True)

    def search(self, query: str) -> list[RetrievedPassage]:
        if self._closed:
            raise RuntimeError("Retriever is closed")
        future: Future[list[RetrievedPassage]] = Future()
        self._search_queue.put(_QueuedSearch(query=str(query), future=future))
        return future.result()

    def _batch_worker_loop(self) -> None:
        while True:
            first = self._search_queue.get()
            if first is None:
                return
            requests = [first]
            deadline = time.monotonic() + self.batch_wait_seconds
            while len(requests) < self.retrieval_batch_size:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    item = self._search_queue.get(timeout=timeout)
                except Empty:
                    break
                if item is None:
                    # close() is only called after sample workers finish, but
                    # preserve the sentinel if shutdown races with collection.
                    self._search_queue.put(None)
                    break
                requests.append(item)
            try:
                results = self._search_batch([item.query for item in requests])
            except BaseException as exc:
                for item in requests:
                    item.future.set_exception(exc)
            else:
                for item, result in zip(requests, results):
                    item.future.set_result(result)

    def _search_batch(self, queries: list[str]) -> list[list[RetrievedPassage]]:
        vectors = self.encoder.encode(queries)
        scores, indices = self.index.search(vectors, min(self.top_k, int(self.index.ntotal)))
        batch_output: list[list[RetrievedPassage]] = []
        for row_scores, row_indices in zip(scores.tolist(), indices.tolist()):
            output: list[RetrievedPassage] = []
            for rank, (score, row_index) in enumerate(zip(row_scores, row_indices)):
                if row_index < 0:
                    continue
                row = self.corpus.get(int(row_index))
                title, text = split_contents(row.get("contents") or row.get("text"))
                output.append(
                    RetrievedPassage(
                        pointer=f"P{rank}",
                        corpus_id=str(row.get("id") or row_index),
                        title=title,
                        text=text,
                        score=float(score),
                        corpus_row=int(row_index),
                    )
                )
            batch_output.append(output)
        self.retrieval_queries += len(queries)
        self.retrieval_batches += 1
        self.max_observed_batch_size = max(self.max_observed_batch_size, len(queries))
        return batch_output

    def batch_stats(self) -> dict[str, float | int]:
        return {
            "queries": self.retrieval_queries,
            "batches": self.retrieval_batches,
            "average_batch_size": self.retrieval_queries / max(1, self.retrieval_batches),
            "max_batch_size": self.max_observed_batch_size,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._search_queue.put(None)
        self._batch_worker.join(timeout=10)
        self.corpus.close()


class RetrievalEnvironment:
    """Lazy unified Wiki18 E5-FAISS environment used by all v2 datasets."""

    def __init__(
        self, *, corpus_path: str | Path, offsets_path: str | Path,
        index_path: str | Path, manifest_path: str | Path, model_path: str,
        device: str, max_length: int, batch_size: int, batch_wait_ms: int,
        top_k: int, mmap: bool,
        use_fp16: bool, max_passage_chars: int,
    ) -> None:
        self.settings = {
            "corpus_path": corpus_path,
            "offsets_path": offsets_path,
            "index_path": index_path,
            "manifest_path": manifest_path,
            "model_path": model_path,
            "device": device,
            "max_length": max_length,
            "batch_size": batch_size,
            "batch_wait_ms": batch_wait_ms,
            "top_k": top_k,
            "mmap": mmap,
            "use_fp16": use_fp16,
        }
        self.max_passage_chars = int(max_passage_chars)
        self._retriever: UnifiedE5FaissRetriever | None = None

    def validate(self, datasets: set[str] | None = None) -> None:
        del datasets
        required = (
            "corpus_path", "offsets_path", "index_path", "manifest_path", "model_path",
        )
        for name in required:
            path = Path(self.settings[name])
            if not path.exists():
                raise FileNotFoundError(f"Missing unified retrieval asset {name}: {path}")
        manifest = json.loads(Path(self.settings["manifest_path"]).read_text(encoding="utf-8"))
        if int(manifest.get("ntotal", 0)) <= 0 or int(manifest.get("dimension", 0)) <= 0:
            raise ValueError("Unified index manifest requires positive ntotal and dimension")

    def _instance(self) -> UnifiedE5FaissRetriever:
        if self._retriever is None:
            self._retriever = UnifiedE5FaissRetriever(**self.settings)
        return self._retriever

    def initialize(self) -> None:
        """Eagerly load E5, FAISS and corpus offsets before timed rollouts."""
        self.validate()
        self._instance()

    def query(self, dataset: str, query: str) -> dict[str, Any]:
        del dataset
        ranked = self._instance().search(query)
        # Evaluation preserves FAISS rank: P0 is rank 0, P1 is rank 1, etc.
        payload = []
        for index, item in enumerate(ranked):
            payload.append({
                "passage_id": index,
                "corpus_passage_id": item.corpus_id,
                "corpus_row": item.corpus_row,
                "title": item.title,
                "text": item.text[:self.max_passage_chars],
                "score": item.score,
            })
        return {"query": query, "passages": payload}

    def batch_stats(self) -> dict[str, float | int]:
        if self._retriever is None:
            return {
                "queries": 0, "batches": 0, "average_batch_size": 0.0,
                "max_batch_size": 0,
            }
        return self._retriever.batch_stats()

    def close(self) -> None:
        if self._retriever is not None:
            self._retriever.close()
