from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


class RetrievalEnvironment:
    """Self-contained E5-FAISS or lexical BM25 retrieval."""

    def __init__(self, *, root: str | Path, backend: str, embedding_model: str, device: str, max_length: int, batch_size: int, top_k: int) -> None:
        self.root = Path(root)
        self.backend = backend
        self.embedding_model = embedding_model
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.top_k = top_k
        self._corpora: dict[str, list[dict[str, Any]]] = {}
        self._indices: dict[str, Any] = {}
        self._encoder: tuple[Any, Any] | None = None
        self._bm25: dict[str, tuple[list[Counter[str]], Counter[str], float]] = {}

    def validate(self, datasets: set[str]) -> None:
        for dataset in datasets:
            folder = self.root / dataset
            corpus = folder / "corpus.jsonl"
            if not corpus.is_file():
                raise FileNotFoundError(f"Missing retrieval corpus: {corpus}")
            if self.backend == "e5_faiss":
                for name in ("e5_Flat.index", "index_metadata.json"):
                    if not (folder / name).is_file():
                        raise FileNotFoundError(f"Missing E5-FAISS asset: {folder / name}")
                metadata = json.loads((folder / "index_metadata.json").read_text(encoding="utf-8"))
                actual = str(
                    metadata.get("retriever_model")
                    or metadata.get("model_name")
                    or metadata.get("embedding_model")
                    or ""
                )
                if actual and actual != self.embedding_model:
                    raise ValueError(f"Embedding model mismatch for {dataset}: index={actual}, config={self.embedding_model}")

    def _corpus(self, dataset: str) -> list[dict[str, Any]]:
        if dataset not in self._corpora:
            rows = []
            with (self.root / dataset / "corpus.jsonl").open("r", encoding="utf-8") as handle:
                for index, line in enumerate(handle):
                    row = json.loads(line)
                    row["passage_id"] = int(row.get("passage_id", index))
                    rows.append(row)
            self._corpora[dataset] = rows
        return self._corpora[dataset]

    def _e5(self) -> tuple[Any, Any]:
        if self._encoder is None:
            import torch
            from transformers import AutoModel, AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.embedding_model, local_files_only=True)
            model = AutoModel.from_pretrained(self.embedding_model, local_files_only=True).eval().to(self.device)
            self._encoder = (tokenizer, model)
        return self._encoder

    def _encode_query(self, query: str) -> Any:
        import numpy as np
        import torch
        import torch.nn.functional as functional
        tokenizer, model = self._e5()
        batch = tokenizer([f"query: {query}"], max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        batch = {key: value.to(self.device) for key, value in batch.items()}
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            pooled = functional.normalize(pooled, p=2, dim=-1)
        return np.ascontiguousarray(pooled.float().cpu().numpy())

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[\w]+", text.casefold())

    def _bm25_index(self, dataset: str) -> tuple[list[Counter[str]], Counter[str], float]:
        if dataset not in self._bm25:
            documents = [Counter(self._tokens(f"{row.get('title', '')} {row.get('text', '')}")) for row in self._corpus(dataset)]
            df: Counter[str] = Counter()
            for document in documents:
                df.update(document.keys())
            avgdl = sum(map(lambda x: sum(x.values()), documents)) / max(1, len(documents))
            self._bm25[dataset] = documents, df, avgdl
        return self._bm25[dataset]

    def query(self, dataset: str, query: str) -> dict[str, Any]:
        corpus = self._corpus(dataset)
        if self.backend == "e5_faiss":
            import faiss
            if dataset not in self._indices:
                self._indices[dataset] = faiss.read_index(str(self.root / dataset / "e5_Flat.index"))
            scores, indices = self._indices[dataset].search(self._encode_query(query), self.top_k)
            pairs = [(int(i), float(s)) for i, s in zip(indices[0], scores[0]) if int(i) >= 0]
        else:
            documents, df, avgdl = self._bm25_index(dataset)
            terms = self._tokens(query)
            n = len(documents)
            ranked = []
            for index, document in enumerate(documents):
                dl = sum(document.values())
                score = 0.0
                for term in terms:
                    freq = document[term]
                    if not freq:
                        continue
                    idf = math.log(1.0 + (n - df[term] + 0.5) / (df[term] + 0.5))
                    score += idf * freq * 2.5 / (freq + 1.5 * (1.0 - 0.75 + 0.75 * dl / max(avgdl, 1.0)))
                ranked.append((index, score))
            pairs = sorted(ranked, key=lambda x: x[1], reverse=True)[:self.top_k]
        passages = []
        for local_id, (index, score) in enumerate(pairs):
            row = dict(corpus[index])
            row["corpus_passage_id"] = row.get("passage_id", index)
            row["passage_id"] = local_id
            row["score"] = score
            passages.append(row)
        return {"query": query, "passages": passages}
