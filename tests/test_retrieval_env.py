from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_processing.retrieval import build_linearrag_assets
from data_processing.retrieval import _chunk_text_from_row, _resolve_example_path
from data_processing.e5_faiss import (
    E5Encoder,
    E5FaissQueryEngine,
    build_e5_faiss_index,
    validate_e5_faiss_assets,
)
from rl_training.retrieval import CachedE5FaissRetrievalEnv, create_retrieval_env


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_build_assets_supports_dataset_prefix_split_files(tmp_path: Path) -> None:
    source = tmp_path / "data" / "processed" / "toyqa"
    _write_jsonl(
        source / "toyqa_train.jsonl",
        [
            {
                "qid": "t1",
                "question": "Who leads the city?",
                "answer": "Ada",
                "question_type": "bridge",
                "hop_count": 1,
            }
        ],
    )
    _write_jsonl(
        source / "toyqa_dev.jsonl",
        [
            {
                "qid": "t2",
                "question": "Where is the city?",
                "answer": "Nowhere",
                "question_type": "bridge",
                "hop_count": 1,
            }
        ],
    )
    _write_jsonl(
        source / "corpus.jsonl",
        [
            {
                "chunk_id": "toyqa:chunk:0",
                "doc_id": "d0",
                "dataset": "toyqa",
                "title": "City Hall",
                "text": "The city is led by Ada.",
                "source": "manual",
            }
        ],
    )

    summary = build_linearrag_assets(
        processed_root=tmp_path / "data" / "processed",
        retrieval_root=tmp_path / "data" / "retrieval",
        datasets=["toyqa"],
        splits=["train", "dev"],
    )

    assert summary["toyqa"]["questions"] == 2
    assert summary["toyqa"]["chunks"] == 1
    assert _resolve_example_path(source, "toyqa", "train").name == "toyqa_train.jsonl"

    target = tmp_path / "data" / "retrieval" / "toyqa"
    assert (target / "questions.json").exists()
    assert (target / "chunks.json").exists()
    assert (target / "chunk_metadata.jsonl").exists()


def test_chunk_text_falls_back_to_sentences(tmp_path: Path) -> None:
    row = {
        "title": "T",
        "sentences": [
            {"text": "First sentence."},
            {"text": "Second sentence."},
        ],
    }
    assert (
        _chunk_text_from_row(row)
        == "First sentence.\nSecond sentence."
    )


def test_retrieval_scripts_use_data_processing_entrypoints() -> None:
    root = Path(__file__).resolve().parents[1]
    build_script = (root / "scripts" / "build_retrieval.sh").read_text(encoding="utf-8")
    query_script = (root / "scripts" / "query_retrieval.sh").read_text(encoding="utf-8")
    cli = (root / "src" / "data_processing" / "retrieval_cli.py").read_text(
        encoding="utf-8"
    )

    assert "-m data_processing.retrieval_cli" in build_script
    assert "config/retrieval_eval.yml" in build_script
    assert '"retrieval_eval.yml"' in cli
    assert "python -m data_processing.retrieval_cli" in query_script
    assert "config/query_retrieval.yml" in query_script


class _FakeTokenizer:
    def __init__(self) -> None:
        self.inputs: list[list[str]] = []

    def __call__(self, texts, **kwargs):
        self.inputs.append(list(texts))
        assert kwargs["max_length"] == 512
        assert kwargs["padding"] is True
        assert kwargs["truncation"] is True
        return {
            "input_ids": torch.tensor([[1, 2], [3, 0]][: len(texts)]),
            "attention_mask": torch.tensor([[1, 1], [1, 0]][: len(texts)]),
        }


class _FakeModel:
    def eval(self):
        return self

    def to(self, device):
        assert device == "cpu"
        return self

    def __call__(self, **inputs):
        hidden = torch.tensor(
            [
                [[3.0, 0.0], [0.0, 4.0]],
                [[0.0, 5.0], [100.0, 100.0]],
            ][: inputs["input_ids"].shape[0]]
        )
        return SimpleNamespace(last_hidden_state=hidden)


def test_e5_encoder_prefixes_masks_and_normalizes() -> None:
    tokenizer = _FakeTokenizer()
    encoder = E5Encoder(
        model_name="intfloat/e5-base-v2",
        device="cpu",
        max_length=512,
        tokenizer=tokenizer,
        model=_FakeModel(),
    )

    query_vectors = encoder.encode_queries(["Who directed Bullitt?"])
    passage_vectors = encoder.encode_passages(["Bullitt", "Peter Yates"])

    assert tokenizer.inputs == [
        ["query: Who directed Bullitt?"],
        ["passage: Bullitt", "passage: Peter Yates"],
    ]
    assert query_vectors.dtype == np.float32
    assert query_vectors.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(query_vectors, axis=1), [1.0])
    np.testing.assert_allclose(np.linalg.norm(passage_vectors, axis=1), [1.0, 1.0])
    np.testing.assert_allclose(passage_vectors[1], [0.0, 1.0])


class _FakeIndexFlatIP:
    def __init__(self, dimension: int) -> None:
        self.d = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)

    @property
    def ntotal(self) -> int:
        return len(self.vectors)

    def add(self, vectors: np.ndarray) -> None:
        self.vectors = np.asarray(vectors, dtype=np.float32)

    def search(self, vectors: np.ndarray, k: int):
        similarities = np.asarray(vectors, dtype=np.float32) @ self.vectors.T
        order = np.argsort(-similarities, axis=1)[:, :k]
        scores = np.take_along_axis(similarities, order, axis=1)
        return scores.astype(np.float32), order.astype(np.int64)


class _FakeFaiss:
    def __init__(self) -> None:
        self.indexes: dict[str, _FakeIndexFlatIP] = {}

    def IndexFlatIP(self, dimension: int) -> _FakeIndexFlatIP:
        return _FakeIndexFlatIP(dimension)

    def write_index(self, index: _FakeIndexFlatIP, path: str) -> None:
        self.indexes[str(path)] = index
        Path(path).write_bytes(b"fake-faiss")

    def read_index(self, path: str) -> _FakeIndexFlatIP:
        if str(path) in self.indexes:
            return self.indexes[str(path)]
        target = Path(path)
        return self.indexes[str(target.with_name(f".{target.name}.tmp"))]


class _LookupEncoder:
    dimension = 2

    @staticmethod
    def encode_passages(texts: list[str]) -> np.ndarray:
        assert texts == ['"Alpha"\nfirst', '"Beta"\nsecond']
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    @staticmethod
    def encode_queries(texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[1.0, 0.0] if "alpha" in text.casefold() else [0.0, 1.0] for text in texts],
            dtype=np.float32,
        )


def test_e5_faiss_build_query_and_hash_validation(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    retrieval_root = tmp_path / "indexes"
    _write_jsonl(
        data_root / "toy" / "corpus.jsonl",
        [
            {"chunk_id": "c0", "doc_id": "d0", "title": "Alpha", "text": "first"},
            {"chunk_id": "c1", "doc_id": "d1", "title": "Beta", "text": "second"},
        ],
    )
    fake_faiss = _FakeFaiss()

    summary = build_e5_faiss_index(
        data_root=data_root,
        output_root=retrieval_root,
        dataset="toy",
        model_name="intfloat/e5-base-v2",
        device="cpu",
        encoder=_LookupEncoder(),
        faiss_module=fake_faiss,
    )

    assert summary["corpus_count"] == 2
    assert summary["dimension"] == 2
    metadata = validate_e5_faiss_assets(
        retrieval_root=retrieval_root,
        dataset="toy",
        expected_model="intfloat/e5-base-v2",
        faiss_module=fake_faiss,
    )
    assert metadata["faiss_type"] == "IndexFlatIP"

    engine = E5FaissQueryEngine(
        retrieval_root=retrieval_root,
        dataset="toy",
        model_name="intfloat/e5-base-v2",
        device="cpu",
        top_k=2,
        encoder=_LookupEncoder(),
        faiss_module=fake_faiss,
    )
    result = engine.query("alpha question")
    assert [item["passage_id"] for item in result.passages] == [0, 1]
    assert result.passages[0]["title"] == "Alpha"
    assert result.scores == [1.0, 0.0]

    with (retrieval_root / "toy" / "corpus.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="corpus SHA256 mismatch"):
        validate_e5_faiss_assets(
            retrieval_root=retrieval_root,
            dataset="toy",
            faiss_module=fake_faiss,
        )


def test_e5_faiss_build_resumes_after_last_persisted_batch(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    retrieval_root = tmp_path / "indexes"
    rows = [
        {"title": f"Title {index}", "text": f"text {index}"}
        for index in range(5)
    ]
    _write_jsonl(data_root / "toy" / "corpus.jsonl", rows)
    fake_faiss = _FakeFaiss()

    class InterruptingEncoder:
        dimension = 2

        def encode_passage_batches(self, texts, *, start=0, **kwargs):
            assert start == 0
            yield 2, np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        build_e5_faiss_index(
            data_root=data_root,
            output_root=retrieval_root,
            dataset="toy",
            encoder=InterruptingEncoder(),
            faiss_module=fake_faiss,
        )

    state_path = retrieval_root / "toy" / ".e5_build_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["encoded_count"] == 2

    starts: list[int] = []

    class ResumingEncoder:
        dimension = 2

        def encode_passage_batches(self, texts, *, start=0, **kwargs):
            starts.append(start)
            yield 5, np.asarray(
                [[1.0, 1.0], [0.5, 0.5], [0.25, 0.75]], dtype=np.float32
            )

    summary = build_e5_faiss_index(
        data_root=data_root,
        output_root=retrieval_root,
        dataset="toy",
        encoder=ResumingEncoder(),
        faiss_module=fake_faiss,
    )

    assert starts == [2]
    assert summary["corpus_count"] == 5
    assert not state_path.exists()
    assert not (retrieval_root / "toy" / ".e5_embeddings.npy").exists()
    index = fake_faiss.read_index(str(retrieval_root / "toy" / "e5_Flat.index"))
    assert index.ntotal == 5


def test_e5_faiss_resume_rejects_changed_source_corpus(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    retrieval_root = tmp_path / "indexes"
    source_path = data_root / "toy" / "corpus.jsonl"
    _write_jsonl(source_path, [{"title": "A", "text": "first"}])

    class InterruptingEncoder:
        dimension = 2

        def encode_passage_batches(self, texts, *, start=0, **kwargs):
            yield 1, np.asarray([[1.0, 0.0]], dtype=np.float32)
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        build_e5_faiss_index(
            data_root=data_root,
            output_root=retrieval_root,
            dataset="toy",
            encoder=InterruptingEncoder(),
            faiss_module=_FakeFaiss(),
        )

    _write_jsonl(source_path, [{"title": "B", "text": "changed"}])
    with pytest.raises(RuntimeError, match="resume contract mismatch"):
        build_e5_faiss_index(
            data_root=data_root,
            output_root=retrieval_root,
            dataset="toy",
            encoder=InterruptingEncoder(),
            faiss_module=_FakeFaiss(),
        )


def test_e5_faiss_complete_index_is_validated_and_skipped(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    retrieval_root = tmp_path / "indexes"
    _write_jsonl(data_root / "toy" / "corpus.jsonl", [{"title": "A", "text": "first"}])
    fake_faiss = _FakeFaiss()

    class OneBatchEncoder:
        dimension = 2
        calls = 0

        def encode_passage_batches(self, texts, *, start=0, **kwargs):
            self.calls += 1
            yield 1, np.asarray([[1.0, 0.0]], dtype=np.float32)

    encoder = OneBatchEncoder()
    build_e5_faiss_index(
        data_root=data_root,
        output_root=retrieval_root,
        dataset="toy",
        encoder=encoder,
        faiss_module=fake_faiss,
    )
    skipped = build_e5_faiss_index(
        data_root=data_root,
        output_root=retrieval_root,
        dataset="toy",
        encoder=encoder,
        faiss_module=fake_faiss,
    )

    assert encoder.calls == 1
    assert skipped["build_status"] == "skipped"


def test_e5_faiss_runtime_factory_deduplicates_cached_batch_queries(tmp_path: Path) -> None:
    created: list[str] = []
    queried: list[list[str]] = []

    class FakeResult:
        def __init__(self, query: str) -> None:
            self.query = query
            self.passages = [{"passage_id": 0, "title": "T", "text": query}]
            self.scores = [1.0]

    class FakeEngine:
        def __init__(self, *, dataset: str, **kwargs) -> None:
            created.append(dataset)

        def query_batch(self, queries: list[str]) -> list[FakeResult]:
            queried.append(list(queries))
            return [FakeResult(query) for query in queries]

    env = create_retrieval_env(
        backend="e5_faiss",
        retrieval_root=tmp_path,
        embedding_model="intfloat/e5-base-v2",
        device="cpu",
        top_k=5,
        max_length=512,
        batch_size=32,
        query_cache_size=8,
        e5_engine_factory=FakeEngine,
        e5_encoder_factory=lambda **kwargs: object(),
    )

    first = env.query_batch("hotpotqa", ["same query", "same query"])
    second = env.query_batch("hotpotqa", ["same query"])

    assert created == ["hotpotqa"]
    assert queried == [["same query"]]
    assert first[0] == first[1] == second[0]
    assert env.stats() == {"cache_hits": 2, "cache_misses": 1, "time_retrieval_seconds": pytest.approx(env.stats()["time_retrieval_seconds"])}


def test_e5_faiss_observation_exposes_local_ids_and_only_agent_fields() -> None:
    result = SimpleNamespace(
        query="director",
        passages=[
            {
                "passage_id": 5062,
                "chunk_id": "2wiki:chunk:76574",
                "doc_id": "2wiki:document:a",
                "dataset": "2wiki",
                "title": "2001 Maniacs: Field of Screams",
                "text": "The film was directed by Tim Sullivan.",
                "sentences": [{"sent_id": 0, "text": "The film was directed by Tim Sullivan."}],
            },
            {
                "passage_id": 1490,
                "chunk_id": "2wiki:chunk:9192",
                "title": "Tim Sullivan",
                "text": "Tim Sullivan is an American film director.",
            },
        ],
        scores=[0.893, 0.779],
    )

    observation = CachedE5FaissRetrievalEnv._observation(result)

    assert observation == {
        "query": "director",
        "passages": [
            {
                "passage_id": 0,
                "title": "2001 Maniacs: Field of Screams",
                "text": "The film was directed by Tim Sullivan.",
                "score": 0.893,
            },
            {
                "passage_id": 1,
                "title": "Tim Sullivan",
                "text": "Tim Sullivan is an American film director.",
                "score": 0.779,
            },
        ],
    }


def test_e5_retrieval_cli_dispatches_native_index_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from data_processing import retrieval_cli

    config = tmp_path / "build.yml"
    config.write_text(
        "\n".join(
            [
                "command: build",
                "backend: e5_faiss",
                f'data_root: "{tmp_path / "source"}"',
                f'retrieval_root: "{tmp_path / "indexes"}"',
                "datasets: [2wiki, hotpotqa]",
                "embedding_model: intfloat/e5-base-v2",
                "device: cpu",
                "max_length: 512",
                "batch_size: 16",
            ]
        ),
        encoding="utf-8",
    )
    calls: list[dict] = []
    shared_encoder = object()

    def fake_build(**kwargs):
        calls.append(kwargs)
        return {
            "dataset": kwargs["dataset"],
            "corpus_count": 1,
            "build_status": "skipped" if kwargs["dataset"] == "2wiki" else "built",
        }

    monkeypatch.setattr(retrieval_cli, "build_e5_faiss_index", fake_build)
    monkeypatch.setattr(retrieval_cli, "E5Encoder", lambda **kwargs: shared_encoder)

    assert retrieval_cli.main(["--config", str(config)]) == 0
    assert [item["dataset"] for item in calls] == ["2wiki", "hotpotqa"]
    assert all(item["model_name"] == "intfloat/e5-base-v2" for item in calls)
    assert all(item["device"] == "cpu" for item in calls)
    assert all(item["encoder"] is shared_encoder for item in calls)
    output = capsys.readouterr().out
    assert "2wiki skipped" in output
    assert "hotpotqa complete" in output
