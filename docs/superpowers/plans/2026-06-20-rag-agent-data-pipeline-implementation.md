# RAG 多智能体数据流水线实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 构建统一数据处理、LinearRAG 多轮检索环境、全 API 教师轨迹生成与过滤流水线，产出每个数据集 1K、总计 3K 的 Qwen2.5-7B SFT 轨迹样本。

**架构：** 先把 HotpotQA、2Wiki、MuSiQue 转成 canonical `examples.*.jsonl` 和 `corpus.jsonl`，再派生 LinearRAG 兼容的 `questions.json/chunks.json/chunk_meta.jsonl/qrels.jsonl`。教师模型只能通过 step-wise 检索环境主动检索，gold evidence 只进入 verifier/filter，用于过滤高质量 SFT 轨迹。

**技术栈：** Python 3.10+，标准库 `json/dataclasses/hashlib/argparse/xml.etree`，`pyarrow` 或 `pandas` 读取 parquet，`pytest` 测试，外部 LinearRAG 仓库作为检索方案参考。

---

## 文件结构

- 创建：`src/macorag/__init__.py`
  - 包入口。
- 创建：`src/macorag/schemas.py`
  - canonical example/corpus、LinearRAG sidecar、trajectory 相关 dataclass 和 JSON 序列化。
- 创建：`src/macorag/io_utils.py`
  - JSONL、JSON、parquet 读取写入、文本规范化、sha1、目录创建。
- 创建：`src/macorag/dataset_builders.py`
  - HotpotQA、2Wiki、MuSiQue 到 canonical schema 的转换。
- 创建：`src/macorag/linearrag_adapter.py`
  - 从 canonical 数据生成 `linearrag_dataset/<dataset>/questions.json/chunks.json/chunk_meta.jsonl/qrels.jsonl`。
- 创建：`src/macorag/retrieval_env.py`
  - Step-wise 检索环境接口：`reset(qid)`、`step(query, top_k)`、`get_state()`。
- 创建：`src/macorag/teacher_protocol.py`
  - `<plan>`、`<retrieval>`、`<update-evidence>`、`<answer>` 标签输出解析和验证。
- 创建：`src/macorag/teacher_api.py`
  - 教师 API 客户端抽象、OpenAI-compatible HTTP 实现、离线 fake client。
- 创建：`src/macorag/trajectory_builder.py`
  - 使用检索环境和教师客户端生成 raw teacher trajectories。
- 创建：`src/macorag/trajectory_filter.py`
  - 基于 qrels、chunk_meta、gold answer 的轨迹过滤器。
- 创建：`src/macorag/sampling.py`
  - 每数据集 1K 目标的分层抽样。
- 创建：`src/macorag/cli.py`
  - 命令行入口：`build-canonical`、`build-linearrag`、`sample`、`generate-trajectories`、`filter-trajectories`。
- 创建：`tests/fixtures/`
  - 小型 JSONL/JSON fixture。
- 创建：`tests/test_schemas.py`
- 创建：`tests/test_dataset_builders.py`
- 创建：`tests/test_linearrag_adapter.py`
- 创建：`tests/test_retrieval_env.py`
- 创建：`tests/test_teacher_protocol.py`
- 创建：`tests/test_trajectory_filter.py`
- 创建：`pyproject.toml`
  - 最小包配置和 pytest 配置。
- 修改：`.gitignore`
  - 保留现有 `.codex/`、`.agents/`、`data/`，增加运行产物忽略项。

## 任务 1：建立 Python 包与基础 Schema

**文件：**
- 创建：`pyproject.toml`
- 创建：`src/macorag/__init__.py`
- 创建：`src/macorag/schemas.py`
- 创建：`tests/test_schemas.py`

- [ ] **步骤 1：编写失败的 schema 测试**

创建 `tests/test_schemas.py`：

```python
from macorag.schemas import CorpusDoc, Example, SupportingFact


def test_example_to_json_roundtrip():
    example = Example(
        qid="q1",
        dataset="2wiki",
        split="train",
        question="Who founded the company?",
        answer="Alice",
        answer_aliases=["A. Smith"],
        question_type="bridge",
        hop_count=2,
        supporting_facts=[
            SupportingFact(
                doc_id="d1",
                title="Alice",
                sent_id=0,
                text="Alice founded the company.",
                source="gold",
            )
        ],
        evidence_chain=[],
        context_doc_ids=["d1"],
        usable_for_sft=True,
        usable_for_retrieval_eval=True,
        quality_flags=[],
        metadata={"level": "medium"},
    )

    loaded = Example.from_json(example.to_json())

    assert loaded.qid == "q1"
    assert loaded.supporting_facts[0].title == "Alice"
    assert loaded.metadata["level"] == "medium"


def test_corpus_doc_chunk_text_includes_title():
    doc = CorpusDoc(
        doc_id="d1",
        dataset="hotpotqa",
        title="Document Title",
        text="Document body.",
        sentences=["Document body."],
        source="beir",
        metadata={},
    )

    assert doc.to_chunk_text() == "Document Title\nDocument body."
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_schemas.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag'`。

- [ ] **步骤 3：创建最小包配置**

创建 `pyproject.toml`：

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "macorag"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
  "pyarrow>=14",
  "pytest>=8",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

创建 `src/macorag/__init__.py`：

```python
"""MACORAG data processing package."""
```

- [ ] **步骤 4：实现 schema dataclass**

创建 `src/macorag/schemas.py`：

```python
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
import json


def _loads_json(value: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value


@dataclass
class SupportingFact:
    doc_id: str | None
    title: str
    sent_id: int | None
    text: str | None
    source: str = "gold"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SupportingFact":
        return cls(
            doc_id=data.get("doc_id"),
            title=data["title"],
            sent_id=data.get("sent_id"),
            text=data.get("text"),
            source=data.get("source", "gold"),
        )


@dataclass
class EvidenceStep:
    step: int
    sub_question: str | None
    answer: str | None
    support_doc_id: str | None
    support_title: str | None
    support_sent_ids: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceStep":
        return cls(
            step=int(data["step"]),
            sub_question=data.get("sub_question"),
            answer=data.get("answer"),
            support_doc_id=data.get("support_doc_id"),
            support_title=data.get("support_title"),
            support_sent_ids=list(data.get("support_sent_ids", [])),
        )


@dataclass
class Example:
    qid: str
    dataset: str
    split: str
    question: str
    answer: str | None
    answer_aliases: list[str]
    question_type: str | None
    hop_count: int
    supporting_facts: list[SupportingFact]
    evidence_chain: list[EvidenceStep]
    context_doc_ids: list[str]
    usable_for_sft: bool
    usable_for_retrieval_eval: bool
    quality_flags: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str | dict[str, Any]) -> "Example":
        data = _loads_json(value)
        return cls(
            qid=data["qid"],
            dataset=data["dataset"],
            split=data["split"],
            question=data["question"],
            answer=data.get("answer"),
            answer_aliases=list(data.get("answer_aliases", [])),
            question_type=data.get("question_type"),
            hop_count=int(data.get("hop_count", 0)),
            supporting_facts=[
                SupportingFact.from_dict(item)
                for item in data.get("supporting_facts", [])
            ],
            evidence_chain=[
                EvidenceStep.from_dict(item)
                for item in data.get("evidence_chain", [])
            ],
            context_doc_ids=list(data.get("context_doc_ids", [])),
            usable_for_sft=bool(data.get("usable_for_sft", False)),
            usable_for_retrieval_eval=bool(data.get("usable_for_retrieval_eval", False)),
            quality_flags=list(data.get("quality_flags", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class CorpusDoc:
    doc_id: str
    dataset: str
    title: str
    text: str
    sentences: list[str]
    source: str
    metadata: dict[str, Any]

    def to_chunk_text(self) -> str:
        return f"{self.title}\n{self.text}".strip()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, value: str | dict[str, Any]) -> "CorpusDoc":
        data = _loads_json(value)
        return cls(
            doc_id=data["doc_id"],
            dataset=data["dataset"],
            title=data.get("title", ""),
            text=data.get("text", ""),
            sentences=list(data.get("sentences", [])),
            source=data["source"],
            metadata=dict(data.get("metadata", {})),
        )
```

- [ ] **步骤 5：运行测试验证通过**

运行：`pytest tests/test_schemas.py -v`

预期：`2 passed`。

- [ ] **步骤 6：Commit**

```bash
git add pyproject.toml src/macorag/__init__.py src/macorag/schemas.py tests/test_schemas.py
git commit -m "feat: add canonical data schemas"
```

## 任务 2：实现 IO 工具和文本规范化

**文件：**
- 创建：`src/macorag/io_utils.py`
- 创建：`tests/test_io_utils.py`

- [ ] **步骤 1：编写失败的 IO 测试**

创建 `tests/test_io_utils.py`：

```python
from macorag.io_utils import normalize_text, sha1_text, read_jsonl, write_jsonl


def test_normalize_text_collapses_whitespace_but_keeps_case():
    assert normalize_text("  Alice\n  Smith\tFounded  X. ") == "Alice Smith Founded X."


def test_sha1_text_uses_normalized_text():
    assert sha1_text("Alice  Smith") == sha1_text(" Alice Smith ")


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "items.jsonl"
    write_jsonl(path, [{"id": "a"}, {"id": "b"}])

    assert list(read_jsonl(path)) == [{"id": "a"}, {"id": "b"}]
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_io_utils.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.io_utils'`。

- [ ] **步骤 3：实现 IO 工具**

创建 `src/macorag/io_utils.py`：

```python
from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any
import hashlib
import json
import re


_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return _WHITESPACE_RE.sub(" ", value).strip()


def normalize_key(value: str | None) -> str:
    return normalize_text(value).lower()


def sha1_text(value: str | None) -> str:
    return hashlib.sha1(normalize_text(value).encode("utf-8")).hexdigest()


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any, *, indent: int = 2) -> None:
    path = ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=indent)
        handle.write("\n")
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_io_utils.py -v`

预期：`3 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/macorag/io_utils.py tests/test_io_utils.py
git commit -m "feat: add data io utilities"
```

## 任务 3：实现数据集转换器

**文件：**
- 创建：`src/macorag/dataset_builders.py`
- 创建：`tests/test_dataset_builders.py`

- [ ] **步骤 1：编写失败的数据转换测试**

创建 `tests/test_dataset_builders.py`：

```python
from macorag.dataset_builders import build_musique_canonical_from_rows


def test_musique_answerable_rows_build_examples_and_corpus():
    rows = [
        {
            "id": "2hop__1_2",
            "question": "Who is the spouse of the Green performer?",
            "answer": "Miquette Giraudy",
            "answer_aliases": [],
            "answerable": True,
            "paragraphs": [
                {"idx": 0, "title": "Distractor", "paragraph_text": "Unused text."},
                {"idx": 5, "title": "Miquette Giraudy", "paragraph_text": "Miquette Giraudy is the spouse."},
                {"idx": 10, "title": "Steve Hillage", "paragraph_text": "Steve Hillage performed with Green."},
            ],
            "question_decomposition": [
                {
                    "id": 1,
                    "question": "Green >> performer",
                    "answer": "Steve Hillage",
                    "paragraph_support_idx": 10,
                },
                {
                    "id": 2,
                    "question": "#1 >> spouse",
                    "answer": "Miquette Giraudy",
                    "paragraph_support_idx": 5,
                },
            ],
        }
    ]

    examples, corpus, report = build_musique_canonical_from_rows(rows, split="train")

    assert report["errors"] == []
    assert len(examples) == 1
    assert examples[0].qid == "2hop__1_2"
    assert examples[0].hop_count == 2
    assert len(examples[0].supporting_facts) == 2
    assert len(examples[0].evidence_chain) == 2
    assert len(corpus) == 3
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_dataset_builders.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.dataset_builders'`。

- [ ] **步骤 3：实现 MuSiQue 转换核心**

创建 `src/macorag/dataset_builders.py`，先实现测试覆盖的函数：

```python
from __future__ import annotations

from collections import OrderedDict
from typing import Any

from macorag.io_utils import normalize_key, normalize_text, sha1_text
from macorag.schemas import CorpusDoc, EvidenceStep, Example, SupportingFact


def _empty_report() -> dict[str, list[dict[str, Any]]]:
    return {"errors": [], "warnings": [], "info": []}


def _musique_doc_id(title: str, text: str) -> str:
    return f"musique:{sha1_text(normalize_key(title) + ' ' + normalize_text(text))}"


def build_musique_canonical_from_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> tuple[list[Example], list[CorpusDoc], dict[str, Any]]:
    report = _empty_report()
    corpus_by_id: OrderedDict[str, CorpusDoc] = OrderedDict()
    examples: list[Example] = []

    for row in rows:
        qid = str(row.get("id", "")).strip()
        question = normalize_text(row.get("question"))
        answer = normalize_text(row.get("answer"))
        if not qid or not question:
            report["errors"].append({"qid": qid, "type": "invalid_example"})
            continue
        if not row.get("answerable", True):
            report["warnings"].append({"qid": qid, "type": "not_answerable"})
            continue

        paragraphs = {int(p["idx"]): p for p in row.get("paragraphs", [])}
        context_doc_ids: list[str] = []
        for paragraph in row.get("paragraphs", []):
            title = normalize_text(paragraph.get("title"))
            text = normalize_text(paragraph.get("paragraph_text"))
            if not title and not text:
                continue
            doc_id = _musique_doc_id(title, text)
            context_doc_ids.append(doc_id)
            if doc_id not in corpus_by_id:
                corpus_by_id[doc_id] = CorpusDoc(
                    doc_id=doc_id,
                    dataset="musique",
                    title=title,
                    text=text,
                    sentences=[text] if text else [],
                    source="musique_context",
                    metadata={"linked_qids": [qid], "paragraph_idx": paragraph.get("idx")},
                )
            else:
                linked = corpus_by_id[doc_id].metadata.setdefault("linked_qids", [])
                if qid not in linked:
                    linked.append(qid)

        evidence_chain: list[EvidenceStep] = []
        supporting_facts: list[SupportingFact] = []
        for step_index, item in enumerate(row.get("question_decomposition", []), start=1):
            support_idx = item.get("paragraph_support_idx")
            paragraph = paragraphs.get(int(support_idx)) if support_idx is not None else None
            support_doc_id = None
            support_title = None
            support_text = None
            if paragraph is not None:
                support_title = normalize_text(paragraph.get("title"))
                support_text = normalize_text(paragraph.get("paragraph_text"))
                support_doc_id = _musique_doc_id(support_title, support_text)
                supporting_facts.append(
                    SupportingFact(
                        doc_id=support_doc_id,
                        title=support_title,
                        sent_id=None,
                        text=support_text,
                        source="gold",
                    )
                )
            else:
                report["warnings"].append(
                    {"qid": qid, "type": "missing_support_paragraph", "step": step_index}
                )

            evidence_chain.append(
                EvidenceStep(
                    step=step_index,
                    sub_question=normalize_text(item.get("question")),
                    answer=normalize_text(item.get("answer")),
                    support_doc_id=support_doc_id,
                    support_title=support_title,
                    support_sent_ids=[],
                )
            )

        usable = bool(answer and supporting_facts)
        examples.append(
            Example(
                qid=qid,
                dataset="musique",
                split=split,
                question=question,
                answer=answer or None,
                answer_aliases=list(row.get("answer_aliases", [])),
                question_type=f"{len(evidence_chain)}hop",
                hop_count=len(evidence_chain),
                supporting_facts=supporting_facts,
                evidence_chain=evidence_chain,
                context_doc_ids=context_doc_ids,
                usable_for_sft=usable,
                usable_for_retrieval_eval=usable,
                quality_flags=[] if usable else ["missing_answer_or_support"],
                metadata={"answerable": True},
            )
        )

    return examples, list(corpus_by_id.values()), report
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_dataset_builders.py -v`

预期：`1 passed`。

- [ ] **步骤 5：添加 HotpotQA 和 2Wiki 纯函数转换测试**

扩展 `tests/test_dataset_builders.py`：

```python
from macorag.dataset_builders import (
    build_hotpot_example_from_row,
    build_2wiki_example_from_row,
)


def test_hotpot_row_maps_supporting_facts():
    row = {
        "id": "h1",
        "question": "Where was Alice born?",
        "answer": "Paris",
        "type": "bridge",
        "level": "easy",
        "supporting_facts": {"title": ["Alice"], "sent_id": [0]},
        "context": {"title": ["Alice"], "sentences": [["Alice was born in Paris."]]},
    }

    example = build_hotpot_example_from_row(row, split="train")

    assert example.dataset == "hotpotqa"
    assert example.supporting_facts[0].text == "Alice was born in Paris."
    assert example.metadata["level"] == "easy"


def test_2wiki_row_maps_evidences_to_chain():
    row = {
        "id": "w1",
        "question": "Who is related to Bob?",
        "answer": "Alice",
        "type": "inference",
        "evidences": [["Bob", "sibling", "Alice"]],
        "supporting_facts": {"title": ["Bob"], "sent_id": [0]},
        "context": {"title": ["Bob"], "sentences": [["Bob is Alice's sibling."]]},
    }

    example = build_2wiki_example_from_row(row, split="train")

    assert example.dataset == "2wiki"
    assert example.evidence_chain[0].sub_question == "Bob sibling ?"
    assert example.evidence_chain[0].answer == "Alice"
```

- [ ] **步骤 6：运行新增测试验证失败**

运行：`pytest tests/test_dataset_builders.py -v`

预期：FAIL，报错包含 `cannot import name 'build_hotpot_example_from_row'`。

- [ ] **步骤 7：实现 HotpotQA 和 2Wiki 行转换函数**

在 `src/macorag/dataset_builders.py` 追加：

```python
def _sequence_struct_to_pairs(value: Any) -> list[tuple[str, int | None]]:
    if isinstance(value, dict):
        titles = value.get("title", [])
        sent_ids = value.get("sent_id", [])
        return [(str(titles[i]), sent_ids[i] if i < len(sent_ids) else None) for i in range(len(titles))]
    return [(str(item[0]), int(item[1])) for item in value]


def _context_sentence_lookup(context: Any) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentences = context.get("sentences", [])
        for title, sent_list in zip(titles, sentences):
            for sent_id, sent_text in enumerate(sent_list):
                lookup[(str(title), sent_id)] = normalize_text(sent_text)
    else:
        for title, sent_list in context:
            for sent_id, sent_text in enumerate(sent_list):
                lookup[(str(title), sent_id)] = normalize_text(sent_text)
    return lookup


def build_hotpot_example_from_row(row: dict[str, Any], *, split: str) -> Example:
    sf_pairs = _sequence_struct_to_pairs(row.get("supporting_facts", {}))
    lookup = _context_sentence_lookup(row.get("context", {}))
    supporting_facts = [
        SupportingFact(
            doc_id=None,
            title=title,
            sent_id=sent_id,
            text=lookup.get((title, sent_id)) if sent_id is not None else None,
            source="gold",
        )
        for title, sent_id in sf_pairs
    ]
    usable = bool(row.get("answer") and supporting_facts)
    return Example(
        qid=str(row["id"]),
        dataset="hotpotqa",
        split=split,
        question=normalize_text(row["question"]),
        answer=normalize_text(row.get("answer")) or None,
        answer_aliases=[],
        question_type=row.get("type"),
        hop_count=2,
        supporting_facts=supporting_facts,
        evidence_chain=[],
        context_doc_ids=[],
        usable_for_sft=usable,
        usable_for_retrieval_eval=usable,
        quality_flags=[] if usable else ["missing_answer_or_support"],
        metadata={"level": row.get("level")},
    )


def build_2wiki_example_from_row(row: dict[str, Any], *, split: str) -> Example:
    sf_pairs = _sequence_struct_to_pairs(row.get("supporting_facts", {}))
    lookup = _context_sentence_lookup(row.get("context", {}))
    supporting_facts = [
        SupportingFact(
            doc_id=None,
            title=title,
            sent_id=sent_id,
            text=lookup.get((title, sent_id)) if sent_id is not None else None,
            source="gold",
        )
        for title, sent_id in sf_pairs
    ]
    evidence_chain = []
    for index, evidence in enumerate(row.get("evidences", []), start=1):
        subject = normalize_text(evidence[0]) if len(evidence) > 0 else ""
        relation = normalize_text(evidence[1]) if len(evidence) > 1 else ""
        obj = normalize_text(evidence[2]) if len(evidence) > 2 else ""
        evidence_chain.append(
            EvidenceStep(
                step=index,
                sub_question=f"{subject} {relation} ?".strip(),
                answer=obj or None,
                support_doc_id=None,
                support_title=subject or None,
                support_sent_ids=[],
            )
        )
    usable = bool(row.get("answer") and (supporting_facts or evidence_chain))
    return Example(
        qid=str(row["id"]),
        dataset="2wiki",
        split=split,
        question=normalize_text(row["question"]),
        answer=normalize_text(row.get("answer")) or None,
        answer_aliases=[],
        question_type=row.get("type"),
        hop_count=max(2, len(evidence_chain)),
        supporting_facts=supporting_facts,
        evidence_chain=evidence_chain,
        context_doc_ids=[],
        usable_for_sft=usable,
        usable_for_retrieval_eval=usable,
        quality_flags=[] if usable else ["missing_answer_or_support"],
        metadata={"evidences": row.get("evidences", [])},
    )
```

- [ ] **步骤 8：运行全部 builder 测试**

运行：`pytest tests/test_dataset_builders.py -v`

预期：`3 passed`。

- [ ] **步骤 9：Commit**

```bash
git add src/macorag/dataset_builders.py tests/test_dataset_builders.py
git commit -m "feat: add canonical dataset builders"
```

## 任务 4：实现 LinearRAG 适配输出

**文件：**
- 创建：`src/macorag/linearrag_adapter.py`
- 创建：`tests/test_linearrag_adapter.py`

- [ ] **步骤 1：编写失败的适配器测试**

创建 `tests/test_linearrag_adapter.py`：

```python
from pathlib import Path

from macorag.linearrag_adapter import build_linearrag_dataset
from macorag.schemas import CorpusDoc, Example, SupportingFact


def test_linearrag_adapter_writes_questions_chunks_meta_and_qrels(tmp_path):
    example = Example(
        qid="q1",
        dataset="hotpotqa",
        split="train",
        question="Where was Alice born?",
        answer="Paris",
        answer_aliases=[],
        question_type="bridge",
        hop_count=2,
        supporting_facts=[
            SupportingFact(doc_id="d1", title="Alice", sent_id=0, text="Alice was born in Paris.", source="gold")
        ],
        evidence_chain=[],
        context_doc_ids=["d1"],
        usable_for_sft=True,
        usable_for_retrieval_eval=True,
        quality_flags=[],
        metadata={},
    )
    corpus = [
        CorpusDoc(
            doc_id="d1",
            dataset="hotpotqa",
            title="Alice",
            text="Alice was born in Paris.",
            sentences=["Alice was born in Paris."],
            source="beir",
            metadata={},
        )
    ]

    build_linearrag_dataset("hotpotqa", [example], corpus, tmp_path)

    assert (tmp_path / "hotpotqa" / "questions.json").exists()
    assert (tmp_path / "hotpotqa" / "chunks.json").exists()
    assert (tmp_path / "hotpotqa" / "chunk_meta.jsonl").exists()
    assert (tmp_path / "hotpotqa" / "qrels.jsonl").exists()
    assert '"question": "Where was Alice born?"' in (tmp_path / "hotpotqa" / "questions.json").read_text()
    assert "Alice\\nAlice was born in Paris." in (tmp_path / "hotpotqa" / "chunks.json").read_text()
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_linearrag_adapter.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.linearrag_adapter'`。

- [ ] **步骤 3：实现 LinearRAG 适配器**

创建 `src/macorag/linearrag_adapter.py`：

```python
from __future__ import annotations

from pathlib import Path

from macorag.io_utils import write_json, write_jsonl
from macorag.schemas import CorpusDoc, Example


def _chunk_id(dataset: str, index: int) -> str:
    return f"{dataset}:chunk:{index}"


def build_linearrag_dataset(
    dataset: str,
    examples: list[Example],
    corpus: list[CorpusDoc],
    output_root: str | Path,
) -> None:
    output_dir = Path(output_root) / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    questions = [
        {
            "id": item.qid,
            "question": item.question,
            "answer": item.answer,
            "dataset": item.dataset,
            "split": item.split,
        }
        for item in examples
    ]
    chunks = [doc.to_chunk_text() for doc in corpus]

    doc_to_chunk_id = {doc.doc_id: _chunk_id(dataset, index) for index, doc in enumerate(corpus)}
    meta_rows = [
        {
            "chunk_id": _chunk_id(dataset, index),
            "chunk_index": index,
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source": doc.source,
            "dataset": doc.dataset,
        }
        for index, doc in enumerate(corpus)
    ]
    qrels = []
    for example in examples:
        gold_doc_ids = sorted({fact.doc_id for fact in example.supporting_facts if fact.doc_id})
        gold_chunk_ids = [doc_to_chunk_id[doc_id] for doc_id in gold_doc_ids if doc_id in doc_to_chunk_id]
        qrels.append(
            {
                "qid": example.qid,
                "gold_doc_ids": gold_doc_ids,
                "gold_chunk_ids": gold_chunk_ids,
                "gold_titles": sorted({fact.title for fact in example.supporting_facts}),
                "gold_sentences": [
                    {
                        "doc_id": fact.doc_id,
                        "sent_id": fact.sent_id,
                        "text": fact.text,
                    }
                    for fact in example.supporting_facts
                ],
            }
        )

    write_json(output_dir / "questions.json", questions)
    write_json(output_dir / "chunks.json", chunks)
    write_jsonl(output_dir / "chunk_meta.jsonl", meta_rows)
    write_jsonl(output_dir / "qrels.jsonl", qrels)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_linearrag_adapter.py -v`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/macorag/linearrag_adapter.py tests/test_linearrag_adapter.py
git commit -m "feat: add linearrag dataset adapter"
```

## 任务 5：实现 Step-wise 检索环境

**文件：**
- 创建：`src/macorag/retrieval_env.py`
- 创建：`tests/test_retrieval_env.py`

- [ ] **步骤 1：编写失败的检索环境测试**

创建 `tests/test_retrieval_env.py`：

```python
from macorag.retrieval_env import InMemoryRetrievalEnv


def test_retrieval_env_tracks_state_and_budget():
    env = InMemoryRetrievalEnv(
        questions={"q1": {"question": "Where was Alice born?", "dataset": "hotpotqa"}},
        chunks=[
            {"chunk_id": "c1", "title": "Alice", "text": "Alice was born in Paris."},
            {"chunk_id": "c2", "title": "Bob", "text": "Bob was born in Rome."},
        ],
        retrieval_budget=2,
    )

    state = env.reset("q1")
    assert state["retrieval_count"] == 0

    observation = env.step("Alice born", top_k=1)

    assert observation["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert env.get_state()["retrieval_count"] == 1
    assert env.get_state()["retrieval_history"][0]["query"] == "Alice born"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_retrieval_env.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.retrieval_env'`。

- [ ] **步骤 3：实现 InMemoryRetrievalEnv**

创建 `src/macorag/retrieval_env.py`：

```python
from __future__ import annotations

from collections import Counter
import math
import re
from typing import Any


TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> Counter[str]:
    return Counter(token.lower() for token in TOKEN_RE.findall(text))


def _cosine(query: Counter[str], doc: Counter[str]) -> float:
    numerator = sum(query[token] * doc.get(token, 0) for token in query)
    if numerator == 0:
        return 0.0
    q_norm = math.sqrt(sum(value * value for value in query.values()))
    d_norm = math.sqrt(sum(value * value for value in doc.values()))
    return numerator / (q_norm * d_norm) if q_norm and d_norm else 0.0


class InMemoryRetrievalEnv:
    def __init__(
        self,
        *,
        questions: dict[str, dict[str, Any]],
        chunks: list[dict[str, Any]],
        retrieval_budget: int,
    ) -> None:
        self.questions = questions
        self.chunks = chunks
        self.retrieval_budget = retrieval_budget
        self._chunk_tokens = [_tokens(f"{chunk.get('title', '')} {chunk.get('text', '')}") for chunk in chunks]
        self._state: dict[str, Any] | None = None

    def reset(self, qid: str) -> dict[str, Any]:
        question = self.questions[qid]
        self._state = {
            "qid": qid,
            "dataset": question.get("dataset"),
            "question": question["question"],
            "evidence": [],
            "retrieval_history": [],
            "retrieval_count": 0,
            "retrieval_budget": self.retrieval_budget,
        }
        return self.get_state()

    def get_state(self) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("reset(qid) must be called before get_state()")
        return dict(self._state)

    def step(self, query: str, top_k: int) -> dict[str, Any]:
        if self._state is None:
            raise RuntimeError("reset(qid) must be called before step()")
        if self._state["retrieval_count"] >= self.retrieval_budget:
            raise RuntimeError("retrieval budget exceeded")

        query_tokens = _tokens(query)
        scored = []
        for index, chunk in enumerate(self.chunks):
            score = _cosine(query_tokens, self._chunk_tokens[index])
            scored.append((score, index, chunk))
        scored.sort(key=lambda item: (-item[0], item[1]))
        retrieved = [
            {
                "chunk_id": chunk["chunk_id"],
                "rank": rank,
                "score": score,
                "title": chunk.get("title", ""),
                "text": chunk.get("text", ""),
            }
            for rank, (score, _index, chunk) in enumerate(scored[:top_k], start=1)
        ]
        observation = {"query": query, "top_k": top_k, "retrieved_chunks": retrieved}
        self._state["retrieval_count"] += 1
        self._state["retrieval_history"].append(observation)
        return observation
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_retrieval_env.py -v`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/macorag/retrieval_env.py tests/test_retrieval_env.py
git commit -m "feat: add stepwise retrieval environment"
```

## 任务 6：实现教师标签协议解析器

**文件：**
- 创建：`src/macorag/teacher_protocol.py`
- 创建：`tests/test_teacher_protocol.py`

- [ ] **步骤 1：编写失败的标签解析测试**

创建 `tests/test_teacher_protocol.py`：

```python
import pytest

from macorag.teacher_protocol import ProtocolError, parse_teacher_message


def test_parse_teacher_message_extracts_json_actions():
    message = """
<plan>{"sub_query": "Alice birthplace", "rationale": "Need birthplace."}</plan>
<retrieval>{"query": "Alice birthplace", "top_k": 5}</retrieval>
<update-evidence>{"accepted_chunk_ids": ["c1"], "rejected_chunk_ids": ["c2"], "reason": "c1 states birthplace."}</update-evidence>
<answer>{"answer": "Paris", "supporting_chunk_ids": ["c1"]}</answer>
"""

    parsed = parse_teacher_message(message)

    assert parsed["plan"]["sub_query"] == "Alice birthplace"
    assert parsed["retrieval"]["top_k"] == 5
    assert parsed["update-evidence"]["accepted_chunk_ids"] == ["c1"]
    assert parsed["answer"]["answer"] == "Paris"


def test_parse_teacher_message_requires_closed_update_evidence_tag():
    with pytest.raises(ProtocolError):
        parse_teacher_message("<update-evidence>{}</answer>")
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_teacher_protocol.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.teacher_protocol'`。

- [ ] **步骤 3：实现协议解析器**

创建 `src/macorag/teacher_protocol.py`：

```python
from __future__ import annotations

from typing import Any
import json
import re


class ProtocolError(ValueError):
    pass


TAG_NAMES = ("plan", "retrieval", "update-evidence", "answer")


def _extract_tag(text: str, tag: str) -> dict[str, Any]:
    pattern = re.compile(rf"<{re.escape(tag)}>(.*?)</{re.escape(tag)}>", re.DOTALL)
    match = pattern.search(text)
    if match is None:
        raise ProtocolError(f"missing closed tag: {tag}")
    payload = match.group(1).strip()
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid json in tag {tag}: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"tag {tag} must contain a json object")
    return value


def parse_teacher_message(text: str) -> dict[str, dict[str, Any]]:
    return {tag: _extract_tag(text, tag) for tag in TAG_NAMES}
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_teacher_protocol.py -v`

预期：`2 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/macorag/teacher_protocol.py tests/test_teacher_protocol.py
git commit -m "feat: add teacher action protocol parser"
```

## 任务 7：实现轨迹过滤器

**文件：**
- 创建：`src/macorag/trajectory_filter.py`
- 创建：`tests/test_trajectory_filter.py`

- [ ] **步骤 1：编写失败的轨迹过滤测试**

创建 `tests/test_trajectory_filter.py`：

```python
from macorag.trajectory_filter import evaluate_trajectory


def test_evaluate_trajectory_accepts_grounded_answer():
    trajectory = {
        "qid": "q1",
        "dataset": "hotpotqa",
        "trajectory": [
            {
                "agent": "evidence_updater",
                "action": {
                    "type": "update_evidence",
                    "accepted_chunk_ids": ["c1"],
                    "rejected_chunk_ids": [],
                    "reason": "c1 is relevant",
                },
            },
            {
                "agent": "answer_generator",
                "action": {
                    "type": "final_answer",
                    "answer": "Paris",
                    "supporting_chunk_ids": ["c1"],
                },
            },
        ],
    }
    qrels = {"q1": {"gold_chunk_ids": ["c1"], "gold_titles": ["Alice"]}}
    answers = {"q1": {"answer": "Paris", "answer_aliases": []}}

    result = evaluate_trajectory(trajectory, qrels, answers)

    assert result.accepted is True
    assert result.reasons == []


def test_evaluate_trajectory_rejects_ungrounded_answer():
    trajectory = {
        "qid": "q1",
        "dataset": "hotpotqa",
        "trajectory": [
            {
                "agent": "evidence_updater",
                "action": {
                    "type": "update_evidence",
                    "accepted_chunk_ids": ["c2"],
                    "rejected_chunk_ids": [],
                    "reason": "c2 is relevant",
                },
            },
            {
                "agent": "answer_generator",
                "action": {
                    "type": "final_answer",
                    "answer": "Paris",
                    "supporting_chunk_ids": ["c2"],
                },
            },
        ],
    }

    result = evaluate_trajectory(
        trajectory,
        {"q1": {"gold_chunk_ids": ["c1"], "gold_titles": ["Alice"]}},
        {"q1": {"answer": "Paris", "answer_aliases": []}},
    )

    assert result.accepted is False
    assert "no_gold_evidence_overlap" in result.reasons
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_trajectory_filter.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.trajectory_filter'`。

- [ ] **步骤 3：实现轨迹过滤器**

创建 `src/macorag/trajectory_filter.py`：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from macorag.io_utils import normalize_key


@dataclass
class FilterResult:
    accepted: bool
    reasons: list[str]


def _contains_answer(prediction: str | None, gold: str | None, aliases: list[str]) -> bool:
    if not prediction or not gold:
        return False
    pred_norm = normalize_key(prediction)
    candidates = [gold, *aliases]
    return any(normalize_key(candidate) in pred_norm for candidate in candidates if candidate)


def _collect_accepted_chunks(steps: list[dict[str, Any]]) -> set[str]:
    accepted: set[str] = set()
    for step in steps:
        action = step.get("action", {})
        if action.get("type") == "update_evidence":
            accepted.update(action.get("accepted_chunk_ids", []))
    return accepted


def _find_final_answer(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in reversed(steps):
        action = step.get("action", {})
        if action.get("type") == "final_answer":
            return action
    return None


def evaluate_trajectory(
    trajectory: dict[str, Any],
    qrels_by_qid: dict[str, dict[str, Any]],
    answers_by_qid: dict[str, dict[str, Any]],
) -> FilterResult:
    reasons: list[str] = []
    qid = trajectory["qid"]
    steps = trajectory.get("trajectory", [])
    accepted_chunks = _collect_accepted_chunks(steps)
    final_action = _find_final_answer(steps)
    qrels = qrels_by_qid.get(qid, {})
    gold_chunks = set(qrels.get("gold_chunk_ids", []))
    answer_info = answers_by_qid.get(qid, {})

    if not steps:
        reasons.append("empty_trajectory")
    if not accepted_chunks:
        reasons.append("no_accepted_evidence")
    if gold_chunks and not (accepted_chunks & gold_chunks):
        reasons.append("no_gold_evidence_overlap")
    if final_action is None:
        reasons.append("missing_final_answer")
    else:
        supporting_chunks = set(final_action.get("supporting_chunk_ids", []))
        if not supporting_chunks:
            reasons.append("missing_supporting_chunks")
        if not supporting_chunks.issubset(accepted_chunks):
            reasons.append("supporting_chunks_not_accepted")
        if not _contains_answer(
            final_action.get("answer"),
            answer_info.get("answer"),
            list(answer_info.get("answer_aliases", [])),
        ):
            reasons.append("answer_mismatch")

    return FilterResult(accepted=not reasons, reasons=reasons)
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_trajectory_filter.py -v`

预期：`2 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/macorag/trajectory_filter.py tests/test_trajectory_filter.py
git commit -m "feat: add trajectory quality filter"
```

## 任务 8：实现教师 API 抽象和离线假客户端

**文件：**
- 创建：`src/macorag/teacher_api.py`
- 创建：`tests/test_teacher_api.py`

- [ ] **步骤 1：编写失败的 fake client 测试**

创建 `tests/test_teacher_api.py`：

```python
from macorag.teacher_api import FakeTeacherClient


def test_fake_teacher_client_returns_configured_message():
    client = FakeTeacherClient([
        "<plan>{\"sub_query\":\"Alice\",\"rationale\":\"Find Alice.\"}</plan>"
        "<retrieval>{\"query\":\"Alice\",\"top_k\":5}</retrieval>"
        "<update-evidence>{\"accepted_chunk_ids\":[\"c1\"],\"rejected_chunk_ids\":[],\"reason\":\"Relevant.\"}</update-evidence>"
        "<answer>{\"answer\":\"Paris\",\"supporting_chunk_ids\":[\"c1\"]}</answer>"
    ])

    assert "<plan>" in client.generate("prompt")
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_teacher_api.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.teacher_api'`。

- [ ] **步骤 3：实现教师客户端抽象**

创建 `src/macorag/teacher_api.py`：

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol
import json
import os
import urllib.request


class TeacherClient(Protocol):
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class FakeTeacherClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if not self.responses:
            raise RuntimeError("FakeTeacherClient has no remaining responses")
        return self.responses.pop(0)


@dataclass
class OpenAICompatibleClient:
    api_key_env: str
    base_url: str
    model: str
    temperature: float = 0.2
    max_tokens: int = 2048

    def generate(self, prompt: str) -> str:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"missing api key env var: {self.api_key_env}")
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload["choices"][0]["message"]["content"]
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_teacher_api.py -v`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/macorag/teacher_api.py tests/test_teacher_api.py
git commit -m "feat: add teacher api client abstraction"
```

## 任务 9：实现采样策略

**文件：**
- 创建：`src/macorag/sampling.py`
- 创建：`tests/test_sampling.py`

- [ ] **步骤 1：编写失败的分层采样测试**

创建 `tests/test_sampling.py`：

```python
from macorag.sampling import sample_examples


def test_sample_examples_balances_question_type():
    examples = []
    for index in range(10):
        examples.append({"qid": f"b{index}", "question_type": "bridge", "hop_count": 2})
        examples.append({"qid": f"c{index}", "question_type": "comparison", "hop_count": 2})

    sampled = sample_examples(examples, target_count=6, seed=7)

    types = [item["question_type"] for item in sampled]
    assert len(sampled) == 6
    assert "bridge" in types
    assert "comparison" in types
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_sampling.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.sampling'`。

- [ ] **步骤 3：实现采样函数**

创建 `src/macorag/sampling.py`：

```python
from __future__ import annotations

from collections import defaultdict
from typing import Any
import random


def _bucket_key(example: dict[str, Any]) -> str:
    return str(example.get("question_type") or f"{example.get('hop_count', 'unknown')}hop")


def sample_examples(
    examples: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example in examples:
        buckets[_bucket_key(example)].append(example)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, Any]] = []
    bucket_names = sorted(buckets)
    while len(selected) < target_count and bucket_names:
        progressed = False
        for name in list(bucket_names):
            bucket = buckets[name]
            if bucket:
                selected.append(bucket.pop())
                progressed = True
                if len(selected) == target_count:
                    break
            else:
                bucket_names.remove(name)
        if not progressed:
            break
    return selected
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_sampling.py -v`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/macorag/sampling.py tests/test_sampling.py
git commit -m "feat: add stratified trajectory sampling"
```

## 任务 10：实现轨迹构造器

**文件：**
- 创建：`src/macorag/trajectory_builder.py`
- 创建：`tests/test_trajectory_builder.py`

- [ ] **步骤 1：编写失败的轨迹构造测试**

创建 `tests/test_trajectory_builder.py`：

```python
from macorag.retrieval_env import InMemoryRetrievalEnv
from macorag.teacher_api import FakeTeacherClient
from macorag.trajectory_builder import build_one_trajectory


def test_build_one_trajectory_uses_teacher_protocol_and_env():
    env = InMemoryRetrievalEnv(
        questions={"q1": {"question": "Where was Alice born?", "dataset": "hotpotqa"}},
        chunks=[{"chunk_id": "c1", "title": "Alice", "text": "Alice was born in Paris."}],
        retrieval_budget=3,
    )
    teacher = FakeTeacherClient([
        """
<plan>{"sub_query": "Alice birthplace", "rationale": "Find birthplace."}</plan>
<retrieval>{"query": "Alice birthplace", "top_k": 1}</retrieval>
<update-evidence>{"accepted_chunk_ids": ["c1"], "rejected_chunk_ids": [], "reason": "The chunk states it."}</update-evidence>
<answer>{"answer": "Paris", "supporting_chunk_ids": ["c1"]}</answer>
"""
    ])

    trajectory = build_one_trajectory("q1", env, teacher)

    assert trajectory["qid"] == "q1"
    assert trajectory["trajectory"][0]["agent"] == "planner"
    assert trajectory["trajectory"][1]["observation"]["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert trajectory["trajectory"][-1]["action"]["answer"] == "Paris"
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_trajectory_builder.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.trajectory_builder'`。

- [ ] **步骤 3：实现单样本轨迹构造**

创建 `src/macorag/trajectory_builder.py`：

```python
from __future__ import annotations

from typing import Any

from macorag.retrieval_env import InMemoryRetrievalEnv
from macorag.teacher_api import TeacherClient
from macorag.teacher_protocol import parse_teacher_message


def _prompt_from_state(state: dict[str, Any]) -> str:
    return (
        "You are a multi-agent RAG teacher. "
        "Use only the visible state and retrieval observations. "
        "Do not use gold answers or hidden evidence.\n"
        f"Question: {state['question']}\n"
        f"Retrieval budget: {state['retrieval_budget']}\n"
        "Return <plan>, <retrieval>, <update-evidence>, and <answer> tags with JSON payloads."
    )


def build_one_trajectory(
    qid: str,
    env: InMemoryRetrievalEnv,
    teacher: TeacherClient,
) -> dict[str, Any]:
    state = env.reset(qid)
    raw_text = teacher.generate(_prompt_from_state(state))
    parsed = parse_teacher_message(raw_text)

    retrieval_action = parsed["retrieval"]
    observation = env.step(
        str(retrieval_action["query"]),
        top_k=int(retrieval_action.get("top_k", 5)),
    )
    update_action = parsed["update-evidence"]
    accepted = update_action.get("accepted_chunk_ids", [])
    state_after_update = env.get_state()
    state_after_update["evidence"] = accepted

    return {
        "qid": qid,
        "dataset": state.get("dataset"),
        "trajectory": [
            {
                "t": 0,
                "state": state,
                "agent": "planner",
                "raw_text": f"<plan>{parsed['plan']}</plan>",
                "action": {
                    "type": "plan_query",
                    "sub_query": parsed["plan"].get("sub_query"),
                    "rationale": parsed["plan"].get("rationale"),
                },
            },
            {
                "t": 1,
                "agent": "retriever",
                "raw_text": f"<retrieval>{parsed['retrieval']}</retrieval>",
                "action": {
                    "type": "retrieve",
                    "query": retrieval_action.get("query"),
                    "top_k": retrieval_action.get("top_k", 5),
                },
                "observation": observation,
            },
            {
                "t": 2,
                "agent": "evidence_updater",
                "raw_text": f"<update-evidence>{parsed['update-evidence']}</update-evidence>",
                "action": {
                    "type": "update_evidence",
                    "accepted_chunk_ids": update_action.get("accepted_chunk_ids", []),
                    "rejected_chunk_ids": update_action.get("rejected_chunk_ids", []),
                    "reason": update_action.get("reason"),
                },
                "state_delta": {"added_evidence": accepted},
            },
            {
                "t": 3,
                "agent": "answer_generator",
                "raw_text": f"<answer>{parsed['answer']}</answer>",
                "action": {
                    "type": "final_answer",
                    "answer": parsed["answer"].get("answer"),
                    "supporting_chunk_ids": parsed["answer"].get("supporting_chunk_ids", []),
                },
            },
        ],
    }
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_trajectory_builder.py -v`

预期：`1 passed`。

- [ ] **步骤 5：Commit**

```bash
git add src/macorag/trajectory_builder.py tests/test_trajectory_builder.py
git commit -m "feat: add teacher trajectory builder"
```

## 任务 11：实现 CLI 串联命令

**文件：**
- 创建：`src/macorag/cli.py`
- 创建：`tests/test_cli.py`

- [ ] **步骤 1：编写失败的 CLI smoke test**

创建 `tests/test_cli.py`：

```python
from macorag.cli import build_parser


def test_cli_parser_has_required_commands():
    parser = build_parser()

    commands = parser._subparsers._group_actions[0].choices

    assert "build-canonical" in commands
    assert "build-linearrag" in commands
    assert "sample" in commands
    assert "generate-trajectories" in commands
    assert "filter-trajectories" in commands
```

- [ ] **步骤 2：运行测试验证失败**

运行：`pytest tests/test_cli.py -v`

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'macorag.cli'`。

- [ ] **步骤 3：实现 CLI parser**

创建 `src/macorag/cli.py`：

```python
from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="macorag")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_canonical = subparsers.add_parser("build-canonical")
    build_canonical.add_argument("--data-root", default="data")
    build_canonical.add_argument("--output-root", default="data/processed")

    build_linearrag = subparsers.add_parser("build-linearrag")
    build_linearrag.add_argument("--processed-root", default="data/processed")
    build_linearrag.add_argument("--output-root", default="linearrag_dataset")

    sample = subparsers.add_parser("sample")
    sample.add_argument("--processed-root", default="data/processed")
    sample.add_argument("--output-root", default="trajectories")
    sample.add_argument("--per-dataset", type=int, default=1000)
    sample.add_argument("--seed", type=int, default=7)

    generate = subparsers.add_parser("generate-trajectories")
    generate.add_argument("--linearrag-root", default="linearrag_dataset")
    generate.add_argument("--sample-root", default="trajectories")
    generate.add_argument("--raw-output-name", default="raw_teacher_trajectories.jsonl")

    filter_cmd = subparsers.add_parser("filter-trajectories")
    filter_cmd.add_argument("--trajectory-root", default="trajectories")
    filter_cmd.add_argument("--linearrag-root", default="linearrag_dataset")
    filter_cmd.add_argument("--filtered-output-name", default="filtered_sft_trajectories.jsonl")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
```

- [ ] **步骤 4：运行测试验证通过**

运行：`pytest tests/test_cli.py -v`

预期：`1 passed`。

- [ ] **步骤 5：在 pyproject 中加入 console script**

修改 `pyproject.toml`，在 `[project]` 后追加：

```toml
[project.scripts]
macorag = "macorag.cli:main"
```

- [ ] **步骤 6：运行 CLI 测试**

运行：`pytest tests/test_cli.py -v`

预期：`1 passed`。

- [ ] **步骤 7：Commit**

```bash
git add src/macorag/cli.py tests/test_cli.py pyproject.toml
git commit -m "feat: add macorag cli skeleton"
```

## 任务 12：端到端测试、文档和忽略规则

**文件：**
- 创建：`tests/test_end_to_end_small.py`
- 修改：`.gitignore`
- 创建：`README.md`

- [ ] **步骤 1：编写小型端到端测试**

创建 `tests/test_end_to_end_small.py`：

```python
from macorag.dataset_builders import build_musique_canonical_from_rows
from macorag.linearrag_adapter import build_linearrag_dataset
from macorag.io_utils import read_json, read_jsonl


def test_small_musique_to_linearrag_end_to_end(tmp_path):
    rows = [
        {
            "id": "2hop__1_2",
            "question": "Who is the spouse of the Green performer?",
            "answer": "Miquette Giraudy",
            "answer_aliases": [],
            "answerable": True,
            "paragraphs": [
                {"idx": 5, "title": "Miquette Giraudy", "paragraph_text": "Miquette Giraudy is the spouse."},
                {"idx": 10, "title": "Steve Hillage", "paragraph_text": "Steve Hillage performed with Green."},
            ],
            "question_decomposition": [
                {"id": 1, "question": "Green >> performer", "answer": "Steve Hillage", "paragraph_support_idx": 10},
                {"id": 2, "question": "#1 >> spouse", "answer": "Miquette Giraudy", "paragraph_support_idx": 5},
            ],
        }
    ]
    examples, corpus, report = build_musique_canonical_from_rows(rows, split="train")
    build_linearrag_dataset("musique", examples, corpus, tmp_path)

    questions = read_json(tmp_path / "musique" / "questions.json")
    qrels = list(read_jsonl(tmp_path / "musique" / "qrels.jsonl"))

    assert report["errors"] == []
    assert questions[0]["id"] == "2hop__1_2"
    assert len(qrels[0]["gold_doc_ids"]) == 2
```

- [ ] **步骤 2：运行端到端测试**

运行：`pytest tests/test_end_to_end_small.py -v`

预期：`1 passed`。

- [ ] **步骤 3：更新 `.gitignore`**

确保 `.gitignore` 包含：

```gitignore
.codex/
.agents/
data/
linearrag_dataset/
trajectories/
.pytest_cache/
__pycache__/
*.pyc
```

- [ ] **步骤 4：创建 README**

创建 `README.md`：

```markdown
# MACORAG

Multi-agent RAG data processing and teacher trajectory construction for HotpotQA, 2Wiki, and MuSiQue.

## First Milestone

- Build canonical `data/processed/<dataset>/examples.*.jsonl` and `corpus.jsonl`.
- Build LinearRAG-compatible `linearrag_dataset/<dataset>/questions.json` and `chunks.json`.
- Generate full-API teacher trajectories with fixed tags:
  `<plan>`, `<retrieval>`, `<update-evidence>`, `<answer>`.
- Filter to 3K SFT trajectories for Qwen2.5-7B warm-up:
  1K HotpotQA, 1K 2Wiki, 1K MuSiQue.

## Safety Rule

Gold answers and gold evidence are used only by verifiers and filters. They must not appear in teacher prompts.
```

- [ ] **步骤 5：运行完整测试**

运行：`pytest -v`

预期：所有测试通过。

- [ ] **步骤 6：Commit**

```bash
git add .gitignore README.md tests/test_end_to_end_small.py
git commit -m "test: add small end-to-end data pipeline check"
```

## 自检清单

- 规格覆盖：
  - canonical schema：任务 1-3
  - LinearRAG adapter：任务 4
  - step-wise retrieval environment：任务 5
  - fixed teacher tags：任务 6
  - trajectory filtering with gold evidence：任务 7
  - API-based teacher generation：任务 8 和任务 10
  - 3K SFT v1 sampling policy：任务 9
  - CLI workflow：任务 11
  - end-to-end validation：任务 12
- 防泄漏要求：
  - `trajectory_builder._prompt_from_state()` 不包含 gold answer 或 gold evidence。
  - `trajectory_filter.evaluate_trajectory()` 单独接收 qrels 和 answers。
- 第一版不做：
  - Qwen2.5-7B 训练。
  - RL。
  - LinearRAG 内部重写。
  - 跨数据集去重。
  - 长文档 window 切分。
