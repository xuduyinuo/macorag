from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union


def _loads_json(value: Union[str, dict[str, Any]]) -> dict[str, Any]:
    if isinstance(value, str):
        return json.loads(value)
    return value


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    raise ValueError(f"{field_name} must be a strict boolean value")


@dataclass
class SupportingFact:
    doc_id: Optional[str]
    title: str
    sent_id: Optional[int]
    text: Optional[str]
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
    sub_question: Optional[str]
    answer: Optional[str]
    support_doc_id: Optional[str]
    support_title: Optional[str]
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
    answer: Optional[str]
    answer_aliases: list[str]
    question_type: Optional[str]
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
    def from_json(cls, value: Union[str, dict[str, Any]]) -> "Example":
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
            usable_for_sft=_parse_bool(
                data.get("usable_for_sft", False),
                field_name="usable_for_sft",
            ),
            usable_for_retrieval_eval=_parse_bool(
                data.get("usable_for_retrieval_eval", False),
                field_name="usable_for_retrieval_eval",
            ),
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
    def from_json(cls, value: Union[str, dict[str, Any]]) -> "CorpusDoc":
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
