from __future__ import annotations

from typing import Any

from macorag.io_utils import normalize_key, normalize_text, sha1_text
from macorag.schemas import CorpusDoc, EvidenceStep, Example, SupportingFact


def _empty_report() -> dict[str, list[Any]]:
    return {"examples": [], "corpus": [], "errors": []}


def _to_py(value: Any) -> Any:
    if hasattr(value, "as_py"):
        return _to_py(value.as_py())
    if isinstance(value, dict):
        return {key: _to_py(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_py(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_py(item) for item in value)
    return value


def _normalized_answer(value: Any) -> str | None:
    normalized = normalize_text(str(_to_py(value) or ""))
    return normalized or None


def _musique_doc_id(title: str, text: str) -> str:
    key = f"{normalize_key(title)} {normalize_text(text)}"
    return f"musique:{sha1_text(key)}"


def _paragraph_title(paragraph: dict[str, Any]) -> str:
    paragraph = _to_py(paragraph)
    return str(paragraph.get("title") or "")


def _paragraph_text(paragraph: dict[str, Any]) -> str:
    paragraph = _to_py(paragraph)
    return normalize_text(
        str(paragraph.get("paragraph_text") or paragraph.get("text") or "")
    )


def _doc_id(dataset: str, title: str, text: str) -> str:
    key = f"{normalize_key(title)} {normalize_text(text)}"
    return f"{dataset}:{sha1_text(key)}"


def _sequence_struct_to_pairs(value: Any) -> list[tuple[str, int]]:
    value = _to_py(value)
    if isinstance(value, dict):
        titles = value.get("title", [])
        sent_ids = value.get("sent_id", [])
        return [
            (str(title), int(sent_id))
            for title, sent_id in zip(titles, sent_ids, strict=False)
        ]

    pairs: list[tuple[str, int]] = []
    for item in value or []:
        item = _to_py(item)
        if len(item) >= 2:
            pairs.append((str(_to_py(item[0])), int(_to_py(item[1]))))
    return pairs


def _context_sentence_lookup(context: Any) -> dict[tuple[str, int], str]:
    context = _to_py(context)
    lookup: dict[tuple[str, int], str] = {}
    if isinstance(context, dict):
        titles = context.get("title", [])
        sentence_groups = context.get("sentences", [])
        for title, sentences in zip(titles, sentence_groups, strict=False):
            for sent_id, sentence in enumerate(sentences):
                lookup[(str(title), sent_id)] = str(sentence)
        return lookup

    for item in context or []:
        if len(item) < 2:
            continue
        title, sentences = item[0], item[1]
        for sent_id, sentence in enumerate(sentences):
            lookup[(str(title), sent_id)] = str(sentence)
    return lookup


def _build_supporting_facts(
    *,
    dataset: str,
    supporting_facts: Any,
    context: Any,
) -> list[SupportingFact]:
    supporting_facts = _to_py(supporting_facts)
    context = _to_py(context)
    sentence_lookup = _context_sentence_lookup(context)
    facts: list[SupportingFact] = []
    for title, sent_id in _sequence_struct_to_pairs(supporting_facts):
        text = sentence_lookup.get((title, sent_id))
        facts.append(
            SupportingFact(
                doc_id=_doc_id(dataset, title, text or ""),
                title=title,
                sent_id=sent_id,
                text=text,
                source="gold",
            )
        )
    return facts


def _context_doc_ids(dataset: str, context: Any) -> list[str]:
    context = _to_py(context)
    doc_ids: list[str] = []
    seen: set[str] = set()
    if isinstance(context, dict):
        items = zip(
            context.get("title", []),
            context.get("sentences", []),
            strict=False,
        )
    else:
        items = ((item[0], item[1]) for item in context or [] if len(item) >= 2)

    for title, sentences in items:
        text = normalize_text(" ".join(str(sentence) for sentence in sentences))
        doc_id = _doc_id(dataset, str(title), text)
        if doc_id not in seen:
            seen.add(doc_id)
            doc_ids.append(doc_id)
    return doc_ids


def build_musique_canonical_from_rows(
    rows: list[dict[str, Any]],
    *,
    split: str,
) -> dict[str, list[Any]]:
    report = _empty_report()
    corpus_by_doc_id: dict[str, CorpusDoc] = {}

    for row in rows:
        row = _to_py(row)
        answerable = _to_py(row.get("answerable"))
        if answerable is False:
            continue

        qid = str(row.get("id") or row.get("qid"))
        paragraphs = [_to_py(paragraph) for paragraph in row.get("paragraphs", [])]
        paragraph_doc_ids: list[str] = []
        doc_ids_by_idx: dict[int, str] = {}
        paragraphs_by_idx = {
            int(_to_py(paragraph["idx"])): paragraph for paragraph in paragraphs
        }
        for paragraph in paragraphs:
            title = _paragraph_title(paragraph)
            text = _paragraph_text(paragraph)
            doc_id = _musique_doc_id(title, text)
            paragraph_doc_ids.append(doc_id)
            if "idx" in paragraph:
                doc_ids_by_idx[int(_to_py(paragraph["idx"]))] = doc_id
            if doc_id not in corpus_by_doc_id:
                corpus_by_doc_id[doc_id] = CorpusDoc(
                    doc_id=doc_id,
                    dataset="musique",
                    title=title,
                    text=text,
                    sentences=[text] if text else [],
                    source="musique_context",
                    metadata={"linked_qids": []},
                )
            linked_qids = corpus_by_doc_id[doc_id].metadata.setdefault(
                "linked_qids",
                [],
            )
            if qid not in linked_qids:
                linked_qids.append(qid)

        supporting_facts: list[SupportingFact] = []
        evidence_chain: list[EvidenceStep] = []
        for step_index, decomposition in enumerate(
            [_to_py(item) for item in row.get("question_decomposition", [])],
            start=1,
        ):
            support_idx = _to_py(decomposition.get("paragraph_support_idx"))
            support_doc_id = None
            support_title = None
            if support_idx is not None and int(support_idx) in paragraphs_by_idx:
                support_key = int(support_idx)
                paragraph = paragraphs_by_idx[support_key]
                support_doc_id = doc_ids_by_idx[support_key]
                support_title = _paragraph_title(paragraph)
                supporting_facts.append(
                    SupportingFact(
                        doc_id=support_doc_id,
                        title=support_title,
                        sent_id=None,
                        text=_paragraph_text(paragraph),
                        source="gold",
                    )
                )

            evidence_chain.append(
                EvidenceStep(
                    step=step_index,
                    sub_question=decomposition.get("question"),
                    answer=decomposition.get("answer"),
                    support_doc_id=support_doc_id,
                    support_title=support_title,
                    support_sent_ids=[],
                )
            )

        answer = _normalized_answer(row.get("answer"))
        is_usable = bool(answer and supporting_facts)
        report["examples"].append(
            Example(
                qid=qid,
                dataset="musique",
                split=split,
                question=str(row.get("question") or ""),
                answer=answer,
                answer_aliases=list(_to_py(row.get("answer_aliases", [])) or []),
                question_type=row.get("type"),
                hop_count=len(evidence_chain),
                supporting_facts=supporting_facts,
                evidence_chain=evidence_chain,
                context_doc_ids=paragraph_doc_ids,
                usable_for_sft=is_usable,
                usable_for_retrieval_eval=is_usable,
                quality_flags=[],
                metadata={"answerable": True},
            )
        )

    report["corpus"].extend(corpus_by_doc_id.values())
    return report


def build_hotpot_example_from_row(row: dict[str, Any], *, split: str) -> Example:
    row = _to_py(row)
    answer = _normalized_answer(row.get("answer"))
    supporting_facts = _build_supporting_facts(
        dataset="hotpotqa",
        supporting_facts=row.get("supporting_facts", []),
        context=row.get("context", []),
    )
    is_usable = bool(answer and supporting_facts)
    return Example(
        qid=str(row.get("id") or row.get("_id") or row.get("qid")),
        dataset="hotpotqa",
        split=split,
        question=str(row.get("question") or ""),
        answer=answer,
        answer_aliases=list(_to_py(row.get("answer_aliases", [])) or []),
        question_type=row.get("type"),
        hop_count=2,
        supporting_facts=supporting_facts,
        evidence_chain=[],
        context_doc_ids=_context_doc_ids("hotpotqa", row.get("context", [])),
        usable_for_sft=is_usable,
        usable_for_retrieval_eval=is_usable,
        quality_flags=[],
        metadata={"level": row.get("level")},
    )


def build_2wiki_example_from_row(row: dict[str, Any], *, split: str) -> Example:
    row = _to_py(row)
    answer = _normalized_answer(row.get("answer"))
    supporting_facts = _build_supporting_facts(
        dataset="2wiki",
        supporting_facts=row.get("supporting_facts", []),
        context=row.get("context", []),
    )
    evidence_chain = [
        EvidenceStep(
            step=step,
            sub_question=f"{_to_py(evidence[0])} {_to_py(evidence[1])} ?",
            answer=str(_to_py(evidence[2])),
            support_doc_id=None,
            support_title=str(_to_py(evidence[0])),
            support_sent_ids=[],
        )
        for step, evidence in enumerate(_to_py(row.get("evidences", [])), start=1)
        if len(evidence) >= 3
    ]
    is_usable = bool(answer and (supporting_facts or evidence_chain))
    return Example(
        qid=str(row.get("id") or row.get("_id") or row.get("qid")),
        dataset="2wiki",
        split=split,
        question=str(row.get("question") or ""),
        answer=answer,
        answer_aliases=list(_to_py(row.get("answer_aliases", [])) or []),
        question_type=row.get("type"),
        hop_count=2,
        supporting_facts=supporting_facts,
        evidence_chain=evidence_chain,
        context_doc_ids=_context_doc_ids("2wiki", row.get("context", [])),
        usable_for_sft=is_usable,
        usable_for_retrieval_eval=is_usable,
        quality_flags=[],
        metadata={"evidences": _to_py(row.get("evidences", []))},
    )
