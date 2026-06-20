# RAG Agent Data Processing Design

Date: 2026-06-20

## Goal

Build a unified data processing layer for HotpotQA, 2Wiki, and MuSiQue so the retrieval environment, teacher trajectory generation, SFT, and later RL optimization all share the same input and output contracts.

The retrieval environment will use the LinearRAG approach from `https://github.com/DEEP-PolyU/LinearRAG`. LinearRAG expects each dataset to provide `questions.json` and `chunks.json` under a dataset-specific directory. This design keeps that compatibility as an adapter layer instead of making LinearRAG's minimal format the canonical data format.

MuSiQue will use the answerable subset only: `musique_ans_v1.0_train.jsonl` and `musique_ans_v1.0_dev.jsonl`.

## Chosen Approach

Use a canonical schema plus a LinearRAG adapter layer.

The canonical layer preserves question, answer, facts, evidence chains, corpus documents, split metadata, and quality flags. The LinearRAG adapter derives `questions.json` and `chunks.json` from canonical data and keeps sidecar metadata files for mapping retrieved chunks back to evidence.

This avoids binding the whole project to one retrieval implementation while still making the first retrieval environment easy to run.

## Output Layout

```text
data/processed/
  hotpotqa/
    corpus.jsonl
    examples.train.jsonl
    examples.dev.jsonl
    examples.test.jsonl
    stats.json
    validation_report.json

  2wiki/
    corpus.jsonl
    examples.train.jsonl
    examples.dev.jsonl
    examples.test.jsonl
    stats.json
    validation_report.json

  musique/
    corpus.jsonl
    examples.train.jsonl
    examples.dev.jsonl
    stats.json
    validation_report.json

linearrag_dataset/
  hotpotqa/
    questions.json
    chunks.json
    chunk_meta.jsonl
    qrels.jsonl

  2wiki/
    questions.json
    chunks.json
    chunk_meta.jsonl
    qrels.jsonl

  musique/
    questions.json
    chunks.json
    chunk_meta.jsonl
    qrels.jsonl
```

## Canonical Example Schema

Each normalized QA sample is written as one JSON object per line:

```json
{
  "qid": "string",
  "dataset": "hotpotqa | 2wiki | musique",
  "split": "train | dev | test",
  "question": "string",
  "answer": "string | null",
  "answer_aliases": [],
  "question_type": "string | null",
  "hop_count": 2,
  "supporting_facts": [
    {
      "doc_id": "string | null",
      "title": "string",
      "sent_id": 0,
      "text": "string | null",
      "source": "gold"
    }
  ],
  "evidence_chain": [
    {
      "step": 1,
      "sub_question": "string | null",
      "answer": "string | null",
      "support_doc_id": "string | null",
      "support_title": "string | null",
      "support_sent_ids": [0]
    }
  ],
  "context_doc_ids": [],
  "usable_for_sft": true,
  "usable_for_retrieval_eval": true,
  "quality_flags": [],
  "metadata": {}
}
```

The schema separates two evidence concepts:

- `supporting_facts`: text evidence that can be cited, used for retrieval evaluation and grounded answer generation.
- `evidence_chain`: ordered reasoning or decomposition steps, used for teacher trajectory construction and agent action supervision.

## Canonical Corpus Schema

Each corpus document or chunk is written as one JSON object per line:

```json
{
  "doc_id": "string",
  "dataset": "hotpotqa | 2wiki | musique",
  "title": "string",
  "text": "string",
  "sentences": ["string"],
  "source": "beir | para_with_hyperlink | musique_context | qa_context",
  "metadata": {}
}
```

## Dataset Mapping

### HotpotQA

Sources:

- QA: `data/hotpotqa/fullwiki/*.parquet`
- Corpus: `data/hotpotqa/beir_corpus/corpus/*.parquet`

Example mapping:

```text
qid              <- id
dataset          <- "hotpotqa"
split            <- train / dev / test
question         <- question
answer           <- answer
answer_aliases   <- []
question_type    <- type
hop_count         <- 2
supporting_facts <- supporting_facts.title + supporting_facts.sent_id
context_doc_ids  <- matched from context.title or BEIR title when possible
metadata.level   <- level
```

Corpus mapping:

```text
doc_id    <- BEIR _id
title     <- title
text      <- text
sentences <- [text] in the first version
source    <- "beir"
```

HotpotQA fullwiki QA includes local `context`, but the retrieval environment should use BEIR corpus as the main corpus. The QA context is used for gold-support text recovery and alignment checks.

### 2Wiki

Sources:

- QA: `data/2wiki/qa/*.parquet`
- Corpus: `data/2wiki/corpus/2wiki_corpus.jsonl`

Example mapping:

```text
qid                 <- id
dataset             <- "2wiki"
split               <- train / dev / test
question            <- question
answer              <- answer
answer_aliases      <- []
question_type       <- type
hop_count            <- inferred from evidences or type, default 2
supporting_facts    <- supporting_facts.title + supporting_facts.sent_id
evidence_chain      <- evidences converted to ordered steps
context_doc_ids     <- matched from context.title to corpus title
metadata.evidences  <- original evidences
```

Corpus mapping:

```text
doc_id            <- id
title             <- title
text              <- sentences joined with spaces
sentences         <- sentences
source            <- "para_with_hyperlink"
metadata.mentions <- mentions
```

2Wiki has both sentence-level supporting facts and relation-level `evidences`; the relation evidence should seed teacher planning trajectories.

### MuSiQue

Sources:

- QA and context: `data/musique/musique_ans_v1.0_train.jsonl`
- QA and context: `data/musique/musique_ans_v1.0_dev.jsonl`

Example mapping:

```text
qid                 <- id
dataset             <- "musique"
split               <- train / dev
question            <- question
answer              <- answer
answer_aliases      <- answer_aliases
question_type       <- inferred from id or question_decomposition
hop_count            <- len(question_decomposition)
supporting_facts    <- paragraph referenced by paragraph_support_idx
evidence_chain      <- question_decomposition in order
context_doc_ids     <- generated paragraph doc ids
metadata.answerable <- true
```

Corpus mapping:

```text
doc_id    <- musique:{dedup_hash}
title     <- paragraphs[*].title
text      <- paragraphs[*].paragraph_text
sentences <- [paragraph_text]
source    <- "musique_context"
metadata.linked_qids <- qids that include the paragraph
```

MuSiQue has no separate global corpus in the downloaded files, so its corpus is reconstructed by deduplicating all provided paragraphs from the answerable train/dev files.

## Cleaning and Deduplication

Text cleaning is conservative:

```text
1. Strip leading and trailing whitespace.
2. Collapse consecutive whitespace.
3. Preserve original casing, punctuation, and entity names.
4. Drop corpus entries only when both title and text are empty.
5. Keep test examples with answer=null, but exclude them from SFT trajectory generation.
```

Sentence handling:

```text
Datasets with existing sentences keep their original sentence lists.
Datasets with only paragraph_text/text use [text] initially.
```

This prevents sentence-id drift in HotpotQA and 2Wiki. MuSiQue evidence is paragraph-level, so paragraph chunks are acceptable.

Corpus deduplication happens within each dataset, not across datasets:

```text
dedup_key = normalize(title) + sha1(normalize(text))
```

`normalize` lowercases text, collapses whitespace, and strips leading/trailing whitespace. When duplicates are merged, keep one `doc_id` and record all linked question ids in `metadata.linked_qids`.

Do not deduplicate across datasets because Wikipedia versions, sentence segmentation, and evidence annotations differ.

## Split Policy

Preserve original splits:

```text
HotpotQA: train / dev / test
2Wiki: train / dev / test
MuSiQue answerable: train / dev
```

Teacher trajectory generation uses only samples from train/dev with non-empty answers and parseable gold evidence.

Test data is retained for evaluation, not SFT labels.

## LinearRAG Adapter

LinearRAG-compatible `questions.json`:

```json
[
  {
    "id": "sample id",
    "question": "string",
    "answer": "string | null",
    "dataset": "hotpotqa",
    "split": "train"
  }
]
```

LinearRAG-compatible `chunks.json`:

```json
[
  "Title\nDocument or paragraph text"
]
```

`chunk_meta.jsonl` maps LinearRAG chunk indices back to canonical documents:

```json
{
  "chunk_id": "hotpotqa:beir:123",
  "chunk_index": 0,
  "doc_id": "string",
  "title": "string",
  "source": "beir",
  "dataset": "hotpotqa"
}
```

`qrels.jsonl` records gold evidence for retrieval evaluation:

```json
{
  "qid": "string",
  "gold_doc_ids": ["string"],
  "gold_titles": ["string"],
  "gold_sentences": [
    {
      "doc_id": "string",
      "sent_id": 0,
      "text": "string"
    }
  ]
}
```

Initial chunking is document-level:

```text
chunk_text = title + "\n" + text
```

Do not split long documents in the first version. Keeping one corpus item as one chunk makes gold evidence mapping simpler. Passage/window splitting can be added later if retrieval quality or memory requires it.

## Teacher Trajectory Data

Teacher input uses canonical examples and the retrieval environment:

```json
{
  "qid": "string",
  "dataset": "2wiki",
  "question": "string",
  "answer": "string",
  "evidence_chain": [],
  "supporting_facts": [],
  "retrieval_budget": 5,
  "corpus_namespace": "2wiki"
}
```

Teacher output is a multi-agent trajectory:

```json
{
  "qid": "string",
  "dataset": "2wiki",
  "trajectory": [
    {
      "t": 0,
      "state": {
        "question": "string",
        "sub_goal": null,
        "evidence": [],
        "retrieval_history": [],
        "retrieval_count": 0
      },
      "agent": "planner",
      "action": {
        "type": "plan_query",
        "sub_query": "string",
        "rationale": "string"
      }
    },
    {
      "t": 1,
      "agent": "retriever",
      "action": {
        "type": "retrieve",
        "query": "string",
        "top_k": 5
      },
      "observation": {
        "retrieved_chunks": [
          {
            "chunk_id": "string",
            "rank": 1,
            "score": 0.83,
            "title": "string",
            "text": "string"
          }
        ]
      }
    },
    {
      "t": 2,
      "agent": "evidence_updater",
      "action": {
        "type": "update_evidence",
        "accepted_chunk_ids": ["string"],
        "rejected_chunk_ids": ["string"],
        "reason": "string"
      },
      "state_delta": {
        "added_evidence": []
      }
    },
    {
      "t": 3,
      "agent": "answer_generator",
      "action": {
        "type": "final_answer",
        "answer": "string",
        "supporting_chunk_ids": ["string"]
      }
    }
  ]
}
```

Trajectory constraints:

- Every action has a `type`.
- Every retrieval step records `query`, `top_k`, and `retrieved_chunks`.
- `accepted_chunk_ids` and `supporting_chunk_ids` must resolve through `chunk_meta.jsonl`.
- Final answers must cite supporting chunks.
- MuSiQue answerable data does not produce refusal trajectories.

## Quality Gates

Each conversion writes `stats.json`:

```json
{
  "num_examples": {"train": 0, "dev": 0, "test": 0},
  "num_corpus_docs": 0,
  "num_questions_with_answer": 0,
  "num_questions_with_supporting_facts": 0,
  "num_questions_with_evidence_chain": 0,
  "num_unmatched_gold_titles": 0
}
```

Each conversion writes `validation_report.json`:

```json
{
  "errors": [],
  "warnings": [
    {
      "qid": "string",
      "type": "unmatched_gold_title",
      "message": "supporting title not found in corpus"
    }
  ]
}
```

Error severity:

```text
error: sample cannot be parsed, qid missing, question missing, corpus doc_id conflict
warning: gold title not found in corpus, answer missing, evidence_chain incomplete
info: duplicate corpus item merged, long text retained as a single chunk
```

Conversion stops on errors. Conversion continues on warnings and records them.

## SFT Filtering

A sample is usable for SFT when all of the following hold:

```text
question is non-empty
answer is non-empty
at least one supporting_fact or evidence_chain exists
qid is unique
split is train or dev
```

If gold evidence cannot be mapped to corpus but QA fields are usable:

```text
usable_for_sft = false
usable_for_retrieval_eval = false
```

The sample remains in canonical examples so data loss is explicit.

## Out of Scope

The first version does not:

- train agents,
- generate teacher trajectories,
- run RL,
- change LinearRAG internals,
- perform cross-dataset corpus deduplication,
- do aggressive sentence segmentation or long-document windowing.

Those steps should be planned after the canonical data and LinearRAG adapter outputs are validated.
