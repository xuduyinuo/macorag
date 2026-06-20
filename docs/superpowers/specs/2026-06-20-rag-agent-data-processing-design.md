# RAG 多智能体数据处理设计

日期：2026-06-20

## 目标

为 HotpotQA、2Wiki 和 MuSiQue 构建统一的数据处理层，使检索环境、教师轨迹构造、SFT 训练以及后续 RL 协同优化共享一致的输入输出契约。

检索环境采用 LinearRAG 方案：`https://github.com/DEEP-PolyU/LinearRAG`。LinearRAG 当前期望每个数据集在独立目录中提供 `questions.json` 和 `chunks.json`。本设计将该格式作为适配层，而不是把 LinearRAG 的最小输入格式作为项目的规范数据格式。

MuSiQue 只使用 answerable 子集：`musique_ans_v1.0_train.jsonl` 和 `musique_ans_v1.0_dev.jsonl`。

## 选定方案

采用“统一规范层 + LinearRAG 适配层”。

规范层保留问题、答案、事实证据、推理证据链、语料文档、数据划分和质量标记。LinearRAG 适配层从规范数据派生 `questions.json` 和 `chunks.json`，并额外保留 sidecar 元数据文件，用于将检索返回的 chunk 映射回规范语料和 gold evidence。

这样既能快速跑通第一版检索环境，也不会把后续教师轨迹、SFT 和 RL 训练绑定死在某个检索实现上。

## 输出目录

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

## 统一样本 Schema

每条规范化 QA 样本写成一行 JSON：

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

统一样本中保留两类证据：

- `supporting_facts`：可引用的文本证据，主要用于检索评估和基于证据的答案生成。
- `evidence_chain`：有顺序的推理步骤或问题分解，主要用于教师轨迹构造和智能体动作监督。

## 统一语料 Schema

每条语料文档或 chunk 写成一行 JSON：

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

## 数据集映射

### HotpotQA

来源：

- QA：`data/hotpotqa/fullwiki/*.parquet`
- 语料：`data/hotpotqa/beir_corpus/corpus/*.parquet`

样本映射：

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
context_doc_ids  <- 尽量由 context.title 或 BEIR title 匹配得到
metadata.level   <- level
```

语料映射：

```text
doc_id    <- BEIR _id
title     <- title
text      <- text
sentences <- 第一版使用 [text]
source    <- "beir"
```

HotpotQA fullwiki QA 自带局部 `context`，但检索环境应以 BEIR corpus 作为主语料。QA context 只用于恢复 gold supporting text 和做对齐检查。

### 2Wiki

来源：

- QA：`data/2wiki/qa/*.parquet`
- 语料：`data/2wiki/corpus/2wiki_corpus.jsonl`

样本映射：

```text
qid                 <- id
dataset             <- "2wiki"
split               <- train / dev / test
question            <- question
answer              <- answer
answer_aliases      <- []
question_type       <- type
hop_count            <- 根据 evidences 或 type 推断，默认 2
supporting_facts    <- supporting_facts.title + supporting_facts.sent_id
evidence_chain      <- evidences 转成有序步骤
context_doc_ids     <- 由 context.title 匹配 corpus title
metadata.evidences  <- 原始 evidences
```

语料映射：

```text
doc_id            <- id
title             <- title
text              <- sentences 用空格拼接
sentences         <- sentences
source            <- "para_with_hyperlink"
metadata.mentions <- mentions
```

2Wiki 同时有句级 `supporting_facts` 和关系级 `evidences`；关系级证据应作为教师规划轨迹的种子。

### MuSiQue

来源：

- QA 和上下文：`data/musique/musique_ans_v1.0_train.jsonl`
- QA 和上下文：`data/musique/musique_ans_v1.0_dev.jsonl`

样本映射：

```text
qid                 <- id
dataset             <- "musique"
split               <- train / dev
question            <- question
answer              <- answer
answer_aliases      <- answer_aliases
question_type       <- 由 id 或 question_decomposition 推断
hop_count            <- len(question_decomposition)
supporting_facts    <- paragraph_support_idx 指向的 paragraph
evidence_chain      <- question_decomposition 按顺序转换
context_doc_ids     <- 生成的 paragraph doc_id
metadata.answerable <- true
```

语料映射：

```text
doc_id    <- musique:{dedup_hash}
title     <- paragraphs[*].title
text      <- paragraphs[*].paragraph_text
sentences <- [paragraph_text]
source    <- "musique_context"
metadata.linked_qids <- 包含该 paragraph 的 qid 列表
```

MuSiQue 下载文件中没有独立全局 corpus，因此语料从 answerable train/dev 的 `paragraphs` 去重重建。

## 清洗与去重

文本清洗保持保守，只做不改变语义的处理：

```text
1. 去除首尾空白。
2. 合并连续空白字符。
3. 保留原始大小写、标点和实体名称。
4. 只有 title 和 text 同时为空时才丢弃语料项。
5. 保留 answer=null 的 test 样本，但不用于 SFT 轨迹构造。
```

句子处理策略：

```text
已有 sentences 的数据保留原始句子列表。
只有 paragraph_text/text 的数据第一版使用 [text]。
```

这样可以避免 HotpotQA 和 2Wiki 的句子编号漂移。MuSiQue 的证据是段落级，因此段落粒度 chunk 是可接受的。

语料只在数据集内部去重，不跨数据集去重：

```text
dedup_key = normalize(title) + sha1(normalize(text))
```

`normalize` 只做小写、空白合并和首尾空白去除。重复语料合并时保留一个 `doc_id`，并在 `metadata.linked_qids` 中记录所有引用它的问题。

不做跨数据集去重，因为 HotpotQA、2Wiki、MuSiQue 的 Wikipedia 版本、句切方式和 evidence 标注不同。跨集合并会增加 gold evidence 对齐难度。

## 数据划分策略

保留原始 split，不重新划分：

```text
HotpotQA: train / dev / test
2Wiki: train / dev / test
MuSiQue answerable: train / dev
```

教师轨迹构造只使用 train/dev 中 answer 非空且 gold evidence 可解析的样本。

test 数据保留用于评估，不生成 SFT 标签。

## LinearRAG 适配层

LinearRAG 兼容的 `questions.json`：

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

LinearRAG 兼容的 `chunks.json`：

```json
[
  "Title\nDocument or paragraph text"
]
```

`chunk_meta.jsonl` 用于把 LinearRAG chunk index 映射回规范语料：

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

`qrels.jsonl` 记录检索评估所需的 gold evidence：

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

第一版采用文档级 chunk：

```text
chunk_text = title + "\n" + text
```

第一版不切长文档。保持“一条语料对应一个 chunk”可以让 `doc_id`、`chunk_id` 和 gold evidence 的映射最简单。若后续检索质量或内存表现要求，再增加 passage/window 切分。

## 教师轨迹数据

教师模型输入来自规范样本和检索环境：

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

教师模型输出统一为多智能体轨迹：

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

轨迹约束：

- 每个 action 必须有 `type`。
- 每轮检索必须记录 `query`、`top_k` 和 `retrieved_chunks`。
- `accepted_chunk_ids` 和 `supporting_chunk_ids` 必须能通过 `chunk_meta.jsonl` 解析。
- 最终答案必须引用 supporting chunks。
- MuSiQue answerable 数据不生成拒答轨迹。

## 质量门禁

每次转换写出 `stats.json`：

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

每次转换写出 `validation_report.json`：

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

错误分级：

```text
error: 样本无法解析、qid 缺失、question 缺失、corpus doc_id 冲突
warning: gold title 匹配不到 corpus、answer 缺失、evidence_chain 不完整
info: 重复语料被合并、长文本被保留为单 chunk
```

转换遇到 error 时中断。遇到 warning 时继续，但必须记录到报告中。

## SFT 过滤规则

样本满足以下条件时可用于 SFT：

```text
question 非空
answer 非空
至少存在一个 supporting_fact 或 evidence_chain
qid 唯一
split 属于 train 或 dev
```

如果 QA 字段可用，但 gold evidence 无法映射到 corpus：

```text
usable_for_sft = false
usable_for_retrieval_eval = false
```

样本仍保留在规范 examples 中，让数据损失显式可见。

## 不在第一版范围内

第一版不做：

- 智能体训练；
- 教师轨迹实际生成；
- RL 训练；
- LinearRAG 内部改造；
- 跨数据集语料去重；
- 激进句切或长文档 window 切分。

这些工作应在规范数据和 LinearRAG 适配输出验证通过后再单独计划。
