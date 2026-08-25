# Stratified Train and Evaluation Extraction Design

## Goal

Create reproducible, additive MACORAG datasets with exactly 2,000 training
examples and 1,000 evaluation examples per source dataset. Training examples
come only from the canonical `train` split, evaluation examples come only from
the canonical `dev` split, and both outputs follow the usable `dev`
distribution so that training coverage matches the evaluation protocol.

The new artifacts must not overwrite `data/rl_train_2000`, `data/eval_1000`, or
any existing retrieval index.

## Source Contract

The canonical sources are:

- `data/processed/2wiki/2wiki_{train,dev}.jsonl`
- `data/processed/hotpotqa/hotpotqa_{train,dev}.jsonl`
- `data/processed/musique/musique_{train,dev}.jsonl`
- `data/processed/<dataset>/corpus.jsonl`

Training eligibility requires all of the following:

- `usable_for_sft` is exactly `true`;
- `usable_for_retrieval_eval` is exactly `true`;
- `quality_flags` is empty;
- `qid`, `question`, `answer`, and `supporting_facts` satisfy the canonical
  schema.

Evaluation eligibility requires `usable_for_retrieval_eval` to be exactly
`true`, `quality_flags` to be empty, and the same canonical schema fields to be
valid. It does not require `usable_for_sft`.

Rows are never moved between source splits. Before writing outputs, the
extractor must verify zero train/evaluation overlap by both `qid` and normalized
question text. Within each output split, `qid` and normalized question text must
be unique. When duplicate normalized questions exist in a source split, at most
one may be selected; this also prevents the known conflicting-answer duplicate
groups in the processed training data from entering the new training set.

## Stratification Contract

Quotas are fixed configuration, not recomputed silently on every run. They were
calculated from the eligible canonical `dev` population using largest-remainder
apportionment. An extraction fails if a configured stratum lacks enough eligible
unique candidates.

### 2Wiki

The stratum key is `question_type`.

| Question type | Train | Evaluation |
|---|---:|---:|
| `compositional` | 831 | 415 |
| `comparison` | 486 | 243 |
| `bridge_comparison` | 440 | 220 |
| `inference` | 243 | 122 |
| **Total** | **2,000** | **1,000** |

### HotpotQA

The stratum key is `(metadata.level, question_type)`. Because the canonical dev
split is entirely `hard`, both new splits use hard questions only.

| Difficulty and type | Train | Evaluation |
|---|---:|---:|
| `hard / bridge` | 1,596 | 798 |
| `hard / comparison` | 404 | 202 |
| **Total** | **2,000** | **1,000** |

### MuSiQue

`question_type` and difficulty are absent, so the stratum key is the topology
prefix of `qid`. This preserves topology variants inside each hop count.

| QID topology | Train | Evaluation |
|---|---:|---:|
| `2hop` | 1,036 | 518 |
| `3hop1` | 470 | 235 |
| `3hop2` | 159 | 79 |
| `4hop1` | 203 | 102 |
| `4hop2` | 53 | 27 |
| `4hop3` | 79 | 39 |
| **Total** | **2,000** | **1,000** |

The aggregate MuSiQue proportions are therefore approximately 51.8% 2-hop,
31.4% 3-hop, and 16.8% 4-hop in both splits.

## Deterministic Sampling

The extraction seed is `20260826`. Each dataset/split/stratum receives an
independent deterministic random stream derived from the global seed and the
literal dataset, split, and stratum names. Selection is without replacement.
After all strata are combined, a second deterministic shuffle removes stratum
ordering from the output file.

Given identical source bytes, configuration, and code, rerunning extraction in
a fresh workspace must produce byte-identical example JSONL and manifest
content. Output paths in manifests are relative to the output root, source paths
are repository-relative, and manifests contain no wall-clock-dependent values.

## Output Contract

Examples and their scoped corpus are written additively under:

```text
data/rl_train_2000_stratified_v2/
  2wiki/2wiki_train.jsonl
  2wiki/corpus.jsonl
  2wiki/extract_summary.json
  hotpotqa/hotpotqa_train.jsonl
  hotpotqa/corpus.jsonl
  hotpotqa/extract_summary.json
  musique/musique_train.jsonl
  musique/corpus.jsonl
  musique/extract_summary.json
  extraction_manifest.json

data/eval_1000_stratified_v2/
  2wiki/2wiki_dev.jsonl
  2wiki/corpus.jsonl
  2wiki/extract_summary.json
  hotpotqa/hotpotqa_dev.jsonl
  hotpotqa/corpus.jsonl
  hotpotqa/extract_summary.json
  musique/musique_dev.jsonl
  musique/corpus.jsonl
  musique/extract_summary.json
  extraction_manifest.json
```

The extractor fails closed if the target root already exists. It first writes
to a uniquely named sibling staging directory and renames the completed tree
only after every dataset passes validation, so interruption cannot leave an
apparently complete target. A handled extraction failure removes its staging
directory. A process-level interruption may leave a uniquely named staging
directory, but a later run neither resumes nor publishes it and reports the
stale staging path for manual inspection.

Each scoped corpus contains exactly the union of:

- every selected example's `context_doc_ids`;
- every selected supporting fact's nonempty `doc_id`.

Every required document must occur exactly once in the scoped corpus. Missing
or duplicate required documents are fatal errors. Unreferenced documents are
not copied.

## Manifest and Provenance

Each top-level `extraction_manifest.json` records:

- schema version and extraction seed;
- source split and corpus paths plus SHA-256 digests;
- eligibility and exclusion counts by reason;
- configured and actual quota by stratum;
- selected source line indices and qids;
- example count, unique-qid count, and unique-question count;
- required and extracted corpus document counts;
- SHA-256 digests for every emitted JSONL file;
- the train/evaluation overlap audit result.

Per-dataset summaries repeat the local fields needed to audit a dataset without
loading the top-level manifest. Paths in manifests must point to the actual new
roots; stale paths such as the existing `data/rl_train_30` summary values are
not permitted.

## Implementation Boundary

Add a dedicated stratified extraction entry point and two explicit YAML
configurations rather than changing the behavior of the existing unstratified
reservoir sampler. The old `data/rl_train_2000` and `data/eval_1000` workflows
remain usable.

The new extractor owns:

1. schema and eligibility validation;
2. deterministic per-stratum selection;
3. normalized-question deduplication;
4. exact scoped-corpus extraction;
5. atomic output publication;
6. manifest generation and final audits.

No model inference, teacher trajectory generation, E5 embedding, or FAISS build
is part of extraction itself.

## Downstream Integration

After extraction validates successfully:

- GRPO must use `rl_data_root: data/rl_train_2000_stratified_v2` and
  `max_samples: 2000`; leaving the current `max_samples: 1000` would silently
  use only half of each new training split.
- Evaluation must use `data_root: data/eval_1000_stratified_v2` with
  `max_samples: null`.
- New E5/FAISS retrieval roots must be built from the new scoped corpora; old
  indexes must never be reused with new question/corpus roots.
- Recommended index roots are
  `data/rl_train_2000_stratified_v2_e5_faiss` and
  `data/eval_1000_stratified_v2_e5_faiss`.

Default training and evaluation configurations should switch to the new roots
only after extraction and retrieval-index validation pass. Existing artifacts
remain available for reproducibility and comparison.

## Validation and Acceptance Criteria

Automated tests use small synthetic JSONL/corpus fixtures and cover:

- exact per-stratum quota fulfillment;
- exclusion of false usability flags and nonempty quality flags;
- rejection of missing fields and underfilled strata;
- deterministic output under the fixed seed;
- distinct deterministic samples when the seed changes;
- normalized-question deduplication;
- scoped-corpus completeness and uniqueness;
- refusal to overwrite an existing target;
- cleanup of staging on handled failure and refusal to publish stale staging;
- manifest paths, counts, hashes, and overlap results.

The real-data extraction is accepted only when all of the following hold:

- train counts are exactly 2,000 per dataset and 6,000 total;
- evaluation counts are exactly 1,000 per dataset and 3,000 total;
- every configured stratum count matches this design;
- every output row has the correct source split and eligibility flag;
- all quality flags are empty;
- qids and normalized questions are unique within each output split;
- train/evaluation qid and normalized-question intersections are empty;
- required corpus count equals extracted unique corpus count for every dataset;
- manifests reproduce every output digest and contain no stale output paths;
- a second extraction into a fresh temporary root is byte-identical.

## Non-Goals

- Changing the canonical processed datasets.
- Replacing the official dev-based evaluation protocol.
- Mixing train rows into evaluation or dev rows into training.
- Rebalancing datasets globally beyond 2,000 train and 1,000 evaluation rows
  per dataset.
- Running teacher generation, GRPO, evaluation, or full retrieval-index builds
  as part of the extraction command.
