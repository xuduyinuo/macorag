# SFT v2

This directory now contains two deliberately separate prompt contracts:

- `prompts.yml`: frozen DeepSeek teacher-trajectory collection contract.
- `policy_prompts.yml`: canonical student policy contract for SFT, future RL,
  and evaluation. It uses tagged JSON, P-prefixed evidence pointers, and no
  Evidence/Answer rationale fields.

## Train the v2 policy

The authoritative split is read directly from:

- `data_v2/teacher_deepseek_v41_flash/train_sft.jsonl` (1,000 trajectories)
- `data_v2/teacher_deepseek_v41_flash/validation_sft.jsonl` (200 source
  trajectories; a deterministic 100-trajectory subset is used for evaluation)

Run the tokenizer/data/protocol preflight without loading the language model:

```bash
bash src/sft_v2/scripts/train_sft.sh --check-only --check-only-max-samples 3
```

Launch QLoRA SFT on the GPU selection in `config/train_sft.yml`:

```bash
bash src/sft_v2/scripts/train_sft.sh
```

Print the exact distributed command without launching:

```bash
SFT_V2_DRY_RUN=1 bash src/sft_v2/scripts/train_sft.sh
```

The data converter creates one target-only decision sample for each Query,
Evidence, and Answer action. Few-shot messages and the current observation are
input context only; their tokens are masked from the loss. The adapter exports
`prompt_contract.json`, which future RL code must verify before rollout.

This directory is an isolated SFT-v2 implementation. The first implemented
stage generates three-agent teacher trajectories from the unified 5,000-question
source file and the shared Wiki18 E5-FAISS index. It does not import or mutate
the legacy `src/sft_training` pipeline.

The current prompt contract is `sft-v2-unified-wiki18-v2`.

## Prompt contract

Every round has three independently supervised decisions:

1. Query Agent sees the question and accumulated state, then emits
   `{"sub_goal": ..., "query": ...}`.
2. Evidence Agent additionally sees the latest `P0...Pk` observation and emits
   only `{"selected_passage_ids": [...]}`. It does not produce a rationale, so
   evidence-selection supervision cannot introduce a rationale-mediated causal
   signal into the later RL stage.
3. Answer Agent sees only accumulated selected evidence and emits
   only `{"can_answer": ..., "answer": ...}`. It also produces no rationale.

Gold answers and source supporting evidence never enter these prompts. They are
used only after generation to reject unsupported or incorrect trajectories.

## Commands

Validate paths, source schema, prompt contract and index manifest without
loading the large index:

```bash
bash src/sft_v2/scripts/generate_teacher_trajectories.sh --check-only
```

Run one end-to-end pipeline sample without calling DeepSeek (this still builds
the corpus offset file and loads the FAISS index):

```bash
bash src/sft_v2/scripts/generate_teacher_trajectories.sh --dry-run --limit 1
```

Generate formal trajectories with DeepSeek V4.1 Flash:

```bash
bash src/sft_v2/scripts/generate_teacher_trajectories.sh
```

The launcher reads `DEEPSEEK_API_KEY` from the repository `.env`. The `.env`
file is Git-ignored and the key is never written to run metadata or logs.

## Outputs and resume behavior

The configured output directory receives:

```text
run_config.json
generation_ledger.jsonl
accepted_sft.jsonl
train_sft.jsonl
validation_sft.jsonl
reserve_sft.jsonl
summary.json
```

Accepted samples are never regenerated on resume. Filtered samples are skipped
by default; failed samples are retried. The executor writes results in stable
source order even though API work is concurrent.

Generation first applies the configured per-dataset candidate-pool limits, then
uses `seed` to deterministically shuffle the complete unified 5,000-example
pool. It scans that shuffled stream without per-dataset acceptance quotas and
stops at the configured qualified-trajectory target. This prevents a dataset block
near the end of the source file (for example MuSiQue) from being starved by the
early-stop rule. Dataset counts remain observed statistics rather than quotas.
A fixed-seed global hash split currently writes `train_sft.jsonl` (1,000),
`validation_sft.jsonl` (200), and `reserve_sft.jsonl` (52). The reserve rows do
not participate in the current SFT run and remain available for audit or
replacement without contaminating validation.

Every retrieval result is also deterministically shuffled per question and
round before fresh `P0...Pk` pointers are assigned. Scores remain attached to
their original passages, but `P0` is no longer synonymous with FAISS rank 0.
The shuffle is stateless and reproducible, so resuming a run cannot change the
pointer-to-passage mapping of an existing question and round.

The first real or dry run builds a compact uint64 byte-offset sidecar for the
14 GB corpus. Retrieval then reads only the top-k JSONL rows instead of loading
21 million Python dictionaries into RAM.

The console reports each initialization stage (FAISS loading, corpus-offset
building/reuse, and E5 loading). Once generation starts, a candidate-level
progress bar reports processed samples together with accepted/2,200, filtered,
failed, accepted-MuSiQue counts, and the observed average retrieval micro-batch
size. Concurrent sample workers submit queries to a coalescing-window retrieval
queue, so E5 and the 21M-vector Flat index process several queries in one call.
Corpus rows use position-independent `pread` instead of a shared seek pointer,
so document reads no longer require a coarse retrieval lock. Because generation
stops when 2,200 rows are accepted, the candidate scan can legitimately finish
before reaching its 5,000-candidate maximum.

The formal configuration uses eight sample workers, a maximum retrieval batch
of eight, and a one-second coalescing window. Completed trajectories are
consumed with `as_completed`, so a slow first item cannot hide other completed
work. The progress postfix reports started Query/API/Retrieval/Evidence/Answer
stages and Query/Answer validation retries. Invalid final-round refusals retry
only the Answer action; answer-leaking queries retry only the Query action, so
earlier valid retrieval work is not discarded.
