# MACORAG RL-v2

This directory is an isolated MAPPO implementation aligned with the latest
SFT-v2 student policy contract (`macorag-policy-v3`). It owns its Python code,
prompt YAML, training YAML, tests and two-GPU launcher. It does not import
`src/rl_mappo` or `src/sft_v2` at runtime.

## Fixed data and prompt contracts

- train: `data_v2/train_rl.jsonl` (6,000 questions)
- validation source: `data_v2/dev_rl.jsonl` (300 questions); formal runs use a
  deterministic proportional 200-question subset (80 2Wiki, 80 HotpotQA,
  40 MuSiQue)
- policy prompts: `src/rl_v2/policy_prompts.yml`
- initialization: the SFT-v2 adapter configured by `sft_adapter_path`
- retrieval: the exact `run_macorag_eval.py` Wiki18 E5-FAISS contract

Training loads the complete train JSONL and deterministically shuffles it with
`seed` before applying an optional smoke-test limit. Each epoch is shuffled
again deterministically. The validation JSONL is never split from training.
The run writes both source SHA-256 values and the exact shuffled train ID order
to `data_split.json`.

At startup, RL verifies that its prompt YAML version and SHA-256 fingerprint
exactly match the SFT adapter's `prompt_contract.json`. Query, Evidence and
Answer inputs use the same system prompts, few-shots, tagged JSON shapes and
P-prefixed temporary evidence pointers used by SFT-v2. Evidence and Answer do
not contain rationale fields.

## Two-GPU and probability contract

`run_mappo.sh` uses physical GPU 0 for the local QLoRA actor plus centralized
critic. Physical GPU 1 runs vLLM and the FP16 E5 query encoder, matching the
evaluation program's co-located generator/retriever topology. FAISS remains on
CPU. The vLLM LoRA is reloaded after every MAPPO update, before the next batch.

Validation and training rollout collection are parallel across independent
samples while every individual trajectory remains sequential. The formal
profile uses 32 outer workers, 16 vLLM engine sequence slots and a 20 ms E5
coalescing window. Historical logs on this host report roughly 16.3 concurrent
4,096-token requests, so the extra outer workers keep those slots occupied
while other trajectories are retrieving or changing roles. Retrieval batch
size is an upper bound; `retrieval_batch_stats` in validation/train metrics
records the actual average and maximum batch sizes.

Model, E5 and FAISS initialization completes before validation timing starts.
Each validation record reports wall-clock seconds/sample and accumulated
generation, retrieval and critic time. Training records rollout, PPO update
and vLLM adapter-sync time separately. The 3 s/sample and 15-minute full-dev
targets must be confirmed by the next real GPU run; they are not inferred from
the configured worker count alone.

The 24 GiB learner uses a GPU-verified 1,024-token prompt ceiling. Long dynamic states keep
the question/header at the beginning and newest evidence/instructions at the
end, with an explicit middle-truncation marker; the complete 383-483-token
SFT-v2 role/few-shot prefix is always retained. Before the expensive baseline
validation, `learner_memory_preflight.json` records a worst-budget trainable
forward/backward result so an unsafe memory configuration fails immediately.
On the 24 GiB RTX 4090 used for the formal run, 1,536 and 2,048 tokens OOMed in
this preflight, while a 1,024-token prompt plus a 128-token action peaked at
19.74 GiB allocated and 19.85 GiB reserved.

Retrieval matches `/data/xudu/baseline/baseline_eval/scripts/run_macorag_eval.py`:
the same local E5 checkpoint, `query: ` prefix, 128-token query limit, mean
pooling, L2 normalization, FP16 query encoding, CPU Flat-IP index, Wiki18
corpus, top-5 rank order, `P0...P4` assignment and 1,200-character passage
limit. Results are not shuffled and previously selected documents are not
removed from later observations.

The formal config uses unmodified sampling (`temperature=1`, `top_p=1`,
`top_k=-1`). vLLM-returned BF16 log-probabilities are diagnostic only. Before
every PPO update, the learner re-scores the fixed prompt/action token IDs with
the current local QLoRA snapshot and replaces `old_token_logprobs`; new and old
probabilities therefore use the same tokenizer, chat template, quantized model
and numerical path. Evidence guided decoding is reproduced by the same legal
pointer trie in learner scoring. Answer guided decoding is disabled because its
grammar partition is not used by learner scoring.
The vLLM service is launched with `--generation-config vllm`, preventing the
base model's generation config from silently adding repetition penalties.

## Commands

Static data/config/prompt-contract preflight (no model or vLLM process):

```bash
/data/conda/envs/macorag/bin/python -m src.rl_v2.train_mappo \
  --config src/rl_v2/train_mappo.yml --check-only
```

Print the exact two-GPU launch commands without starting either model:

```bash
MAPPO_LAUNCH_DRY_RUN=1 PYTHON=/data/conda/envs/macorag/bin/python \
  bash src/rl_v2/run_mappo.sh
```

Run focused tests:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q src/rl_v2/test_rl_v2.py
```

Start formal training:

```bash
PYTHON=/data/conda/envs/macorag/bin/python bash src/rl_v2/run_mappo.sh
```

Resume a complete checkpoint:

```bash
PYTHON=/data/conda/envs/macorag/bin/python bash src/rl_v2/run_mappo.sh \
  --resume-from-checkpoint outputs/rl_v2_Qwen2.5-7B-Instruct/<run>/checkpoint-<step>
```

For a small smoke run, pass `--max-samples`. The loader always shuffles the
complete 6,000-row source first and only then takes the requested number; it
does not modify or truncate the source JSONL:

```bash
PYTHON=/data/conda/envs/macorag/bin/python bash src/rl_v2/run_mappo.sh \
  --max-samples 16
```
