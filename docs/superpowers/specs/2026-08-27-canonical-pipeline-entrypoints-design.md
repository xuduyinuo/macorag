# Canonical pipeline entrypoints design

## Goal

Make the validated stratified-v2 data pipeline the only active training and
evaluation version while keeping concise, stable filenames. Running the default
shell entrypoints must never select the older non-stratified configuration.

## Root cause

The default GRPO and evaluation launchers point to `config/train_grpo.yml` and
`config/eval_macorag.yml`, while the newer data roots and sampling policy live in
separate `*_stratified_v2.yml` files. The newer files are referenced by tests but
not by default launchers. Data extraction and retrieval-index construction have
the same split between old generic files and suffixed v2 files. A fixed historical
resume launcher adds another obsolete operational entrypoint.

## Canonical configuration set

The canonical runtime files remain `config/train_grpo.yml` and
`config/eval_macorag.yml`. The GRPO config combines the stratified-v2 data roots,
proportional-stratified sampling and seed with the current BF16,
FlashAttention-2, checkpoint and optimization settings. The evaluation config
uses the stratified-v2 evaluation data and E5-FAISS index.

Extraction configurations become:

- `config/extract_train.yml`
- `config/extract_eval.yml`

They retain the existing stratified quotas, seeds, splits, expected totals and
`*_stratified_v2` output directories. The artifact directories are not renamed
or deleted.

Retrieval configurations become:

- `config/retrieval_train.yml`
- `config/retrieval_eval.yml`
- `config/retrieval_teacher.yml`

These are three distinct data roles rather than versions. They retain the
current E5-FAISS backend, `intfloat/e5-base-v2`, CPU indexing settings and their
respective stratified training, stratified evaluation and teacher trajectory
roots.

## Canonical scripts

The following scripts remain separate because they perform different jobs:

- `scripts/run_train_grpo.sh`
- `scripts/run_grpo_vllm_server.sh`
- `scripts/run_train_grpo_resume.sh`
- `scripts/eval_vllm_server.sh`
- `scripts/eval_macorag.sh`
- `scripts/build_retrieval.sh`
- `scripts/build_teacher_retrieval.sh`
- `scripts/extract_datasets.sh`

`scripts/run_train_grpo_resume.sh` is the only resume entrypoint. The fixed
checkpoint-3600 launcher is removed. `scripts/validate_pipeline.sh` replaces
`scripts/validate_retraining_v2.sh` and validates only canonical filenames.

`scripts/extract_datasets.sh` invokes the stratified extraction module with both
canonical extraction configs. `scripts/build_retrieval.sh` defaults to
`config/retrieval_eval.yml`; callers select the training config through
`CONFIG_PATH`. `scripts/build_teacher_retrieval.sh` defaults to
`config/retrieval_teacher.yml`.

## Removed files

After references are migrated, remove:

- `config/train_grpo_stratified_v2.yml`
- `config/eval_macorag_stratified_v2.yml`
- `config/extract_datasets.yml`
- `config/extract_stratified_train_v2.yml`
- `config/extract_stratified_eval_v2.yml`
- `config/build_retrieval.yml`
- `config/build_retrieval_eval_e5.yml`
- `config/build_retrieval_eval_stratified_v2_e5.yml`
- `config/build_retrieval_train_e5.yml`
- `config/build_retrieval_train_stratified_v2_e5.yml`
- `config/build_retrieval_trajectory_train_e5.yml`
- `scripts/run_train_grpo_resume_3600.sh`
- `scripts/validate_retraining_v2.sh`

The relevant content is migrated before removal. These deletions are recoverable
from Git, but `.git` is read-only in the current environment, so no commit can be
created here.

## Compatibility and scope

No `data/`, `outputs/`, model, adapter or checkpoint artifact is renamed or
deleted. SFT training, teacher generation, serving and evaluation remain distinct
operations. Adapter paths continue to be manually configured in
`train_grpo.yml` and `eval_vllm_server.yml`; no automatic model discovery is
reintroduced.

Tests and source defaults that refer to renamed canonical configs are updated.
Historical reports may retain old names as historical evidence, while active
scripts, source defaults, tests and operational documentation must not reference
removed files.

## Error handling and verification

Launchers fail before GPU work when a selected canonical config is missing or
invalid. Dry-run output prints the effective config path. Tests verify:

- default GRPO and evaluation roots are the stratified-v2 artifacts;
- the proportional-stratified sampling policy is active in `train_grpo.yml`;
- canonical extraction and retrieval configs preserve their data roles;
- removed versioned files and the fixed resume launcher do not exist;
- active source, scripts and tests contain no references to removed filenames;
- modified Bash scripts pass syntax checks and focused pipeline tests pass.
