# Canonical Pipeline Entrypoints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the validated stratified training/evaluation pipeline the only active pipeline while exposing one concise configuration or shell-script name per operational role.

**Architecture:** Keep stable runtime names (`train_grpo.yml`, `eval_macorag.yml`) and move the validated stratified data contracts into them. Give extraction and retrieval files role-based names, update every active default and test to those names, then remove the superseded versioned and duplicated files. Preserve all generated data, indexes, adapters, checkpoints, outputs, and historical design documents.

**Tech Stack:** Bash, Python 3.9+, PyYAML, pytest.

## Global Constraints

- Preserve unrelated worktree changes; inspect each overlapping diff before editing.
- Use `apply_patch` for source, config, script, test, and documentation edits.
- Do not rename or delete anything under `data/`, `outputs/`, model directories, adapter directories, or checkpoint directories.
- Keep adapter selection explicit in configuration: `sft_adapter_path` in `config/train_grpo.yml` and `adapter_path` in `config/eval_vllm_server.yml`.
- Do not restore adapter auto-discovery or the removed LLM-judge metric.
- Historical files under `docs/superpowers/specs/` and old implementation plans may retain old filenames as historical evidence.
- `.git` is read-only in this environment. The commit commands below document intended commit boundaries but cannot be executed here.

---

### Task 1: Promote stratified runtime settings into the canonical training and evaluation configs

**Files:**

- Modify: `tests/test_retraining_launchers.py`
- Modify: `tests/test_rl_data_sampling.py`
- Modify: `tests/test_evaluation.py`
- Modify: `config/train_grpo.yml`
- Modify: `config/eval_macorag.yml`

- [ ] **Step 1: Write failing assertions for the canonical GRPO config**

In `tests/test_retraining_launchers.py`, extend the canonical configuration contract so that `_yaml("train_grpo.yml")` must contain:

```python
assert grpo["rl_data_root"] == "data/rl_train_2000_stratified_v2"
assert grpo["retrieval_root"] == "data/rl_train_2000_stratified_v2_e5_faiss"
assert grpo["max_samples"] == 2000
assert grpo["data_sampling_strategy"] == "proportional_stratified"
assert grpo["data_sampling_seed"] == 20260826
assert grpo["sft_adapter_path"] == "/path/to/sft_adapter"
assert grpo["bf16"] is True
assert grpo["fp16"] is False
assert grpo["attn_implementation"] == "flash_attention_2"
```

In `tests/test_rl_data_sampling.py`, replace the test that compares the default and `train_grpo_stratified_v2.yml` files with a canonical-only test:

```python
def test_train_grpo_config_enables_proportional_stratified_sampling() -> None:
    config = yaml.safe_load(Path("config/train_grpo.yml").read_text(encoding="utf-8"))
    assert config["max_samples"] == 2000
    assert config["data_sampling_strategy"] == "proportional_stratified"
    assert config["data_sampling_seed"] == 20260826
```

- [ ] **Step 2: Write failing assertions for the canonical evaluation config**

In `tests/test_retraining_launchers.py` and the existing config contract in `tests/test_evaluation.py`, require:

```python
assert evaluation["data_root"] == "data/eval_1000_stratified_v2"
assert evaluation["retrieval_root"] == "data/eval_1000_stratified_v2_e5_faiss"
```

Keep the existing assertions that evaluation uses only local metrics and that the serving adapter remains explicitly configured.

- [ ] **Step 3: Run the focused tests and confirm they fail on the old roots**

Run:

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_retraining_launchers.py \
  tests/test_rl_data_sampling.py \
  tests/test_evaluation.py \
  -k 'config or sampling or canonical'
```

Expected: failures identify `data/rl_train_2000`, `data/eval_1000`, or the absence of proportional sampling in the simple-name configs.

- [ ] **Step 4: Merge the validated GRPO settings into `config/train_grpo.yml`**

Change only the data/sampling portion of the current canonical file:

```yaml
rl_data_root: "data/rl_train_2000_stratified_v2"
retrieval_root: "data/rl_train_2000_stratified_v2_e5_faiss"

max_samples: 2000
data_sampling_strategy: "proportional_stratified"
data_sampling_seed: 20260826
```

Retain the current manual `sft_adapter_path`, BF16, FP16, FlashAttention-2, gradient-checkpointing, vLLM, reward, optimization, logging, and checkpoint values. Update the opening comment to say this is the canonical stratified pipeline, without a version suffix in the filename.

- [ ] **Step 5: Promote the validated evaluation roots into `config/eval_macorag.yml`**

Set:

```yaml
data_root: "data/eval_1000_stratified_v2"
retrieval_root: "data/eval_1000_stratified_v2_e5_faiss"
```

Do not change local metric behavior, generation settings, retrieval settings, service URLs, or output paths.

- [ ] **Step 6: Re-run the focused tests**

Run the Step 3 command.

Expected: the new canonical-root and sampling assertions pass. Tests that still explicitly open files scheduled for removal may remain until Task 4.

- [ ] **Step 7: Record the intended commit boundary**

```bash
git add config/train_grpo.yml config/eval_macorag.yml \
  tests/test_retraining_launchers.py tests/test_rl_data_sampling.py tests/test_evaluation.py
git commit -m "refactor: promote stratified runtime configs"
```

Do not run these commands while `.git` remains read-only.

---

### Task 2: Replace versioned extraction configs with concise role-based configs

**Files:**

- Modify: `tests/test_stratified_extraction.py`
- Create: `config/extract_train.yml`
- Create: `config/extract_eval.yml`
- Modify: `scripts/extract_datasets.sh`
- Modify: `src/data_processing/extract_stratified_datasets.py`

- [ ] **Step 1: Write failing canonical extraction tests**

Update the extraction config tests to load `config/extract_train.yml` and `config/extract_eval.yml`. Assert the complete operational contract, including:

```python
assert train_config["split"] == "train"
assert train_config["seed"] == 20260826
assert train_config["expected_total"] == 2000
assert train_config["output_root"] == "data/rl_train_2000_stratified_v2"
assert eval_config["split"] == "dev"
assert eval_config["seed"] == 20260826
assert eval_config["expected_total"] == 1000
assert eval_config["output_root"] == "data/eval_1000_stratified_v2"
```

Retain the existing exact per-dataset and per-stratum quota assertions. Add a launcher assertion that `scripts/extract_datasets.sh` names both canonical config files and invokes the paired `data_processing.extract_stratified_datasets` entrypoint once.

- [ ] **Step 2: Run the extraction tests and confirm the new files are missing**

Run:

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_stratified_extraction.py -k 'config or launcher or runtime'
```

Expected: failure because `extract_train.yml`, `extract_eval.yml`, or the canonical launcher references do not yet exist.

- [ ] **Step 3: Create `config/extract_train.yml` with the exact validated train contract**

Copy the complete content of `config/extract_stratified_train_v2.yml` unchanged except for the filename. This must preserve:

- `source_root: data/processed`
- `output_root: data/rl_train_2000_stratified_v2`
- `split: train`, seed `20260826`, total `2000`
- 2Wiki quotas `831/486/440/243`
- HotpotQA quotas `1596/404`
- MuSiQue quotas `1036/470/159/203/53/79`

- [ ] **Step 4: Create `config/extract_eval.yml` with the exact validated eval contract**

Copy the complete content of `config/extract_stratified_eval_v2.yml` unchanged except for the filename. Preserve:

- `source_root: data/processed`
- `output_root: data/eval_1000_stratified_v2`
- `split: dev`, seed `20260826`, total `1000`
- 2Wiki quotas `415/243/220/122`
- HotpotQA quotas `798/202`
- MuSiQue quotas `518/235/79/102/27/39`

- [ ] **Step 5: Make `scripts/extract_datasets.sh` run both canonical extraction jobs**

Keep strict Bash mode and the existing `PYTHONPATH` setup. Replace the legacy trajectory extractor call with:

```bash
"${PYTHON:-python}" -m data_processing.extract_stratified_datasets \
  --train-config "${REPO_ROOT}/config/extract_train.yml" \
  --eval-config "${REPO_ROOT}/config/extract_eval.yml" \
  "$@"
```

Forward optional arguments to the paired call. Add a dry-run branch before execution that prints both resolved config paths, so validation does not regenerate data.

- [ ] **Step 6: Update source defaults to the canonical filenames**

In `src/data_processing/extract_stratified_datasets.py`, change the paired defaults to `config/extract_train.yml` and `config/extract_eval.yml` while retaining the existing `--train-config` and `--eval-config` interface.

- [ ] **Step 7: Verify Bash syntax, dry-run output, and extraction tests**

Run:

```bash
bash -n scripts/extract_datasets.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/extract_datasets.sh
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_stratified_extraction.py
```

Expected: no data is regenerated; dry-run output lists `extract_train.yml` and `extract_eval.yml`; tests pass apart from old-file absence checks deferred to Task 4.

- [ ] **Step 8: Record the intended commit boundary**

```bash
git add config/extract_train.yml config/extract_eval.yml \
  scripts/extract_datasets.sh src/data_processing/extract_stratified_datasets.py \
  tests/test_stratified_extraction.py
git commit -m "refactor: simplify extraction entrypoints"
```

Do not run these commands while `.git` remains read-only.

---

### Task 3: Replace duplicated retrieval configs with role-based configs

**Files:**

- Modify: `tests/test_retrieval_env.py`
- Modify: `tests/test_retraining_launchers.py`
- Modify: `tests/test_stratified_extraction.py`
- Create: `config/retrieval_train.yml`
- Create: `config/retrieval_eval.yml`
- Create: `config/retrieval_teacher.yml`
- Modify: `scripts/build_retrieval.sh`
- Modify: `scripts/build_teacher_retrieval.sh`
- Modify: `src/data_processing/retrieval_cli.py`

- [ ] **Step 1: Write failing tests for the three retrieval roles**

Replace old retrieval filenames in tests with the role-based names. Assert:

```python
assert train["data_root"] == "data/rl_train_2000_stratified_v2"
assert train["retrieval_root"] == "data/rl_train_2000_stratified_v2_e5_faiss"
assert evaluation["data_root"] == "data/eval_1000_stratified_v2"
assert evaluation["retrieval_root"] == "data/eval_1000_stratified_v2_e5_faiss"
assert teacher["data_root"] == "data/trajectory_train"
assert teacher["retrieval_root"] == "data/trajectory_train_e5_faiss"
for config in (train, evaluation, teacher):
    assert config["backend"] == "e5_faiss"
    assert config["embedding_model"] == "intfloat/e5-base-v2"
    assert config["device"] == "cpu"
```

Require `build_retrieval.sh` and `retrieval_cli.py` to default to `config/retrieval_eval.yml`, and `build_teacher_retrieval.sh` to default to `config/retrieval_teacher.yml`.

- [ ] **Step 2: Run the tests and confirm the canonical retrieval files are missing**

Run:

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_retrieval_env.py tests/test_retraining_launchers.py \
  tests/test_stratified_extraction.py -k 'retrieval or launcher or config'
```

- [ ] **Step 3: Create the three canonical retrieval configs**

Create exact role-preserving copies:

- `retrieval_train.yml` from `build_retrieval_train_stratified_v2_e5.yml`
- `retrieval_eval.yml` from `build_retrieval_eval_stratified_v2_e5.yml`
- `retrieval_teacher.yml` from `build_retrieval_trajectory_train_e5.yml`

All three retain `command: build`, datasets `2wiki/hotpotqa/musique`, E5 model, CPU device, max length `512`, batch size `128`, and top-k `5`.

- [ ] **Step 4: Update launcher and source defaults**

Set:

```bash
# scripts/build_retrieval.sh
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/retrieval_eval.yml}"

# scripts/build_teacher_retrieval.sh
CONFIG_PATH="${CONFIG_PATH:-${REPO_ROOT}/config/retrieval_teacher.yml}"
```

Change `DEFAULT_RETRIEVAL_CONFIG` in `src/data_processing/retrieval_cli.py` to `config/retrieval_eval.yml`. Keep `CONFIG_PATH=config/retrieval_train.yml bash scripts/build_retrieval.sh` as the explicit training-index invocation.

- [ ] **Step 5: Verify syntax, defaults, overrides, and focused tests**

Run:

```bash
bash -n scripts/build_retrieval.sh scripts/build_teacher_retrieval.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/build_retrieval.sh
MACORAG_LAUNCH_DRY_RUN=1 CONFIG_PATH=config/retrieval_train.yml bash scripts/build_retrieval.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/build_teacher_retrieval.sh
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_retrieval_env.py tests/test_retraining_launchers.py \
  tests/test_stratified_extraction.py -k 'retrieval or launcher or config'
```

Expected: dry runs print the eval, train override, and teacher role configs respectively; no index build starts.

- [ ] **Step 6: Record the intended commit boundary**

```bash
git add config/retrieval_train.yml config/retrieval_eval.yml config/retrieval_teacher.yml \
  scripts/build_retrieval.sh scripts/build_teacher_retrieval.sh \
  src/data_processing/retrieval_cli.py tests/test_retrieval_env.py \
  tests/test_retraining_launchers.py tests/test_stratified_extraction.py
git commit -m "refactor: simplify retrieval config roles"
```

Do not run these commands while `.git` remains read-only.

---

### Task 4: Consolidate resume and validation launchers, then remove superseded files

**Files:**

- Modify: `tests/test_rl_training.py`
- Modify: `tests/test_rl_checkpointing.py`
- Modify: `tests/test_retraining_launchers.py`
- Modify: `tests/test_stratified_extraction.py`
- Create: `scripts/validate_pipeline.sh`
- Delete: `scripts/validate_retraining_v2.sh`
- Delete: `scripts/run_train_grpo_resume_3600.sh`
- Delete: `config/train_grpo_stratified_v2.yml`
- Delete: `config/eval_macorag_stratified_v2.yml`
- Delete: `config/extract_datasets.yml`
- Delete: `config/extract_stratified_train_v2.yml`
- Delete: `config/extract_stratified_eval_v2.yml`
- Delete: `config/build_retrieval.yml`
- Delete: `config/build_retrieval_eval_e5.yml`
- Delete: `config/build_retrieval_eval_stratified_v2_e5.yml`
- Delete: `config/build_retrieval_train_e5.yml`
- Delete: `config/build_retrieval_train_stratified_v2_e5.yml`
- Delete: `config/build_retrieval_trajectory_train_e5.yml`

- [ ] **Step 1: Replace fixed-resume tests with the generic full-checkpoint contract**

Remove `test_run_train_grpo_resume_3600_script_has_fixed_resume_contract`. Strengthen the generic resume tests to require:

- a user-supplied `RESUME_CHECKPOINT` or first positional checkpoint;
- `checkpoint_manifest.json` and `COMPLETE` validation;
- delegation to `run_train_grpo.sh` with `--resume-from-checkpoint`;
- no embedded historical output path, epoch, sample count, or global step.

- [ ] **Step 2: Update launcher tests to expect `validate_pipeline.sh`**

Rename active validation-script references in `tests/test_retraining_launchers.py`. Add assertions that its dry run succeeds and that the script reads only canonical filenames.

- [ ] **Step 3: Add an explicit obsolete-file and active-reference test**

In `tests/test_retraining_launchers.py`, define the exact removed paths and assert they do not exist. Scan only active surfaces (`config/`, `scripts/`, `src/`, `tests/`, and top-level operational docs such as `README.md`), excluding historical `docs/superpowers/specs/` and `docs/superpowers/plans/`, and assert none of the removed basenames appear.

- [ ] **Step 4: Run the new tests and confirm they fail before cleanup**

Run:

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_rl_training.py tests/test_rl_checkpointing.py \
  tests/test_retraining_launchers.py tests/test_stratified_extraction.py \
  -k 'resume or validate or obsolete or canonical or config'
```

Expected: failures identify the fixed resume script, old validator, versioned configs, and remaining active references.

- [ ] **Step 5: Create `scripts/validate_pipeline.sh`**

Move the read-only preflight logic from `validate_retraining_v2.sh` into the concise filename. Keep strict Bash mode, dry-run behavior, prompt-contract loading, and cross-stage checks for:

```python
names = ("generate_teacher_sft.yml", "train_sft.yml", "train_grpo.yml", "eval_macorag.yml")
```

Extend the check to load the canonical extraction and retrieval role files and verify that runtime data roots exactly match their producer/index roots. Do not require GPU access or regenerate artifacts.

- [ ] **Step 6: Migrate remaining active references**

Use:

```bash
rg -n 'train_grpo_stratified_v2|eval_macorag_stratified_v2|extract_stratified_(train|eval)_v2|build_retrieval(_trajectory_train_e5|_(train|eval)(_stratified_v2)?_e5)?|validate_retraining_v2|run_train_grpo_resume_3600' \
  config scripts src tests README.md
```

Update every match to a canonical role-based filename or remove the obsolete fixed behavior. Do not rewrite historical design/spec/plan files.

- [ ] **Step 7: Delete only the confirmed superseded files**

Use `apply_patch` to delete the 13 paths listed in this task after their content and references have been migrated. Do not delete artifact directories with similar suffixes.

- [ ] **Step 8: Verify the consolidated launcher set**

Run:

```bash
bash -n scripts/run_train_grpo.sh scripts/run_grpo_vllm_server.sh \
  scripts/run_train_grpo_resume.sh scripts/eval_vllm_server.sh \
  scripts/eval_macorag.sh scripts/build_retrieval.sh \
  scripts/build_teacher_retrieval.sh scripts/extract_datasets.sh \
  scripts/validate_pipeline.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/validate_pipeline.sh
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_rl_training.py tests/test_rl_checkpointing.py \
  tests/test_retraining_launchers.py tests/test_stratified_extraction.py
```

- [ ] **Step 9: Record the intended commit boundary**

```bash
git add scripts/validate_pipeline.sh scripts/run_train_grpo_resume.sh \
  tests/test_rl_training.py tests/test_rl_checkpointing.py \
  tests/test_retraining_launchers.py tests/test_stratified_extraction.py
git add -u config scripts
git commit -m "refactor: remove obsolete pipeline versions"
```

Do not run these commands while `.git` remains read-only.

---

### Task 5: Final verification and handoff

**Files:**

- Verify: `config/`
- Verify: `scripts/`
- Verify: `src/`
- Verify: `tests/`
- Verify: `README.md`

- [ ] **Step 1: Verify the concise file inventory**

Run:

```bash
find config -maxdepth 1 -type f -printf '%f\n' | sort
find scripts -maxdepth 1 -type f -name '*.sh' -printf '%f\n' | sort
```

Confirm that each affected role has only these active names:

- runtime: `train_grpo.yml`, `eval_macorag.yml`, `eval_vllm_server.yml`
- extraction: `extract_train.yml`, `extract_eval.yml`
- retrieval: `retrieval_train.yml`, `retrieval_eval.yml`, `retrieval_teacher.yml`
- resume/validation: `run_train_grpo_resume.sh`, `validate_pipeline.sh`

- [ ] **Step 2: Verify there are no active references to removed names**

Run the Task 4 Step 6 `rg` command.

Expected: no output. Historical references under `docs/superpowers/specs/` and `docs/superpowers/plans/` are intentionally outside this check.

- [ ] **Step 3: Run all affected focused tests**

Run:

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_retraining_launchers.py \
  tests/test_rl_data_sampling.py \
  tests/test_rl_training.py \
  tests/test_rl_checkpointing.py \
  tests/test_evaluation.py \
  tests/test_retrieval_env.py \
  tests/test_stratified_extraction.py
```

- [ ] **Step 4: Run static verification**

Run:

```bash
/data/conda/envs/macorag/bin/python -m compileall -q src
bash -n scripts/*.sh
git diff --check
```

- [ ] **Step 5: Run safe launcher dry runs**

Run:

```bash
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/run_train_grpo.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/run_grpo_vllm_server.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/eval_vllm_server.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/eval_macorag.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/build_retrieval.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/build_teacher_retrieval.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/extract_datasets.sh
MACORAG_LAUNCH_DRY_RUN=1 bash scripts/validate_pipeline.sh
```

Expected: each launcher prints a concise canonical config path and exits before model loading, GPU work, extraction, retrieval indexing, or output mutation.

- [ ] **Step 6: Audit the final diff and report validation boundaries**

Run:

```bash
git status --short
git diff -- config scripts src tests README.md
```

Report:

- the canonical command/config mapping;
- the exact obsolete files removed;
- focused/static/dry-run results;
- any unrelated pre-existing failures separately;
- that no data, index, adapter, checkpoint, model, or output artifact was removed;
- that no commit was created because `.git` is read-only.
