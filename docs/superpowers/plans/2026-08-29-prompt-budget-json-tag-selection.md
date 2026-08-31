# Prompt Budget JSON Tag Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make prompt compaction select the actual JSON protocol block when prompt prose mentions `<state>` or `<observation>` before that block.

**Architecture:** Enumerate overlapping tagged-block candidates with a zero-width lookahead, parse each candidate payload, and select the first JSON object. Replace the captured full-tag span while preserving the existing compaction order and token budget.

**Tech Stack:** Python 3.9, standard-library `json` and `re`, pytest, local Qwen tokenizer, GRPO/vLLM smoke runtime

## Global Constraints

- Do not change prompt templates, `max_prompt_length`, retrieval content, evidence selection, or compaction order.
- Preserve all unrelated working-tree changes.
- Do not stage or commit `src/rag/prompt_budget.py` or `tests/test_prompt_budget.py`; both are part of the user's existing untracked work and cannot form an independent commit against `HEAD`.
- Do not resume from the failed `2026-08-29_22-17-55` output directory because it contains no checkpoint.

---

### Task 1: Select parseable JSON tag candidates

**Files:**
- Modify: `tests/test_prompt_budget.py`
- Modify: `src/rag/prompt_budget.py:22-36`

**Interfaces:**
- Consumes: `compact_tagged_json_prompt(prompt: str, *, token_count: Callable[[str], int], max_tokens: int) -> CompactedPrompt`.
- Produces: `_tag_payload(text: str, tag: str) -> tuple[re.Match[str] | None, dict]`, where capture group 1 spans the complete parseable tag and capture group 2 contains its JSON text.

- [ ] **Step 1: Write the failing regression test**

Add to `tests/test_prompt_budget.py`:

```python
def test_prompt_compaction_skips_prose_tag_references() -> None:
    prompt = (
        "Use <state> and <observation> below.\n"
        '<state>{"evidence":[{"text":"old"}],'
        '"retrieval_history":[{"query":"q"}]}</state>\n'
        '<observation>{"passages":[{"text":"passage"}]}</observation>\n'
        "Return exactly."
    )
    expected = (
        "Use <state> and <observation> below.\n"
        '<state>{"evidence":[],"retrieval_history":[]}</state>\n'
        '<observation>{"passages":[]}</observation>\n'
        "Return exactly."
    )

    result = compact_tagged_json_prompt(
        prompt,
        token_count=len,
        max_tokens=len(expected),
    )

    assert result.text == expected
    assert result.removed_retrieval_history == 1
    assert result.removed_evidence == 1
    assert result.removed_observation_passages == 1
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_prompt_budget.py::test_prompt_compaction_skips_prose_tag_references
```

Expected: FAIL with `PromptBudgetError: fixed prompt content exceeds max_tokens`.

- [ ] **Step 3: Implement overlapping parseable-tag selection**

Replace `_tag_payload()` and update `_replace_tag()` in
`src/rag/prompt_budget.py`:

```python
def _tag_payload(text: str, tag: str) -> tuple[re.Match[str] | None, dict]:
    escaped_tag = re.escape(tag)
    pattern = re.compile(
        rf"(?=(<{escaped_tag}>(.*?)</{escaped_tag}>))",
        flags=re.DOTALL,
    )
    for match in pattern.finditer(text):
        try:
            payload = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return match, payload
    return None, {}


def _replace_tag(text: str, match: re.Match[str], tag: str, payload: dict) -> str:
    replacement = f"<{tag}>{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}</{tag}>"
    return text[: match.start(1)] + replacement + text[match.end(1) :]
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_prompt_budget.py::test_prompt_compaction_skips_prose_tag_references
```

Expected: `1 passed`.

- [ ] **Step 5: Run focused and related regression tests**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_prompt_budget.py tests/test_rag.py
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'not test_active_config_exposes_manual_sft_adapter_path'
```

Expected: all selected tests pass. The known stale adapter-path assertion is
explicitly deselected because it failed before this task.

- [ ] **Step 6: Run compilation and diff checks**

Run:

```bash
/data/conda/envs/macorag/bin/python -m py_compile \
  src/rag/prompt_budget.py src/rl_training/policy.py
git diff --check -- src/rag/prompt_budget.py tests/test_prompt_budget.py
```

Expected: both commands exit with status 0 and emit no errors.

- [ ] **Step 7: Run a one-step GPU smoke training**

Run from the repository root while the configured vLLM LoRA server is healthy:

```bash
bash scripts/run_train_grpo.sh \
  --max-steps 1 \
  --output-root outputs/grpo_qwen2.5-7b-v2-smoke-prompt-budget \
  --disable-tqdm
```

Expected: the first sample completes one training step without
`PromptBudgetError`, writes metrics/artifacts under a new timestamped smoke
directory, and exits successfully.

- [ ] **Step 8: Preserve the user's existing untracked file ownership**

Run:

```bash
git status --short -- src/rag/prompt_budget.py tests/test_prompt_budget.py
```

Expected: both files remain untracked or otherwise retain their prior status.
Do not stage or commit them as an isolated change because the target function
does not exist in `HEAD`.
