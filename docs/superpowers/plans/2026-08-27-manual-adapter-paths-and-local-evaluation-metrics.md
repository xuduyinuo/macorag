# Manual Adapter Paths and Local Evaluation Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Require operators to provide adapter paths at launch time and remove external LLM judging while always writing deterministic local evaluation metrics.

**Architecture:** Shell launchers are the operational boundary: they accept one required environment variable, validate adapter artifacts, and pass the selected path to the existing Python CLI. Evaluation uses a small local scorer over `predictions.jsonl`; the main evaluator calls it unconditionally and contains no API-judge branch.

**Tech Stack:** Bash, Python 3.10, argparse, PyYAML, pytest.

## Final contract correction

The user clarified after initial implementation that adapter paths must remain
visible in YAML. The final implementation therefore reads `sft_adapter_path`
from the GRPO config and `adapter_path` from the evaluation vLLM config. The
environment-variable examples in the original Task 1 steps below are superseded
by this config-backed contract; automatic directory scanning remains prohibited.

## Global Constraints

- Do not select adapters by timestamp, directory sorting, or any automatic fallback.
- Do not persist experiment-specific adapter paths in YAML defaults.
- Require `SFT_ADAPTER_PATH` for GRPO training and its LoRA vLLM service.
- Require `ADAPTER_PATH` for the evaluation vLLM service.
- Preserve existing EM, containment accuracy, F1, prediction, retrieval, prompt, and checkpoint behavior.
- Remove all Bailian/Qwen judge calls, `llm_accuracy`, `skip_judge`, and `judge_*` configuration.
- Preserve unrelated existing worktree changes; edit only requirement-related lines.
- `.git` is read-only in this environment, so record intended commit boundaries but do not claim commits were created.

---

### Task 1: Make adapter selection an explicit runtime contract

**Files:**
- Modify: `tests/test_retraining_launchers.py`
- Modify: `tests/test_evaluation.py`
- Modify: `scripts/run_train_grpo.sh`
- Modify: `scripts/run_grpo_vllm_server.sh`
- Modify: `scripts/eval_vllm_server.sh`
- Modify: `config/train_grpo.yml`
- Modify: `config/train_grpo_stratified_v2.yml`
- Modify: `config/eval_vllm_server.yml`
- Modify: `src/rl_training/config.py`
- Modify: `tests/test_rl_training.py`

**Interfaces:**
- Consumes: `SFT_ADAPTER_PATH: str` and `ADAPTER_PATH: str` environment variables.
- Produces: validated `--sft-adapter-path PATH`, `--lora-adapter-path PATH`, and `--adapter-path PATH` arguments.

- [ ] **Step 1: Write failing launcher tests**

Add helpers that create an adapter directory with both required files and run launchers in dry-run mode. Add tests equivalent to:

```python
def _make_adapter(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text("{}", encoding="utf-8")
    (path / "prompt_contract.json").write_text("{}", encoding="utf-8")
    return path


def test_grpo_launchers_require_manual_sft_adapter_path(tmp_path: Path) -> None:
    env = {"PATH": "/usr/bin:/bin", "MACORAG_LAUNCH_DRY_RUN": "1"}
    for script in ("run_train_grpo.sh", "run_grpo_vllm_server.sh"):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / script)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2
        assert "SFT_ADAPTER_PATH=/path/to/adapter" in result.stderr


def test_grpo_launchers_use_manual_sft_adapter_path(tmp_path: Path) -> None:
    adapter = _make_adapter(tmp_path / "chosen-sft")
    env = {
        "PATH": "/usr/bin:/bin",
        "MACORAG_LAUNCH_DRY_RUN": "1",
        "SFT_ADAPTER_PATH": str(adapter),
    }
    for script in ("run_train_grpo.sh", "run_grpo_vllm_server.sh"):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / script)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        assert str(adapter) in result.stdout


def test_eval_vllm_launcher_requires_and_uses_manual_adapter_path(tmp_path: Path) -> None:
    missing = subprocess.run(
        ["bash", str(ROOT / "scripts/eval_vllm_server.sh")],
        cwd=ROOT,
        env={"PATH": "/usr/bin:/bin", "MACORAG_LAUNCH_DRY_RUN": "1"},
        text=True,
        capture_output=True,
    )
    assert missing.returncode == 2
    assert "ADAPTER_PATH=/path/to/adapter" in missing.stderr

    adapter = _make_adapter(tmp_path / "chosen-eval")
    selected = subprocess.run(
        ["bash", str(ROOT / "scripts/eval_vllm_server.sh")],
        cwd=ROOT,
        env={
            "PATH": "/usr/bin:/bin",
            "MACORAG_LAUNCH_DRY_RUN": "1",
            "ADAPTER_PATH": str(adapter),
        },
        text=True,
        capture_output=True,
        check=True,
    )
    assert str(adapter) in selected.stdout
```

Update the shared config assertion so `sft_adapter_path` and `adapter_path` are absent, and assert launcher source contains neither `AUTO_FROM_` nor `glob(`.
Add a parser regression test asserting `parse_args([]).sft_adapter_path == ""`
so direct Python invocation also has no historical fixed adapter fallback.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
pytest -q tests/test_retraining_launchers.py tests/test_evaluation.py -k 'manual or retraining_configs_freeze_shared_contract'
```

Expected: failures because the launchers still auto-discover paths and the YAML files still persist auto markers.

- [ ] **Step 3: Remove automatic path selection and validate runtime input**

In each launcher, validate before the dry-run exit:

```bash
if [[ -z "${SFT_ADAPTER_PATH:-}" ]]; then
  printf 'SFT_ADAPTER_PATH is required. Example: SFT_ADAPTER_PATH=/path/to/adapter bash %s\n' "$0" >&2
  exit 2
fi
for required_file in adapter_config.json prompt_contract.json; do
  if [[ ! -f "${SFT_ADAPTER_PATH}/${required_file}" ]]; then
    printf 'Invalid SFT_ADAPTER_PATH=%s: missing %s\n' "${SFT_ADAPTER_PATH}" "${required_file}" >&2
    exit 2
  fi
done
```

Use the same structure with `ADAPTER_PATH` in `eval_vllm_server.sh`. Delete all embedded Python directory scans. Override the GRPO vLLM value with `YAML_LORA_ADAPTER_PATH="${SFT_ADAPTER_PATH}"` when sync mode is `lora`. Remove `sft_adapter_path` from both GRPO YAML files and `adapter_path` from the evaluation-server YAML. Change `DEFAULT_ARG_VALUES["sft_adapter_path"]` in `src/rl_training/config.py` from the historical run path to an empty string; launchers provide the required value explicitly.

- [ ] **Step 4: Run focused tests and Bash syntax checks**

Run:

```bash
pytest -q tests/test_retraining_launchers.py tests/test_evaluation.py tests/test_rl_training.py -k 'manual or retraining_configs_freeze_shared_contract or default_sft_adapter_path'
bash -n scripts/run_train_grpo.sh scripts/run_grpo_vllm_server.sh scripts/eval_vllm_server.sh
```

Expected: selected tests pass and Bash exits 0.

- [ ] **Step 5: Intended commit boundary**

When Git metadata is writable:

```bash
git add tests/test_retraining_launchers.py tests/test_evaluation.py tests/test_rl_training.py scripts/run_train_grpo.sh scripts/run_grpo_vllm_server.sh scripts/eval_vllm_server.sh config/train_grpo.yml config/train_grpo_stratified_v2.yml config/eval_vllm_server.yml src/rl_training/config.py
git commit -m "feat: require explicit adapter paths"
```

---

### Task 2: Replace LLM judging with unconditional local metric aggregation

**Files:**
- Create: `src/evaluation/local_evaluator.py`
- Modify: `tests/test_evaluation.py`
- Modify: `src/evaluation/evaluate_rag_model.py`
- Remove: `src/evaluation/bailian_evaluator.py`

**Interfaces:**
- Consumes: `evaluate_predictions(predictions_path: str | Path) -> dict[str, Any]`.
- Produces: `evaluation_results.json` with `contain_accuracy`, `exact_match`, `f1`, and `num_samples` only.

- [ ] **Step 1: Write failing local-evaluator tests**

Replace judge-specific test fixtures and tests with:

```python
from evaluation.local_evaluator import evaluate_predictions


def test_evaluate_predictions_writes_only_local_metrics(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    original = "\n".join([
        json.dumps({"pred_answer": "David Arquette", "gold_answer": "David Arquette"}),
        json.dumps({"pred_answer": "wrong", "gold_answer": "Right"}),
        "",
    ])
    predictions_path.write_text(original, encoding="utf-8")

    summary = evaluate_predictions(predictions_path)

    assert summary == {
        "contain_accuracy": 0.5,
        "exact_match": 0.5,
        "f1": 0.5,
        "num_samples": 2,
    }
    assert "llm_accuracy" not in summary
    assert predictions_path.read_text(encoding="utf-8") == original
    assert json.loads((tmp_path / "evaluation_results.json").read_text(encoding="utf-8")) == summary


def test_evaluate_predictions_preserves_falsy_answers(tmp_path: Path) -> None:
    predictions_path = tmp_path / "predictions.jsonl"
    predictions_path.write_text(
        json.dumps({"pred_answer": 0, "gold_answer": 0}) + "\n"
        + json.dumps({"pred_answer": False, "gold_answer": False}) + "\n",
        encoding="utf-8",
    )
    assert evaluate_predictions(predictions_path)["exact_match"] == 1.0
```

Delete tests for `BailianJudgeClient`, retries, judge metadata, and `calculate_llm_accuracy`.

- [ ] **Step 2: Run local evaluator tests and verify RED**

Run:

```bash
pytest -q tests/test_evaluation.py -k 'evaluate_predictions'
```

Expected: collection or assertion failure because `evaluation.local_evaluator` does not exist and the current scorer requires a judge client.

- [ ] **Step 3: Implement the minimal deterministic scorer**

Create `src/evaluation/local_evaluator.py` using the shared answer metric functions:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from answer_metrics import calculate_contain, calculate_exact_match, calculate_f1


def _coerce_answer(value: Any) -> str:
    return "" if value is None else str(value)


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid predictions format at {path}: expected a JSON list.")
    return payload


def evaluate_predictions(predictions_path: str | Path) -> dict[str, Any]:
    path = Path(predictions_path)
    predictions = _load_predictions(path)
    count = len(predictions)
    totals = {"contain_accuracy": 0.0, "exact_match": 0.0, "f1": 0.0}
    for prediction in predictions:
        pred = _coerce_answer(prediction.get("pred_answer"))
        gold = _coerce_answer(prediction.get("gold_answer"))
        totals["contain_accuracy"] += calculate_contain(pred, gold)
        totals["exact_match"] += calculate_exact_match(pred, gold)
        totals["f1"] += calculate_f1(pred, gold)
    summary = {
        key: (value / count if count else 0.0)
        for key, value in totals.items()
    }
    summary["num_samples"] = count
    (path.parent / "evaluation_results.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
```

Keep the existing JSON-object validation from the old loader when transferring it. Remove `src/evaluation/bailian_evaluator.py` only after no import references remain.

- [ ] **Step 4: Wire local metrics into every dataset evaluation**

Change the main evaluator import to:

```python
from .local_evaluator import evaluate_predictions
```

Delete judge-client construction and metadata. After `run_predictions(...)` and protocol metric output, call unconditionally:

```python
evaluate_predictions(dataset_dir / "predictions.jsonl")
```

Update the main-flow test to capture one local evaluation call per dataset and assert there is no client or judge metadata argument.

- [ ] **Step 5: Run focused evaluation tests**

Run:

```bash
pytest -q tests/test_evaluation.py -k 'evaluate_predictions or main_writes or main_passes'
```

Expected: all selected tests pass with no judge API setup.

- [ ] **Step 6: Intended commit boundary**

When Git metadata is writable:

```bash
git add src/evaluation/local_evaluator.py src/evaluation/evaluate_rag_model.py src/evaluation/bailian_evaluator.py tests/test_evaluation.py
git commit -m "feat: use local evaluation metrics only"
```

---

### Task 3: Remove obsolete judge configuration and verify the complete contract

**Files:**
- Modify: `tests/test_evaluation.py`
- Modify: `src/evaluation/config.py`
- Modify: `config/eval_macorag.yml`
- Modify: `config/eval_macorag_stratified_v2.yml`

**Interfaces:**
- Consumes: evaluation YAML containing only local inference, retrieval, and vLLM-client fields.
- Produces: argparse namespace with no `skip_judge` or `judge_*` attributes.

- [ ] **Step 1: Write failing configuration-removal tests**

Add assertions:

```python
def test_eval_config_has_no_external_judge_fields() -> None:
    args = parse_args(["--config", "config/eval_macorag.yml"])
    text = Path("config/eval_macorag.yml").read_text(encoding="utf-8")
    assert "skip_judge" not in text
    assert "judge_" not in text
    assert not any(name == "skip_judge" or name.startswith("judge_") for name in vars(args))


def test_obsolete_judge_yaml_fields_are_rejected(tmp_path: Path) -> None:
    config = tmp_path / "stale.yml"
    config.write_text("judge_model: qwen-plus\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="Unknown evaluation config keys.*judge_model"):
        parse_args(["--config", str(config)])
```

- [ ] **Step 2: Run configuration tests and verify RED**

Run:

```bash
pytest -q tests/test_evaluation.py -k 'external_judge_fields or obsolete_judge_yaml_fields'
```

Expected: failures because judge defaults, parser options, and YAML fields still exist.

- [ ] **Step 3: Delete judge defaults, CLI options, and YAML keys**

Remove `skip_judge` and every `judge_*` entry from `DEFAULT_ARG_VALUES` and `_build_parser()` in `src/evaluation/config.py`. Remove the corresponding blocks from both evaluation YAML files. Preserve strict unknown-key validation so old configs fail clearly.

- [ ] **Step 4: Run complete relevant verification**

Run:

```bash
pytest -q tests/test_evaluation.py tests/test_retraining_launchers.py
bash -n scripts/run_train_grpo.sh scripts/run_grpo_vllm_server.sh scripts/eval_vllm_server.sh scripts/eval_macorag.sh
python -m compileall -q src/evaluation
rg -n 'AUTO_FROM_SFT_V2|AUTO_FROM_GRPO_V2|BailianJudgeClient|calculate_llm_accuracy|llm_accuracy|skip_judge|judge_' config scripts src/evaluation tests/test_evaluation.py tests/test_retraining_launchers.py
```

Expected: pytest passes, Bash and compileall exit 0, and the final `rg` returns exit 1 with no matches.

- [ ] **Step 5: Verify diffs preserve unrelated work**

Run:

```bash
git diff --check
git diff -- config/train_grpo.yml config/train_grpo_stratified_v2.yml config/eval_vllm_server.yml config/eval_macorag.yml config/eval_macorag_stratified_v2.yml scripts/run_train_grpo.sh scripts/run_grpo_vllm_server.sh scripts/eval_vllm_server.sh src/evaluation/config.py src/evaluation/evaluate_rag_model.py src/evaluation/local_evaluator.py src/evaluation/bailian_evaluator.py tests/test_evaluation.py tests/test_retraining_launchers.py
```

Expected: no whitespace errors; diff contains only manual adapter selection and local-metric removal alongside pre-existing relevant changes.

- [ ] **Step 6: Intended commit boundary**

When Git metadata is writable:

```bash
git add src/evaluation/config.py config/eval_macorag.yml config/eval_macorag_stratified_v2.yml tests/test_evaluation.py
git commit -m "refactor: remove external evaluation judge"
```
