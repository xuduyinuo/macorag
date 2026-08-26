# Proportional RL Sample Limits Implementation Plan

> **For the AI agent worker:** Required sub-skill: use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task by task. Track every step with the checkboxes below.

**Goal:** Make opt-in RL `max_samples` limits select deterministic, nested, per-dataset proportional question-stratum prefixes instead of the first rows.

**Architecture:** Add focused stratum and proportional-prefix primitives to `rl_training.data`, then integrate them behind a backward-compatible loader strategy. Pass an independent sampling seed through GRPO config. Enable the strategy only in the validated stratified-v2 config and verify exact 500/1,000 real-data quotas; the existing dataset fingerprint continues to protect checkpoint/data identity.

**Tech stack:** Python 3.9, dataclasses, hashlib, `random.Random`, argparse, PyYAML, pytest.

---

## File structure

- Create `tests/test_rl_data_sampling.py`: proportional-prefix and loader tests.
- Modify `src/rl_training/data.py`: strata, stable prefixes, and loader integration.
- Modify `src/rl_training/config.py`: strategy and independent seed arguments.
- Modify `src/rl_training/train_grpo_macorag.py`: loader wiring.
- Modify `tests/test_rl_training.py`: YAML/CLI and checked-in config tests.
- Modify `config/train_grpo_stratified_v2.yml`: opt-in settings.

### Task 1: Proportional-prefix primitives

**Files:**
- Create: `tests/test_rl_data_sampling.py`
- Modify: `src/rl_training/data.py`

- [ ] **Step 1: Write failing exact-quota and nesting tests**

Create `tests/test_rl_data_sampling.py` with helpers that construct samples:

```python
from collections import Counter

from rl_training.data import RLSample, select_proportional_prefix


def make_samples(dataset: str, counts: dict[str, int]) -> list[RLSample]:
    return [
        RLSample(
            qid=f"{dataset}:{stratum}:{index}",
            dataset=dataset,
            question=f"question {dataset} {stratum} {index}",
            answer="answer",
            answer_aliases=[],
            supporting_facts=[],
            context_doc_ids=[],
            metadata={},
            sampling_stratum=stratum,
        )
        for stratum, count in counts.items()
        for index in range(count)
    ]


def stratum_counts(samples: list[RLSample]) -> dict[str, int]:
    return dict(Counter(sample.sampling_stratum for sample in samples))
```

Parameterize the three accepted source distributions and assert these results:

```python
EXPECTED = {
    "2wiki": {
        "source": {"compositional": 831, "comparison": 486, "bridge_comparison": 440, "inference": 243},
        500: {"compositional": 208, "comparison": 121, "bridge_comparison": 110, "inference": 61},
        1000: {"compositional": 416, "comparison": 243, "bridge_comparison": 220, "inference": 121},
    },
    "hotpotqa": {
        "source": {"hard/bridge": 1596, "hard/comparison": 404},
        500: {"hard/bridge": 399, "hard/comparison": 101},
        1000: {"hard/bridge": 798, "hard/comparison": 202},
    },
    "musique": {
        "source": {"2hop": 1036, "3hop1": 470, "3hop2": 159, "4hop1": 203, "4hop2": 53, "4hop3": 79},
        500: {"2hop": 259, "3hop1": 117, "3hop2": 40, "4hop1": 51, "4hop2": 13, "4hop3": 20},
        1000: {"2hop": 518, "3hop1": 235, "3hop2": 80, "4hop1": 101, "4hop2": 27, "4hop3": 39},
    },
}


def test_proportional_prefix_matches_v2_quotas_and_is_nested() -> None:
    for dataset, contract in EXPECTED.items():
        source = make_samples(dataset, contract["source"])
        selected_500 = select_proportional_prefix(source, max_samples=500, seed=20260826)
        selected_1000 = select_proportional_prefix(source, max_samples=1000, seed=20260826)
        assert stratum_counts(selected_500) == contract[500]
        assert stratum_counts(selected_1000) == contract[1000]
        assert {item.qid for item in selected_500} <= {item.qid for item in selected_1000}
```

Also test identical qid order for repeated calls, changed membership but unchanged
counts for a changed sampling seed, and full membership for `None` or a limit
at least the source size.

- [ ] **Step 2: Run the test and verify RED**

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_data_sampling.py::test_proportional_prefix_matches_v2_quotas_and_is_nested
```

Expected: collection fails because the new API and dataclass field do not exist.

- [ ] **Step 3: Implement stable strata and seed derivation**

In `src/rl_training/data.py`, import `hashlib`, append
`sampling_stratum: str = ""` to `RLSample`, and add:

```python
STRATA_BY_DATASET = {
    "2wiki": ("compositional", "comparison", "bridge_comparison", "inference"),
    "hotpotqa": ("hard/bridge", "hard/comparison"),
    "musique": ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3"),
}


def _derive_sampling_seed(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
```

- [ ] **Step 4: Implement the minimal proportional scheduler**

```python
def select_proportional_prefix(
    samples: list[RLSample], *, max_samples: int | None, seed: int
) -> list[RLSample]:
    if max_samples is not None and max_samples < 0:
        raise ValueError("max_samples must be non-negative or None.")
    if not samples:
        return []
    dataset = samples[0].dataset
    if any(sample.dataset != dataset for sample in samples):
        raise ValueError("Proportional prefix requires exactly one dataset.")
    canonical = STRATA_BY_DATASET.get(dataset)
    if canonical is None:
        raise ValueError(f"Unsupported proportional sampling dataset: {dataset}")
    buckets = {stratum: [] for stratum in canonical}
    for sample in samples:
        if sample.sampling_stratum not in buckets:
            raise ValueError(f"Unknown sampling stratum for {dataset}: {sample.sampling_stratum!r}")
        buckets[sample.sampling_stratum].append(sample)
    source_counts = {stratum: len(bucket) for stratum, bucket in buckets.items()}
    for stratum, bucket in buckets.items():
        random.Random(_derive_sampling_seed(seed, dataset, stratum)).shuffle(bucket)
    total = len(samples)
    selected_counts = {stratum: 0 for stratum in canonical}
    schedule = []
    for position in range(1, total + 1):
        available = [stratum for stratum in canonical if buckets[stratum]]
        chosen = max(
            available,
            key=lambda stratum: (
                position * source_counts[stratum] - selected_counts[stratum] * total,
                -canonical.index(stratum),
            ),
        )
        schedule.append(buckets[chosen].pop())
        selected_counts[chosen] += 1
    limit = total if max_samples is None else min(max_samples, total)
    return schedule[:limit]
```

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_data_sampling.py
git add src/rl_training/data.py tests/test_rl_data_sampling.py
git commit -m "feat: add proportional RL sample prefixes"
```

Expected: all Task 1 tests pass before committing.

### Task 2: Loader strategy, stratum derivation, and summary

**Files:**
- Modify: `src/rl_training/data.py`
- Modify: `tests/test_rl_data_sampling.py`
- Regression: `tests/test_rl_training.py`

- [ ] **Step 1: Write failing loader tests**

Create v2-shaped temporary JSONL files and call:

```python
samples_500, summary_500 = load_rl_samples(
    data_root=data_root,
    max_samples=500,
    data_sampling_strategy="proportional_stratified",
    data_sampling_seed=20260826,
)
samples_1000, summary_1000 = load_rl_samples(
    data_root=data_root,
    max_samples=1000,
    data_sampling_strategy="proportional_stratified",
    data_sampling_seed=20260826,
)
assert {item.qid for item in samples_500} <= {item.qid for item in samples_1000}
assert summary_500["counts_by_dataset_and_stratum"]["hotpotqa"] == {
    "hard/bridge": 399,
    "hard/comparison": 101,
}
```

Add separate tests proving:

- `head` still returns the first row per dataset for `max_samples=1`;
- 2Wiki derives top-level `question_type`;
- HotpotQA derives `metadata.level/question_type`;
- MuSiQue derives the qid prefix before `__`;
- missing fields, unsupported datasets, unknown strategies, and non-integer
  sampling seeds fail explicitly;
- the summary reports strategy, seed, dataset counts, and stratum counts.

- [ ] **Step 2: Run loader tests and verify RED**

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_data_sampling.py -k 'loader or head or missing or unsupported'
```

Expected: failures because the loader does not accept the new options.

- [ ] **Step 3: Implement strict row-to-stratum derivation**

Add to `src/rl_training/data.py`:

```python
def _sampling_stratum(dataset: str, row: dict[str, Any]) -> str:
    if dataset == "2wiki":
        value = str(row.get("question_type") or "").strip()
    elif dataset == "hotpotqa":
        level = str((row.get("metadata") or {}).get("level") or "").strip()
        kind = str(row.get("question_type") or "").strip()
        value = f"{level}/{kind}" if level and kind else ""
    elif dataset == "musique":
        qid = str(row.get("qid") or "")
        value = qid.split("__", 1)[0] if "__" in qid else ""
    else:
        raise ValueError(f"Unsupported proportional sampling dataset: {dataset}")
    if not value:
        raise ValueError(f"Missing sampling stratum for {dataset}/{row.get('qid', '')}")
    if value not in STRATA_BY_DATASET[dataset]:
        raise ValueError(f"Unknown sampling stratum for {dataset}: {value!r}")
    return value
```

Attach the result to `RLSample.sampling_stratum` only under the proportional
strategy. Do not require new fields under `head`.

- [ ] **Step 4: Integrate the opt-in loader strategy**

Change the loader signature to:

```python
def load_rl_samples(
    *,
    data_root: str | Path,
    data_files: list[str] | tuple[str, ...] | None = None,
    max_samples: int | None = None,
    data_sampling_strategy: str = "head",
    data_sampling_seed: int = 20260826,
) -> tuple[list[RLSample], dict[str, Any]]:
```

Validate strategy and seed before reading. Preserve the current streaming
first-N logic for `head`. For `proportional_stratified`, read every valid row,
group by dataset, call `select_proportional_prefix` in sorted dataset order,
and concatenate the selected prefixes. Add:

```python
summary["data_sampling_strategy"] = data_sampling_strategy
summary["data_sampling_seed"] = data_sampling_seed
summary["counts_by_dataset_and_stratum"] = {
    dataset: dict(Counter(item.sampling_stratum for item in selected))
    for dataset, selected in selected_by_dataset.items()
}
```

For `head`, set `counts_by_dataset_and_stratum` to `{}` so legacy datasets do
not need type fields.

- [ ] **Step 5: Run loader regressions and commit**

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_data_sampling.py
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py -k 'load_rl_samples'
git add src/rl_training/data.py tests/test_rl_data_sampling.py
git commit -m "feat: stratify RL loader sample limits"
```

Expected: new tests and existing head-limit tests pass.

### Task 3: GRPO config and main wiring

**Files:**
- Modify: `src/rl_training/config.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Modify: `config/train_grpo_stratified_v2.yml`
- Modify: `tests/test_rl_training.py`

- [ ] **Step 1: Write failing configuration tests**

Extend `test_parse_args_loads_train_grpo_yaml` with YAML lines and assertions:

```python
"data_sampling_strategy: proportional_stratified",
"data_sampling_seed: 20260826",
assert args.data_sampling_strategy == "proportional_stratified"
assert args.data_sampling_seed == 20260826
```

Add explicit CLI and checked-in config tests:

```python
def test_parse_args_overrides_data_sampling_yaml(tmp_path: Path) -> None:
    config = tmp_path / "train.yml"
    config.write_text(
        "data_sampling_strategy: head\ndata_sampling_seed: 7\n",
        encoding="utf-8",
    )
    args = parse_args([
        "--config", str(config),
        "--data-sampling-strategy", "proportional_stratified",
        "--data-sampling-seed", "20260826",
    ])
    assert args.data_sampling_strategy == "proportional_stratified"
    assert args.data_sampling_seed == 20260826


def test_stratified_v2_config_enables_proportional_sampling_only_opt_in() -> None:
    import yaml
    v2 = yaml.safe_load(Path("config/train_grpo_stratified_v2.yml").read_text())
    default = yaml.safe_load(Path("config/train_grpo.yml").read_text())
    assert v2["data_sampling_strategy"] == "proportional_stratified"
    assert v2["data_sampling_seed"] == 20260826
    assert "data_sampling_strategy" not in default
    assert "data_sampling_seed" not in default
```

- [ ] **Step 2: Run tests and verify RED**

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py::test_parse_args_loads_train_grpo_yaml
```

Expected: failures because the config keys and CLI arguments do not exist.

- [ ] **Step 3: Add defaults and CLI arguments**

Add to `ROLLOUT_DEFAULTS`:

```python
"data_sampling_strategy": "head",
"data_sampling_seed": 20260826,
```

Add to the rollout parser:

```python
rollout.add_argument(
    "--data-sampling-strategy",
    choices=("head", "proportional_stratified"),
    default=defaults["data_sampling_strategy"],
)
rollout.add_argument(
    "--data-sampling-seed", type=int, default=defaults["data_sampling_seed"]
)
```

- [ ] **Step 4: Wire loading while preserving checkpoint compatibility**

Pass both arguments in `train_grpo_macorag.main`:

```python
samples, data_summary = load_rl_samples(
    data_root=args.rl_data_root,
    data_files=list(args.rl_data_files or []),
    max_samples=args.max_samples,
    data_sampling_strategy=args.data_sampling_strategy,
    data_sampling_seed=args.data_sampling_seed,
)
```

Do not change `_CRITICAL_CONFIG_FIELDS`. `fingerprint_dataset(samples)` already
includes final sample qids and order, so a changed selected set fails resume
identity without invalidating older checkpoints merely because new config keys
exist.

- [ ] **Step 5: Enable only the v2 config**

After `max_samples` in `config/train_grpo_stratified_v2.yml`, add:

```yaml
data_sampling_strategy: "proportional_stratified"
data_sampling_seed: 20260826
```

Do not modify `config/train_grpo.yml`.

- [ ] **Step 6: Run affected tests and commit**

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_data_sampling.py tests/test_rl_training.py -k 'parse_args or load_rl_samples or stratified_v2'
git add src/rl_training/config.py src/rl_training/train_grpo_macorag.py config/train_grpo_stratified_v2.yml tests/test_rl_training.py
git commit -m "config: enable proportional v2 RL sampling"
```

Expected: all selected tests pass before committing.

### Task 4: Real-data acceptance and final regression

**Files:**
- No required code changes.
- Modify code only after adding a failing regression test if canonical data exposes a defect.

- [ ] **Step 1: Verify real 500/1,000 quotas and nesting**

Run:

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python - <<'PY'
from rl_training.data import load_rl_samples

root = "data/rl_train_2000_stratified_v2"
expected = {
    500: {
        "2wiki": {"compositional": 208, "comparison": 121, "bridge_comparison": 110, "inference": 61},
        "hotpotqa": {"hard/bridge": 399, "hard/comparison": 101},
        "musique": {"2hop": 259, "3hop1": 117, "3hop2": 40, "4hop1": 51, "4hop2": 13, "4hop3": 20},
    },
    1000: {
        "2wiki": {"compositional": 416, "comparison": 243, "bridge_comparison": 220, "inference": 121},
        "hotpotqa": {"hard/bridge": 798, "hard/comparison": 202},
        "musique": {"2hop": 518, "3hop1": 235, "3hop2": 80, "4hop1": 101, "4hop2": 27, "4hop3": 39},
    },
}
loaded = {}
for limit in (500, 1000):
    samples, summary = load_rl_samples(
        data_root=root,
        max_samples=limit,
        data_sampling_strategy="proportional_stratified",
        data_sampling_seed=20260826,
    )
    assert len(samples) == 3 * limit
    assert summary["counts_by_dataset_and_stratum"] == expected[limit]
    loaded[limit] = {sample.qid for sample in samples}
assert loaded[500] <= loaded[1000]
print("real-v2 quotas and nesting verified")
PY
```

Expected: prints `real-v2 quotas and nesting verified`.

- [ ] **Step 2: Verify training-seed independence**

Run:

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python - <<'PY'
from rl_training.config import parse_args
from rl_training.data import load_rl_samples

config = "config/train_grpo_stratified_v2.yml"
args_42 = parse_args(["--config", config, "--seed", "42", "--max-samples", "500"])
args_43 = parse_args(["--config", config, "--seed", "43", "--max-samples", "500"])
assert args_42.seed != args_43.seed
assert args_42.data_sampling_seed == args_43.data_sampling_seed == 20260826

def selected_qids(args):
    samples, _ = load_rl_samples(
        data_root=args.rl_data_root,
        data_files=list(args.rl_data_files or []),
        max_samples=args.max_samples,
        data_sampling_strategy=args.data_sampling_strategy,
        data_sampling_seed=args.data_sampling_seed,
    )
    return [sample.qid for sample in samples]

assert selected_qids(args_42) == selected_qids(args_43)
print("training seed independence verified")
PY
```

Expected: prints `training seed independence verified`.

- [ ] **Step 3: Run the complete affected suite**

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_data_sampling.py tests/test_rl_training.py tests/test_rl_checkpointing.py tests/test_rl_checkpoint_seeds.py tests/test_retraining_launchers.py
```

Expected: zero failures. If an unrelated optional dependency prevents
collection, record the exact dependency error and run every runnable affected
test explicitly; do not report the complete suite as passing.

- [ ] **Step 4: Check formatting and commit boundary**

```bash
git diff --check
git status --short
git log --oneline --max-count=8
```

Expected: no whitespace errors; only planned files appear in the new commits;
pre-existing unrelated changes remain untouched.

- [ ] **Step 5: Record operational commands**

For the planned experiments, use:

```bash
bash scripts/run_train_grpo.sh --config config/train_grpo_stratified_v2.yml --max-samples 500
bash scripts/run_train_grpo.sh --config config/train_grpo_stratified_v2.yml --max-samples 1000
```

Keep `max_total_samples` unset. Both commands use the same fixed data sampling
seed; only the nested proportional prefix length changes.
