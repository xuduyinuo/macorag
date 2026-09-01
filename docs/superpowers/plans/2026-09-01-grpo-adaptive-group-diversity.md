# GRPO Adaptive Group Diversity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add bounded adaptive rollout-group expansion and conservative singleton terminal-advantage fallback so tied `group_size=4` samples produce useful GRPO updates without permanently doubling every sample's cost.

**Architecture:** Keep four candidates as the initial group, score them with the existing task rewards, and generate up to four retry candidates only when every trainable action advantage is zero. Recompute credit over all original and retry candidates, retain role-round normalization as the primary signal, and use a weighted normalized action-terminal fallback only for singleton role-round buckets.

**Tech Stack:** Python 3.10+, PyTorch, argparse/YAML configuration, vLLM LoRA generation, pytest, JSONL diagnostics, repository-native atomic checkpoints.

## Global Constraints

- Preserve the existing query, evidence, answer, aggregate, and terminal reward formulas in `src/rl_training/rewards.py`.
- Preserve canonical `group_size: 4`; default adaptive expansion is one retry of four candidates with `max_group_size: 8`.
- Never select candidates by observed reward and never discard the original four candidates.
- Retry generation uses new generation-counter seeds and `temperature: 1.0`; all other generation and retrieval settings remain unchanged.
- Multi-member `(role, round_index)` buckets retain the current normalized decision-return formula without rollout-level blending.
- Singleton fallback uses action `terminal_reward`, never legacy aggregate `reward_total`, with default weight `0.2`.
- Disabled adaptive expansion and fallback weight `0.0` preserve the existing behavior.
- All new sampling and fallback fields are critical checkpoint fields; bump the checkpoint schema so legacy checkpoints fail clearly instead of silently changing semantics.
- Preserve current unrelated working-tree edits in `config/eval_vllm_server.yml`, `config/train_grpo.yml`, `src/rl_training/train_grpo_macorag.py`, and `tests/test_rl_training.py`. Inspect the diff before every edit and do not revert or overwrite existing hunks.
- Because implementation paths already contain uncommitted user changes, do not create implementation commits that would bundle those changes. Keep a scoped diff and report this exception; commit only files proven clean at execution time.

---

## File Map

- Modify `src/rl_training/config.py`: define adaptive-expansion defaults, CLI fields, and validation.
- Modify `config/train_grpo.yml`: enable the approved canonical four-to-eight policy.
- Modify `src/rl_training/checkpointing.py`: add critical fields and increment checkpoint schema.
- Modify `src/rl_training/policy.py`: retain primary and fallback advantage components on each generated action.
- Modify `src/rl_training/trainer.py`: assign singleton terminal fallback while preserving normal role-round credit.
- Modify `src/rl_training/train_grpo_macorag.py`: separate candidate generation from rescoring, orchestrate bounded expansion, restore retry temperature, and emit diagnostics.
- Modify `tests/test_rl_training.py`: unit and integration coverage for every new contract.

### Task 1: Configuration and checkpoint semantic identity

**Files:**
- Modify: `src/rl_training/config.py:24-75, 200-285, 372-402`
- Modify: `config/train_grpo.yml:12-28, 45-51`
- Modify: `src/rl_training/checkpointing.py:12-65`
- Test: `tests/test_rl_training.py:214-455, 1180-1245`

**Interfaces:**
- Produces argparse fields `adaptive_group_expansion: bool`, `max_group_size: int`, `group_expansion_step: int`, `max_group_expansion_attempts: int`, `diversity_retry_temperature: float`, and `singleton_terminal_fallback_weight: float`.
- Produces checkpoint schema version `3` and fingerprints all six fields.
- Later tasks consume these exact `args` attributes.

- [ ] **Step 1: Add failing configuration parsing and validation tests**

Add tests with the exact accepted values and invalid boundaries:

```python
def test_parse_args_loads_adaptive_group_diversity_config(tmp_path: Path) -> None:
    config = tmp_path / "train.yml"
    config.write_text(
        "\n".join(
            [
                "adaptive_group_expansion: true",
                "group_size: 4",
                "max_group_size: 8",
                "group_expansion_step: 4",
                "max_group_expansion_attempts: 1",
                "diversity_retry_temperature: 1.0",
                "singleton_terminal_fallback_weight: 0.2",
            ]
        ),
        encoding="utf-8",
    )
    args = parse_args(["--config", str(config)])
    assert args.adaptive_group_expansion is True
    assert args.max_group_size == 8
    assert args.group_expansion_step == 4
    assert args.max_group_expansion_attempts == 1
    assert args.diversity_retry_temperature == pytest.approx(1.0)
    assert args.singleton_terminal_fallback_weight == pytest.approx(0.2)


@pytest.mark.parametrize(
    "lines,error",
    [
        (["adaptive_group_expansion: true", "group_size: 4", "max_group_size: 4"], "max_group_size"),
        (["group_expansion_step: 0"], "group_expansion_step"),
        (["max_group_expansion_attempts: -1"], "max_group_expansion_attempts"),
        (["diversity_retry_temperature: 0.0"], "diversity_retry_temperature"),
        (["singleton_terminal_fallback_weight: 1.1"], "singleton_terminal_fallback_weight"),
    ],
)
def test_parse_args_rejects_invalid_adaptive_group_diversity(
    tmp_path: Path,
    lines: list[str],
    error: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "train.yml"
    config.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config)])
    assert error in capsys.readouterr().err
```

- [ ] **Step 2: Run the new parser tests and confirm red state**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'adaptive_group_diversity_config or invalid_adaptive_group_diversity'
```

Expected: FAIL because the new arguments are not recognized or absent.

- [ ] **Step 3: Add defaults, CLI flags, and exact validation**

Extend `ROLLOUT_DEFAULTS` and the rollout parser:

```python
ROLLOUT_DEFAULTS.update(
    {
        "adaptive_group_expansion": False,
        "max_group_size": 8,
        "group_expansion_step": 4,
        "max_group_expansion_attempts": 1,
        "diversity_retry_temperature": 1.0,
        "singleton_terminal_fallback_weight": 0.2,
    }
)

rollout.add_argument(
    "--adaptive-group-expansion",
    action=BooleanOptionalAction,
    default=defaults["adaptive_group_expansion"],
)
rollout.add_argument("--max-group-size", type=int, default=defaults["max_group_size"])
rollout.add_argument("--group-expansion-step", type=int, default=defaults["group_expansion_step"])
rollout.add_argument(
    "--max-group-expansion-attempts",
    type=int,
    default=defaults["max_group_expansion_attempts"],
)
rollout.add_argument(
    "--diversity-retry-temperature",
    type=float,
    default=defaults["diversity_retry_temperature"],
)
rollout.add_argument(
    "--singleton-terminal-fallback-weight",
    type=float,
    default=defaults["singleton_terminal_fallback_weight"],
)
```

Validate with `math.isfinite`, importing `math` at module scope:

```python
if args.max_group_size <= 0:
    parser.error("max_group_size must be positive")
if args.group_expansion_step <= 0:
    parser.error("group_expansion_step must be positive")
if args.max_group_expansion_attempts < 0:
    parser.error("max_group_expansion_attempts must be non-negative")
if not math.isfinite(args.diversity_retry_temperature) or args.diversity_retry_temperature <= 0.0:
    parser.error("diversity_retry_temperature must be a positive finite value")
if not 0.0 <= args.singleton_terminal_fallback_weight <= 1.0:
    parser.error("singleton_terminal_fallback_weight must satisfy 0 <= value <= 1")
if args.adaptive_group_expansion:
    if args.max_group_size <= args.group_size:
        parser.error("max_group_size must exceed group_size when adaptive expansion is enabled")
    if args.max_group_expansion_attempts < 1:
        parser.error("adaptive expansion requires at least one expansion attempt")
```

- [ ] **Step 4: Add checkpoint fingerprint and schema tests**

Create two otherwise identical namespaces and prove each field changes `fingerprint_config`. Assert `CHECKPOINT_SCHEMA_VERSION == 3`:

```python
def test_adaptive_group_fields_are_checkpoint_critical() -> None:
    from rl_training import checkpointing

    baseline = parse_args([])
    baseline_hash = checkpointing.fingerprint_config(baseline)
    for field, value in {
        "adaptive_group_expansion": not baseline.adaptive_group_expansion,
        "max_group_size": baseline.max_group_size + 1,
        "group_expansion_step": baseline.group_expansion_step + 1,
        "max_group_expansion_attempts": baseline.max_group_expansion_attempts + 1,
        "diversity_retry_temperature": baseline.diversity_retry_temperature + 0.1,
        "singleton_terminal_fallback_weight": 0.3,
    }.items():
        changed = Namespace(**vars(baseline))
        setattr(changed, field, value)
        assert checkpointing.fingerprint_config(changed) != baseline_hash
    assert checkpointing.CHECKPOINT_SCHEMA_VERSION == 3
```

- [ ] **Step 5: Implement checkpoint identity and canonical YAML**

Set `CHECKPOINT_SCHEMA_VERSION = 3`, append all six names to `_CRITICAL_CONFIG_FIELDS`, and add the approved YAML values immediately after `group_size: 4`:

```yaml
adaptive_group_expansion: true
max_group_size: 8
group_expansion_step: 4
max_group_expansion_attempts: 1
diversity_retry_temperature: 1.0
singleton_terminal_fallback_weight: 0.2
```

- [ ] **Step 6: Run focused tests and inspect the scoped diff**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'parse_args or checkpoint_critical or checkpoint_schema'
git diff --check -- src/rl_training/config.py src/rl_training/checkpointing.py config/train_grpo.yml tests/test_rl_training.py
```

Expected: focused tests PASS and `git diff --check` prints nothing. Do not commit overlapping dirty files; record the scoped diff for final handoff.

### Task 2: Primary and singleton fallback advantage accounting

**Files:**
- Modify: `src/rl_training/policy.py:20-38`
- Modify: `src/rl_training/trainer.py:67-128`
- Test: `tests/test_rl_training.py:3856-4005`

**Interfaces:**
- Produces `GeneratedAction.primary_advantage: float` and `GeneratedAction.fallback_advantage: float`.
- Extends `assign_action_advantages(..., singleton_terminal_fallback_weight: float = 0.0)` while preserving its return type.
- Task 3 consumes the final `action.advantage` and Task 4 consumes all three advantage fields.

- [ ] **Step 1: Write failing singleton and compatibility tests**

Construct three rollouts where only one reaches round 1. Assert normal round-0 advantages remain unchanged, while the round-1 singleton receives `0.2 * normalized_terminal_advantage`:

```python
class _FallbackAction:
    def __init__(self, round_index: int) -> None:
        self.role = AgentRole.QUERY_RETRIEVER
        self.round_index = round_index
        self.local_reward = 0.0
        self.terminal_reward = 0.0
        self.decision_return = 0.0
        self.advantage = 0.0


def _fallback_test_rollouts(terminal_rewards: list[float]) -> list[dict[str, object]]:
    rollouts: list[dict[str, object]] = []
    for index, terminal_reward in enumerate(terminal_rewards):
        actions = [_FallbackAction(round_index=0)]
        action_rewards = [
            {"role": "query_retriever", "round_index": 0, "local_reward": 0.0}
        ]
        if index == len(terminal_rewards) - 1:
            actions.append(_FallbackAction(round_index=1))
            action_rewards.append(
                {"role": "query_retriever", "round_index": 1, "local_reward": 0.0}
            )
        rollouts.append(
            {
                "actions": actions,
                "action_rewards": action_rewards,
                "terminal_reward": terminal_reward,
            }
        )
    return rollouts


def test_assign_action_advantages_uses_terminal_fallback_only_for_singleton() -> None:
    rollouts = _fallback_test_rollouts(terminal_rewards=[0.0, 1.0, 2.0])
    stats = trainer_module.assign_action_advantages(
        rollouts,
        global_weights={"query_retriever": 1.0},
        granularity="role_round",
        singleton_terminal_fallback_weight=0.2,
    )
    singleton = rollouts[2]["actions"][1]
    assert singleton.primary_advantage == 0.0
    assert singleton.fallback_advantage == pytest.approx(0.2 * (2.0 - 1.0) / (2 / 3) ** 0.5)
    assert singleton.advantage == pytest.approx(singleton.fallback_advantage)
    assert stats["query_retriever@round=1"]["count"] == 1
```

Also assert weight `0.0` keeps existing singleton advantage at zero and identical terminal rewards yield zero fallback.

- [ ] **Step 2: Run advantage tests and confirm red state**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'assign_action_advantages and (singleton or role_and_round)'
```

Expected: FAIL because the new keyword and fields do not exist.

- [ ] **Step 3: Add explicit advantage component fields**

Extend `GeneratedAction` without changing constructor callers:

```python
primary_advantage: float = 0.0
fallback_advantage: float = 0.0
```

- [ ] **Step 4: Implement conservative singleton fallback**

Store `(rollout, action)` pairs in each bucket, compute group terminal advantages once with `normalize_group_advantages`, and apply:

```python
terminal_advantages = normalize_group_advantages(
    [float(rollout.get("terminal_reward", 0.0)) for rollout in rollouts]
)
terminal_by_rollout = {
    id(rollout): terminal_advantage
    for rollout, terminal_advantage in zip(rollouts, terminal_advantages)
}

if len(entries) == 1:
    rollout, action = entries[0]
    action.primary_advantage = 0.0
    action.fallback_advantage = (
        float(singleton_terminal_fallback_weight) * terminal_by_rollout[id(rollout)]
    )
    action.advantage = action.fallback_advantage
else:
    for (_, action), primary in zip(entries, normalized_returns):
        action.primary_advantage = primary
        action.fallback_advantage = 0.0
        action.advantage = primary
```

Validate the fallback weight is finite and in `[0, 1]`. Expand bucket stats with `primary_advantage_*` and `fallback_advantage_*` fields while retaining existing `advantage_*` keys.

- [ ] **Step 5: Run complete credit-assignment coverage**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'assign_action_advantages or action_credit'
```

Expected: all existing and new credit tests PASS. Do not commit files with pre-existing edits.

### Task 3: Bounded adaptive candidate expansion

**Files:**
- Modify: `src/rl_training/train_grpo_macorag.py:590-720`
- Test: `tests/test_rl_training.py:3089-3170, 4357-4405`

**Interfaces:**
- Produces `_generate_rollout_candidates(..., candidate_count: int, group_index_offset: int) -> tuple[list[dict[str, Any]], dict[str, float]]`.
- Produces `_score_rollout_candidates(..., rollouts: list[dict[str, Any]]) -> dict[str, Any]` that replaces derived rewards/advantages on every call.
- Produces `_group_is_informative(rollouts: list[dict[str, Any]], tolerance: float = 1e-12) -> bool`.
- Keeps `_rollout_group(...) -> tuple[list[dict[str, Any]], dict[str, Any]]` as the training-loop entry point.

- [ ] **Step 1: Write failing expansion behavior tests**

Add deterministic tests for four required paths:

```python
def _adaptive_args(**overrides: object) -> Namespace:
    values: dict[str, object] = {
        "group_size": 4,
        "max_rounds": 2,
        "adaptive_group_expansion": True,
        "max_group_size": 8,
        "group_expansion_step": 4,
        "max_group_expansion_attempts": 1,
        "diversity_retry_temperature": 1.0,
        "singleton_terminal_fallback_weight": 0.2,
        "query_global_reward_weight": 1.0 / 3.0,
        "evidence_global_reward_weight": 3.0 / 7.0,
        "answer_global_reward_weight": 7.0 / 3.0,
        "advantage_epsilon": 1.0e-8,
        "advantage_granularity": "role_round",
    }
    values.update(overrides)
    return Namespace(**values)


def _temperature_policy(temperature: float = 0.8) -> types.SimpleNamespace:
    return types.SimpleNamespace(temperature=temperature)


def _install_candidate_batches(
    monkeypatch: pytest.MonkeyPatch,
    *,
    reward_batches: list[list[float]],
) -> list[tuple[int, int, float]]:
    remaining = iter(reward_batches)
    calls: list[tuple[int, int, float]] = []

    def fake_generate(
        *, policy, candidate_count: int, group_index_offset: int, **kwargs
    ):
        del kwargs
        rewards = next(remaining)
        assert len(rewards) == candidate_count
        calls.append((candidate_count, group_index_offset, float(policy.temperature)))
        rollouts = []
        for offset, reward in enumerate(rewards):
            action = types.SimpleNamespace(
                completion_ids=[1],
                advantage=0.0,
                primary_advantage=0.0,
                fallback_advantage=0.0,
            )
            rollouts.append(
                {
                    "group_index": group_index_offset + offset,
                    "actions": [action],
                    "_test_reward": float(reward),
                }
            )
        return rollouts, {
            "time_rollout_seconds": 0.0,
            "time_vllm_generate_seconds": 0.0,
            "time_behavior_rescore_seconds": 0.0,
            "time_reward_seconds": 0.0,
            "time_retrieval_seconds": 0.0,
            "retrieval_cache_hits": 0,
            "retrieval_cache_misses": 0,
        }

    def fake_score(*, rollouts, **kwargs):
        del kwargs
        advantages = trainer_module.normalize_group_advantages(
            [item["_test_reward"] for item in rollouts]
        )
        for rollout, advantage in zip(rollouts, advantages):
            action = rollout["actions"][0]
            action.primary_advantage = advantage
            action.fallback_advantage = 0.0
            action.advantage = advantage
            rollout["rewards"] = {"total": rollout["_test_reward"]}
            rollout["terminal_reward"] = rollout["_test_reward"]
            rollout["advantage"] = advantage
        return {"agent_credit_stats": {}}

    monkeypatch.setattr(train_grpo_module, "_generate_rollout_candidates", fake_generate)
    monkeypatch.setattr(train_grpo_module, "_score_rollout_candidates", fake_score)
    return calls


def test_rollout_group_does_not_expand_informative_initial_group(monkeypatch) -> None:
    calls = _install_candidate_batches(monkeypatch, reward_batches=[[0.0, 0.0, 1.0, 1.0]])
    rollouts, timing = _rollout_group(
        args=_adaptive_args(), sample=object(), policy=_temperature_policy(), retrieval_env=object()
    )
    assert calls == [(4, 0, 0.8)]
    assert len(rollouts) == 4
    assert timing["group_expansion_attempts"] == 0


def test_rollout_group_expands_tied_group_and_keeps_all_candidates(monkeypatch) -> None:
    calls = _install_candidate_batches(
        monkeypatch,
        reward_batches=[[1.0, 1.0, 1.0, 1.0], [0.0, 1.0, 2.0, 3.0]],
    )
    policy = _temperature_policy(temperature=0.8)
    rollouts, timing = _rollout_group(
        args=_adaptive_args(), sample=object(), policy=policy, retrieval_env=object()
    )
    assert calls == [(4, 0, 0.8), (4, 4, 1.0)]
    assert [item["group_index"] for item in rollouts] == list(range(8))
    assert policy.temperature == pytest.approx(0.8)
    assert timing["group_expansion_triggered"] is True
    assert timing["effective_group_size"] == 8
```

Add tests that a still-tied expanded group remains skippable, `max_group_size` truncates the retry count, disabled expansion makes exactly one call, and a retry exception restores the original temperature.

- [ ] **Step 2: Run expansion tests and confirm red state**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'rollout_group and (expand or informative or temperature or max_group)'
```

Expected: FAIL because `_rollout_group` still performs one fixed-size generation.

- [ ] **Step 3: Extract candidate generation without changing existing behavior**

Move the current batch/non-batch generation branches into `_generate_rollout_candidates`. Accept `candidate_count` instead of reading `args.group_size`, add `group_index_offset` to serialized indices, and return generation/retrieval timing without assigning rewards.

Use this interface and initialize every returned rollout with only raw execution fields:

```python
def _generate_rollout_candidates(
    *,
    args: Any,
    sample: RLSample,
    policy: HFSharedPolicy,
    retrieval_env: CachedLinearRAGRetrievalEnv,
    candidate_count: int,
    group_index_offset: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if candidate_count <= 0:
        raise ValueError("candidate_count must be positive")

    batch_results = run_batched_rollouts(
        question=sample.question,
        dataset=sample.dataset,
        group_size=candidate_count,
        max_rounds=args.max_rounds,
        policy=policy,
        retrieval_env=retrieval_env,
    )

    rollouts = [
        {
            "group_index": group_index_offset + index,
            "result": item.result,
            "trajectory": item.result.trajectory,
            "parse_errors": item.result.parse_errors,
            "final_answer": item.result.final_answer,
            "actions": list(item.trace.actions),
        }
        for index, item in enumerate(batch_results)
    ]
```

Retain the existing scalar-policy fallback with `range(candidate_count)` and the same offset. Keep retrieval cache/timing deltas local to this call.

- [ ] **Step 4: Extract idempotent scoring and diversity summaries**

Implement `_score_rollout_candidates` so each call overwrites `rewards`, `action_rewards`, `terminal_reward`, rollout `advantage`, and action advantage components. Call:

```python
agent_credit_stats = assign_action_advantages(
    rollouts,
    global_weights={
        "query_retriever": float(args.query_global_reward_weight),
        "evidence_updater": float(args.evidence_global_reward_weight),
        "answer_generator": float(args.answer_global_reward_weight),
    },
    epsilon=float(args.advantage_epsilon),
    granularity=str(args.advantage_granularity),
    singleton_terminal_fallback_weight=float(args.singleton_terminal_fallback_weight),
)
```

Before normalization, reject non-finite aggregate and terminal values:

```python
for rollout in rollouts:
    if not math.isfinite(float(rollout["rewards"]["total"])):
        raise RuntimeError("Rollout aggregate reward must be finite.")
    if not math.isfinite(float(rollout["terminal_reward"])):
        raise RuntimeError("Rollout terminal reward must be finite.")
```

Use completion-bearing actions only in `_group_is_informative`, matching `_train_on_rollouts`:

```python
return any(
    action.completion_ids and abs(float(action.advantage)) > tolerance
    for rollout in rollouts
    for action in rollout.get("actions", [])
)
```

- [ ] **Step 5: Implement finally-safe retry temperature**

Add a context manager local to the trainer module:

```python
@contextmanager
def _temporary_policy_temperature(policy: Any, temperature: float):
    previous = float(policy.temperature)
    policy.temperature = float(temperature)
    try:
        yield
    finally:
        policy.temperature = previous
```

Fail clearly if adaptive expansion is enabled for a policy without a mutable `temperature` attribute.

- [ ] **Step 6: Implement the bounded expansion loop**

Generate and score the initial group, preserve before-expansion summaries, then execute:

```python
attempts = 0
while (
    args.adaptive_group_expansion
    and not _group_is_informative(rollouts)
    and attempts < args.max_group_expansion_attempts
    and len(rollouts) < args.max_group_size
):
    retry_count = min(args.group_expansion_step, args.max_group_size - len(rollouts))
    with _temporary_policy_temperature(policy, args.diversity_retry_temperature):
        retry_rollouts, retry_timing = _generate_rollout_candidates(
            args=args,
            sample=sample,
            policy=policy,
            retrieval_env=retrieval_env,
            candidate_count=retry_count,
            group_index_offset=len(rollouts),
        )
    rollouts.extend(retry_rollouts)
    attempts += 1
    _score_rollout_candidates(args=args, sample=sample, rollouts=rollouts)
    _accumulate_rollout_timing(total_timing, retry_timing)
```

No conditional filtering of retry results is permitted.

Accumulate numeric timing explicitly and replace `agent_credit_stats` only with the final joint scoring result:

```python
def _accumulate_rollout_timing(total: dict[str, Any], update: dict[str, Any]) -> None:
    for key in (
        "time_rollout_seconds",
        "time_vllm_generate_seconds",
        "time_behavior_rescore_seconds",
        "time_reward_seconds",
        "time_retrieval_seconds",
        "retrieval_cache_hits",
        "retrieval_cache_misses",
    ):
        total[key] = total.get(key, 0) + update.get(key, 0)
```

- [ ] **Step 7: Run rollout, skip, and generation-counter tests**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'rollout_group or batched_rollout or zero_advantage or generation_counter'
```

Expected: existing fixed behavior and all adaptive cases PASS.

### Task 4: Diagnostics, serialized rollout evidence, and metadata

**Files:**
- Modify: `src/rl_training/train_grpo_macorag.py:1088-1248, 1390-1405, 1680-1725`
- Test: `tests/test_rl_training.py:1258-1425`

**Interfaces:**
- Consumes timing/diversity fields from Task 3 and action component fields from Task 2.
- Produces stable JSONL fields named exactly as the design spec.

- [ ] **Step 1: Write failing payload tests for diagnostic separation**

Extend metric and rollout-payload fixtures with primary/fallback values and assert:

```python
assert payload["initial_group_size"] == 4
assert payload["effective_group_size"] == 8
assert payload["group_expansion_attempts"] == 1
assert payload["group_expansion_triggered"] is True
assert payload["reward_unique_count_before"] == 1
assert payload["reward_unique_count_after"] == 3
assert payload["terminal_unique_count_before"] == 1
assert payload["terminal_unique_count_after"] == 2
assert payload["primary_nonzero_action_fraction"] == pytest.approx(0.5)
assert payload["fallback_nonzero_action_fraction"] == pytest.approx(0.125)
assert payload["final_nonzero_action_fraction"] == pytest.approx(0.625)
assert rollout_payload["group_rollouts"][0]["action_credit"][0]["primary_advantage"] == 1.0
assert rollout_payload["group_rollouts"][0]["action_credit"][0]["fallback_advantage"] == 0.0
```

- [ ] **Step 2: Run payload tests and confirm red state**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'metrics_payload or rollout_log_payload'
```

Expected: FAIL on missing diagnostic keys.

- [ ] **Step 3: Implement diversity metric calculation**

Add helpers returning before/after unique counts and nonzero fractions. Count unique float rewards with their exact stored values; do not round for training decisions. Fractions use completion-bearing actions as the denominator and return `0.0` for an empty set:

```python
def _completion_actions(rollouts: list[dict[str, Any]]) -> list[Any]:
    return [
        action
        for rollout in rollouts
        for action in rollout.get("actions", [])
        if action.completion_ids
    ]


def _nonzero_action_fraction(
    rollouts: list[dict[str, Any]],
    *,
    field: str,
    tolerance: float = 1e-12,
) -> float:
    actions = _completion_actions(rollouts)
    if not actions:
        return 0.0
    return sum(
        abs(float(getattr(action, field, 0.0))) > tolerance for action in actions
    ) / len(actions)


def _group_diversity_snapshot(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "reward_unique_count": len(
            {float(rollout["rewards"]["total"]) for rollout in rollouts}
        ),
        "terminal_unique_count": len(
            {float(rollout["terminal_reward"]) for rollout in rollouts}
        ),
        "primary_nonzero_action_fraction": _nonzero_action_fraction(
            rollouts, field="primary_advantage"
        ),
        "fallback_nonzero_action_fraction": _nonzero_action_fraction(
            rollouts, field="fallback_advantage"
        ),
        "final_nonzero_action_fraction": _nonzero_action_fraction(
            rollouts, field="advantage"
        ),
    }
```

Capture `initial_snapshot` immediately after scoring the first four and `final_snapshot` after the last scoring pass. Map them to the exact `*_before`, `*_after`, and final fraction fields in the design.

Expansion success is true only when the initial group lacked primary signal and the final group has at least one nonzero `primary_advantage`. Fallback-only signal must not mark expansion success.

- [ ] **Step 4: Add fields to training and rollout JSONL payloads**

Copy the approved keys from `rollout_timing` into `_build_train_metrics_payload`. Extend `_action_credit_payload` with:

```python
"primary_advantage": float(getattr(action, "primary_advantage", action.advantage)),
"fallback_advantage": float(getattr(action, "fallback_advantage", 0.0)),
```

Preserve existing `advantage` for downstream compatibility.

- [ ] **Step 5: Record configuration in run metadata and checkpoint optimization contract**

`resolved_args` already captures parser fields. Also add a nested diversity contract to `train_meta.json` and checkpoint `optimization_contract`:

```python
"group_diversity": {
    "adaptive_group_expansion": bool(args.adaptive_group_expansion),
    "initial_group_size": int(args.group_size),
    "max_group_size": int(args.max_group_size),
    "group_expansion_step": int(args.group_expansion_step),
    "max_group_expansion_attempts": int(args.max_group_expansion_attempts),
    "diversity_retry_temperature": float(args.diversity_retry_temperature),
    "singleton_terminal_fallback_weight": float(args.singleton_terminal_fallback_weight),
},
```

- [ ] **Step 6: Run logging and checkpoint tests**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q tests/test_rl_training.py \
  -k 'metrics_payload or rollout_log_payload or train_meta or checkpoint'
```

Expected: PASS with existing payload keys preserved.

### Task 5: End-to-end regression verification and smoke handoff

**Files:**
- Modify only if a test exposes an implementation defect: files listed in Tasks 1-4.
- Verify: `tests/test_rl_training.py`, `tests/test_vllm_lora_server.py`, `tests/test_rag.py`.

**Interfaces:**
- Consumes the complete configuration, credit, expansion, diagnostics, and checkpoint contracts.
- Produces launch-readiness evidence; it does not launch the 100-sample GPU experiment automatically.

- [ ] **Step 1: Run the complete focused RL suite**

Run:

```bash
/data/conda/envs/macorag/bin/python -m pytest -q \
  tests/test_rl_training.py tests/test_vllm_lora_server.py tests/test_rag.py
```

Expected: all tests PASS.

- [ ] **Step 2: Run static and launcher verification**

Run:

```bash
/data/conda/envs/macorag/bin/python -m compileall -q src/rl_training src/rag
bash -n scripts/run_train_grpo.sh scripts/run_train_grpo_resume.sh scripts/run_grpo_vllm_lora_server.sh
MACORAG_LAUNCH_DRY_RUN=1 PATH=/data/conda/envs/macorag/bin:$PATH \
  bash scripts/run_train_grpo.sh
git diff --check
```

Expected: compile and shell checks exit 0, dry-run prints the canonical config/SFT/GPU contract, and `git diff --check` prints nothing.

- [ ] **Step 3: Verify parsed canonical contract without loading models**

Run:

```bash
PYTHONPATH=src /data/conda/envs/macorag/bin/python -c '
from rl_training.config import parse_args
a = parse_args(["--config", "config/train_grpo.yml"])
print(a.group_size, a.adaptive_group_expansion, a.max_group_size,
      a.group_expansion_step, a.diversity_retry_temperature,
      a.singleton_terminal_fallback_weight)
'
```

Expected output: `4 True 8 4 1.0 0.2`.

- [ ] **Step 4: Produce the operator-owned 100-sample smoke command**

Do not start this expensive run during implementation. Hand off:

```bash
PATH=/data/conda/envs/macorag/bin:$PATH \
  bash scripts/run_train_grpo.sh \
    --max-total-samples 100 \
    --max-steps 100 \
    --run-until-step 100 \
    --save-steps 100
```

After it completes, calculate the spec gate from its `train_metrics.jsonl`: final zero-advantage rate below 25%, optimizer update rate above 70%, mean unique reward count at least 2.3, each role with primary signal, protocol failures below 1%, and finite gradients.

- [ ] **Step 5: Audit the final diff against pre-existing user changes**

Run:

```bash
git status --short
git diff --stat
git diff -- config/train_grpo.yml src/rl_training/config.py \
  src/rl_training/checkpointing.py src/rl_training/policy.py \
  src/rl_training/trainer.py src/rl_training/train_grpo_macorag.py \
  tests/test_rl_training.py
```

Expected: only approved adaptive-diversity changes plus the pre-existing user hunks remain. Do not stage or commit overlapping files automatically; report the exact test evidence and the retained dirty paths.
