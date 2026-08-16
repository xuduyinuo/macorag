# Role-Round Credit Assignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every Query, Evidence, and Answer action a role- and round-specific GRPO advantage while preserving the shared policy.

**Architecture:** Reward computation emits per-action local rewards and a process-free terminal score. The rollout group normalizes local and terminal values in `(role, round_index)` buckets, mixes them with role-specific weights, and training consumes the resulting action advantage.

**Tech Stack:** Python, PyTorch, pytest, YAML, existing custom GRPO trainer

## Global Constraints

- Keep one shared policy model and one shared optimizer.
- Do not add a critic or migrate from the existing custom GRPO loss.
- Use no temporal discount across RAG rounds.
- Preserve existing aggregate reward fields and best-rollout selection.
- Do not modify or unpack `src/rl_single.tar.gz`.

---

### Task 1: Define and test action-level reward outputs

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `src/rl_training/rewards.py`

**Interfaces:**
- Produces: `compute_action_rewards(rollout, sample) -> dict[str, object]`
- Produces: action records with `role`, `round_index`, and `local_reward`
- Produces: `terminal_reward: float`

- [x] Add tests showing two rounds receive distinct Query, Evidence, and Answer local rewards.
- [x] Run the targeted tests and confirm failure because `compute_action_rewards` is missing.
- [x] Implement per-round local rewards and the process-free terminal reward.
- [x] Run reward tests and preserve existing `compute_rl_rewards` behavior.

### Task 2: Attach round identity and credit fields to generated actions

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `src/rl_training/policy.py`

**Interfaces:**
- Extends: `GeneratedAction.round_index`, `local_reward`, `terminal_reward`, `advantage`

- [x] Add tests for round propagation through the shared HF/vLLM trace implementation.
- [x] Run tests and confirm the new field assertions fail.
- [x] Derive round indices from prior same-role trace actions and store them on generated actions.
- [x] Run policy and executor tests.

### Task 3: Compute role-round GRPO advantages

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `src/rl_training/trainer.py`
- Modify: `src/rl_training/train_grpo_macorag.py`
- Modify: `src/rl_training/config.py`
- Modify: `config/train_grpo.yml`

**Interfaces:**
- Produces: `assign_action_advantages(rollouts, local_weights) -> None`
- Consumes: per-action local rewards and rollout terminal rewards

- [x] Add tests for same-role/same-round normalization, mixed advantages, singleton buckets, and config parsing.
- [x] Run tests and confirm failure because action assignment is absent.
- [x] Implement bucket normalization and configurable role weights.
- [x] Update rollout generation to compute and assign action credit.
- [x] Run trainer and configuration tests.

### Task 4: Train from action-level advantages and verify compatibility

**Files:**
- Modify: `tests/test_rl_training.py`
- Modify: `src/rl_training/train_grpo_macorag.py`

**Interfaces:**
- Consumes: `GeneratedAction.advantage`
- Preserves: `compute_grpo_loss(...)`

- [x] Add a test proving two actions in one rollout backpropagate with different advantages.
- [x] Run the test and confirm it fails while rollout advantage is still shared.
- [x] Change training to read `action.advantage` and expose credit diagnostics in rollout logs.
- [x] Run focused tests, then the full RL training test module.
