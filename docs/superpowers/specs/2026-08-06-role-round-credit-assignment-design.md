# Role-Round Credit Assignment Design

## Goal

Replace trajectory-level equal feedback with action-level credit assignment while retaining the existing shared policy, GRPO loss, vLLM generation, and retrieval pipeline.

## Credit Model

Each generated action is identified by `role` and `round_index`. Its training advantage is:

```text
action_advantage = local_weight(role) * local_advantage
                 + (1 - local_weight(role)) * terminal_advantage
```

Local advantages compare equivalent actions from the same question using the key `(role, round_index)`. Terminal advantages compare the terminal task score of rollouts that contain that key. No temporal discount is applied.

Default local weights are:

- Query Retriever: `0.75`
- Evidence Updater: `0.70`
- Answer Generator: `0.30`

## Reward Boundaries

Query local reward uses only query output and the passages returned by that query: valid structure, useful length, novelty, retrieval cost, and newly retrieved supporting facts.

Evidence local reward uses only the current observation and evidence update: selected IDs, invalid IDs, excessive selection, irrelevant passages, and newly selected supporting facts.

Answer local reward evaluates the current round decision: correct waiting when all labeled support is not yet covered, premature answering, false abstention when all labeled support is covered, answer correctness when answering, and parse validity. Datasets without support labels do not receive inferred sufficiency rewards.

Terminal reward excludes query and evidence process terms to avoid double counting. It contains final answer correctness, final support coverage over all labeled facts, premature final-answer penalty, and trajectory parse penalty. Answer text contributes to terminal correctness only when `can_answer` is true.

If parsing fails after a role has generated output, the executor records a partial turn with `generated_roles` and `parse_error_role`. Completed upstream work keeps its local credit, while the parse penalty is assigned to the failed role.

## Variable-Length Rollouts

Only rollouts containing a given `(role, round_index)` participate in that comparison bucket. A bucket with one member receives zero normalized advantage for that component. Missing actions receive no loss.

## Data Flow

`GeneratedAction` records `round_index`, local reward, terminal reward, and final advantage. Because the executor always invokes each role once per completed Q-E-A round, the shared policy derives the round from the number of prior actions with the same role. Reward computation returns per-action local rewards plus a terminal score. The rollout group normalizes rewards per role and round and writes the mixed advantage to each action. Training reads `action.advantage`; rollout-level total reward remains available for logging and best-rollout selection.

## Compatibility

The shared model, old log probabilities, reference-model KL, clipping, optimizer behavior, and existing aggregate reward fields remain unchanged. New YAML weights are optional and have code defaults.

## Verification

Tests must cover per-round reward differences, role/round group normalization, terminal/local mixing, variable-length buckets, action-level advantage use in training, configuration parsing, and legacy aggregate reward compatibility.
