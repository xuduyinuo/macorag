# Proportional RL Sample Limits Design

## Goal

When RL training limits each dataset to fewer than the extracted 2,000
examples, select a deterministic subset that preserves that dataset's current
question-stratum proportions. A 500-example subset must be contained in the
corresponding 1,000-example subset, and changing the training seed must not
change either sample set.

This behavior is opt-in for the stratified-v2 training configuration. Existing
training configurations retain their current head-limiting behavior.

## Current behavior and problem

`load_rl_samples` currently stops accepting examples after it has encountered
`max_samples` valid rows for a dataset. The extracted files are deterministically
shuffled, but taking their first 500 or 1,000 rows does not enforce the intended
question-type distribution.

The relevant strata in the validated v2 data are:

- 2Wiki: top-level `question_type`.
- HotpotQA: `metadata.level/question_type` (currently `hard/bridge` and
  `hard/comparison`).
- MuSiQue: the qid prefix before `__` (for example `3hop1`).

Training's later epoch shuffle is a separate concern. It may change processing
order, but it must not choose the limited sample set.

## Configuration contract

Add two RL data-loading options:

```yaml
data_sampling_strategy: proportional_stratified
data_sampling_seed: 20260826
```

`data_sampling_strategy` accepts:

- `head`: preserve the existing per-dataset first-N behavior. This remains the
  default for backward compatibility.
- `proportional_stratified`: build a deterministic proportional prefix for each
  dataset, then take its first `max_samples` examples.

`data_sampling_seed` controls only membership and within-stratum ranking. It is
independent of the existing training `seed`, which continues to control epoch
ordering and training randomness.

`config/train_grpo_stratified_v2.yml` enables the proportional strategy and
sets `data_sampling_seed: 20260826`. Existing default configurations are not
changed.

## Proportional prefix algorithm

The loader first reads and validates all candidate examples. For each dataset:

1. Derive each example's stratum using the dataset rules above.
2. Group examples by stratum.
3. Deterministically shuffle each bucket using a seed derived from
   `data_sampling_seed`, dataset, and stratum.
4. Construct one full-length schedule. At schedule position `p`, choose the
   nonempty stratum with the largest cumulative deficit. Compare the integer
   numerator below rather than floating-point ratios:

   ```text
   deficit_numerator = p * stratum_size - selected_so_far * dataset_size
   ```

5. Resolve equal deficits by these canonical orders:
   - 2Wiki: `compositional`, `comparison`, `bridge_comparison`, `inference`.
   - HotpotQA: `hard/bridge`, `hard/comparison`.
   - MuSiQue: `2hop`, `3hop1`, `3hop2`, `4hop1`, `4hop2`, `4hop3`.
6. Append the next example from the selected stratum's shuffled bucket.
7. Return the first `min(max_samples, dataset_size)` scheduled examples.

Because every limit uses a prefix of the same schedule, sample membership is
house-monotonic: the 500-example set is contained in the 1,000-example set, and
both are contained in the full set. At every prefix, stratum counts remain as
close as possible to their cumulative target under this scheduling rule.

The implementation must not use Python's process-randomized `hash()`. Seed
derivation must use a stable digest so results are reproducible across
processes.

## Expected v2 quotas

The accepted algorithm and stable stratum order produce these counts:

| Dataset | Stratum | 500 | 1,000 |
|---|---|---:|---:|
| 2Wiki | compositional | 208 | 416 |
| 2Wiki | comparison | 121 | 243 |
| 2Wiki | bridge_comparison | 110 | 220 |
| 2Wiki | inference | 61 | 121 |
| HotpotQA | hard/bridge | 399 | 798 |
| HotpotQA | hard/comparison | 101 | 202 |
| MuSiQue | 2hop | 259 | 518 |
| MuSiQue | 3hop1 | 117 | 235 |
| MuSiQue | 3hop2 | 40 | 80 |
| MuSiQue | 4hop1 | 51 | 101 |
| MuSiQue | 4hop2 | 13 | 27 |
| MuSiQue | 4hop3 | 20 | 39 |

The one-example rounding differences are intentional consequences of integer
prefix allocation. Exact counts, deterministic qids, and nesting are acceptance
requirements.

## Loader and training data flow

`load_rl_samples` receives the strategy and sampling seed. Under `head`, it
retains the existing streaming limit. Under `proportional_stratified`, it reads
all valid rows, derives strata, constructs each dataset schedule, applies the
per-dataset limit, and returns the selected samples.

`train_grpo_macorag.main` passes the parsed strategy and sampling seed to the
loader. The existing `select_balanced_samples` call remains responsible only
for the later optional `max_total_samples` limit and dataset interleaving.
Normal 500/1,000-example experiments must leave `max_total_samples` unset;
otherwise that later global reduction can change per-stratum counts.

The existing epoch shuffle remains unchanged. Consequently:

- `data_sampling_seed` determines the limited set.
- `seed` determines training and epoch order.
- `max_samples` determines the per-dataset prefix length.
- `max_total_samples` remains a distinct, subsequent global cap.

## Validation and errors

The proportional strategy supports only `2wiki`, `hotpotqa`, and `musique`.
It fails before training if:

- a supported dataset row lacks the fields needed to derive its stratum;
- a derived stratum is not in that dataset's canonical order;
- a dataset name is unsupported;
- `data_sampling_seed` is not an integer;
- the selected strategy is unknown.

The loader must not silently fall back to head selection or random sampling.
`max_samples=None` loads all valid samples without changing membership.
`max_samples` greater than or equal to a dataset's valid count also returns the
full dataset.

## Summary and observability

The data summary retains existing keys and adds:

- `data_sampling_strategy`
- `data_sampling_seed`
- `counts_by_dataset_and_stratum`

The nested mapping reports the actual selected count for every dataset and
stratum. It is written into the existing training metadata through the current
`data_summary` flow, making experiment composition auditable.

## Testing

Focused tests must cover:

1. Exact 500 and 1,000 quotas for representative v2-shaped samples.
2. Set containment: 500 qids are a subset of 1,000 qids.
3. Repeated calls with the same sampling seed return identical qids.
4. Changing the training seed does not affect loader membership because it is
   not passed as the sampling seed.
5. Changing `data_sampling_seed` changes membership while preserving counts.
6. Full limits return every valid example.
7. Missing stratum fields and unsupported datasets fail clearly.
8. `head` retains the existing first-N behavior.
9. CLI/YAML parsing exposes both options and the v2 config enables them.
10. The real 2,000-example v2 artifacts produce the quota table above for
    limits 500 and 1,000.

No changes to evaluation sampling, extraction quotas, retrieval indexes,
reward calculation, epoch ordering, or global balancing are in scope.
