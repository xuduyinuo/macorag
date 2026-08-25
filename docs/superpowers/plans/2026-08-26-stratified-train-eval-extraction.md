# Stratified Train and Evaluation Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a deterministic paired extractor that emits 2,000 training and 1,000 dev-based evaluation examples per dataset with approved strata, clean scoped corpora, and reproducible manifests.

**Architecture:** `stratified_extraction.py` owns eligibility, deduplication, deterministic selection, corpus scoping, validation, and atomic publication. A thin paired CLI loads explicit train/evaluation YAMLs, selects both splits, audits overlap, then publishes both roots. Existing extraction remains unchanged; new downstream configs are opt-in until matching E5 indexes exist.

**Tech Stack:** Python 3.9+, standard library, PyYAML, pytest, existing `data_processing.io_utils`.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-26-stratified-train-eval-extraction-design.md` exactly.
- Training comes only from `train`; evaluation comes only from `dev`.
- Require both usability flags for training, retrieval usability for evaluation, and empty `quality_flags` everywhere.
- Use seed `20260826`, fixed quotas, sampling without replacement, and deterministic final shuffle.
- Never overwrite old datasets/indexes or an existing v2 target.
- Do not modify `src/data_processing/extract_trajectory_datasets.py`.
- Do not commit generated datasets/indexes; preserve unrelated worktree changes.
- Do not launch teacher generation, GRPO, evaluation, or full E5 builds.

## File Map

- Create `src/data_processing/stratified_extraction.py`: extraction domain logic.
- Create `src/data_processing/extract_stratified_datasets.py`: paired YAML/CLI boundary.
- Create `tests/test_stratified_extraction.py`: synthetic contract tests.
- Create train/eval extraction, retrieval, and opt-in runtime configs under `config/`.
- Generate but do not commit the two v2 data roots during acceptance.

The test file defines reusable builders before its tests. `make_row(dataset,
split, qid, stratum, *, level=None, question=None)` returns a schema-complete
row with both usability flags true, empty quality flags, one context document,
and one supporting fact. `write_candidate_fixture`, `write_one_row_fixture`,
`write_corpus`, `write_duplicate_corpus`, and `selected_fixture` write only
under pytest's `tmp_path`. `paired_fixture_configs` creates canonical processed
train/dev files and corpora for all three dataset names plus non-existing output
roots; its `overlap_question=True` option gives one train/dev pair the same
normalized question. `write_pair_configs` serializes that pair as YAML, while
`write_config` writes one deliberately minimal YAML for loader failures. These
builders are test-only and must not be imported by production code.

---

### Task 1: Eligibility, stratum keys, and deterministic selection

**Files:**
- Create: `src/data_processing/stratified_extraction.py`
- Create: `tests/test_stratified_extraction.py`

**Interfaces:**
- `normalize_question(value: str) -> str`
- `stratum_key(dataset: str, row: dict[str, Any]) -> str`
- `eligibility_error(row: dict[str, Any], *, split: str) -> str | None`
- `derive_seed(seed: int, *parts: str) -> int`
- `select_rows(*, source_path: Path, dataset: str, split: str, quotas: dict[str, int], seed: int) -> SelectionResult`
- `SelectionResult(rows, source_indices, qids, quota_actual, eligible_count, excluded_by_reason)`

- [ ] **Step 1: Write failing primitive tests**

```python
def test_primitives():
    assert normalize_question("  Who's Alice? ") == "who s alice"
    row = make_row("2wiki", "train", "q1", "inference")
    assert eligibility_error(row, split="train") is None
    row["usable_for_retrieval_eval"] = False
    assert eligibility_error(row, split="train") == "not_usable_for_retrieval_eval"
    row["usable_for_retrieval_eval"] = True
    row["quality_flags"] = ["missing_supporting_fact_text"]
    assert eligibility_error(row, split="train") == "quality_flags"
    assert stratum_key("2wiki", make_row("2wiki", "train", "w", "inference")) == "inference"
    hotpot = make_row("hotpotqa", "train", "h", "bridge", level="hard")
    assert stratum_key("hotpotqa", hotpot) == "hard/bridge"
    assert stratum_key("musique", make_row("musique", "train", "4hop2__1", None)) == "4hop2"
    assert derive_seed(20260826, "2wiki", "train") != derive_seed(20260826, "2wiki", "dev")
```

- [ ] **Step 2: Verify red**

Run: `pytest -q tests/test_stratified_extraction.py -k primitives`

Expected: `ModuleNotFoundError` for `data_processing.stratified_extraction`.

- [ ] **Step 3: Implement primitives**

```python
@dataclass(frozen=True)
class SelectionResult:
    rows: list[dict[str, Any]]
    source_indices: list[int]
    qids: list[str]
    quota_actual: dict[str, int]
    eligible_count: int
    excluded_by_reason: dict[str, int]


def normalize_question(value: str) -> str:
    return " ".join(re.sub(r"[^\w]+", " ", str(value).casefold()).split())


def derive_seed(seed: int, *parts: str) -> int:
    payload = "\0".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
```

Implement deterministic error precedence: `wrong_split`, `missing_qid`,
`missing_question`, `missing_answer`, `invalid_supporting_facts`,
`not_usable_for_sft`, `not_usable_for_retrieval_eval`, `quality_flags`.

- [ ] **Step 4: Write failing selection tests**

```python
def test_select_rows_is_exact_deduplicated_and_deterministic(tmp_path):
    source = write_candidate_fixture(tmp_path)
    kwargs = dict(source_path=source, dataset="2wiki", split="train",
                  quotas={"comparison": 3, "inference": 2}, seed=20260826)
    first = select_rows(**kwargs)
    assert first == select_rows(**kwargs)
    assert first.quota_actual == {"comparison": 3, "inference": 2}
    assert len({normalize_question(row["question"]) for row in first.rows}) == 5


def test_select_rows_rejects_underfilled_stratum(tmp_path):
    source = write_one_row_fixture(tmp_path, stratum="inference")
    with pytest.raises(ValueError, match="inference.*required=2.*available=1"):
        select_rows(source_path=source, dataset="2wiki", split="train",
                    quotas={"inference": 2}, seed=20260826)


def test_select_rows_changes_when_seed_changes(tmp_path):
    source = write_candidate_fixture(tmp_path)
    one = select_rows(source_path=source, dataset="2wiki", split="train",
                      quotas={"comparison": 2}, seed=1)
    two = select_rows(source_path=source, dataset="2wiki", split="train",
                      quotas={"comparison": 2}, seed=2)
    assert one.qids != two.qids
```

- [ ] **Step 5: Verify red**

Run: `pytest -q tests/test_stratified_extraction.py -k select_rows`

Expected: failure because `select_rows` is absent.

- [ ] **Step 6: Implement selection**

Read once into configured stratum buckets, deduplicate normalized questions
before sampling, use `random.Random(derive_seed(seed, dataset, split,
stratum)).sample`, combine exact quotas, and shuffle using the derived `final`
seed. Preserve selected source indices in final row order.

- [ ] **Step 7: Verify green and commit**

```bash
pytest -q tests/test_stratified_extraction.py -k 'primitives or select_rows'
git add src/data_processing/stratified_extraction.py tests/test_stratified_extraction.py
git commit -m "feat: add deterministic stratified selection"
```

Expected: focused tests pass; commit contains only the two named files.

### Task 2: Scoped corpus, manifest, and independent validation

**Files:**
- Modify: `src/data_processing/stratified_extraction.py`
- Modify: `tests/test_stratified_extraction.py`

**Interfaces:**
- `required_doc_ids(rows: list[dict[str, Any]]) -> set[str]`
- `scope_corpus(source: Path, required: set[str]) -> list[dict[str, Any]]`
- `sha256_file(path: Path) -> str`
- `write_dataset_output(...) -> dict[str, Any]`
- `validate_dataset_output(dataset_dir: Path, *, dataset: str, split: str, quotas: dict[str, int]) -> dict[str, Any]`

- [ ] **Step 1: Write failing corpus tests**

```python
def test_scope_corpus_is_exact(tmp_path):
    corpus = write_corpus(tmp_path, ["d1", "d2", "unused"])
    assert [r["doc_id"] for r in scope_corpus(corpus, {"d2", "d1"})] == ["d1", "d2"]
    with pytest.raises(ValueError, match="missing required corpus docs: missing"):
        scope_corpus(corpus, {"missing"})


def test_scope_corpus_rejects_duplicate_required_id(tmp_path):
    corpus = write_duplicate_corpus(tmp_path, "d1")
    with pytest.raises(ValueError, match="duplicate corpus doc_id: d1"):
        scope_corpus(corpus, {"d1"})
```

- [ ] **Step 2: Verify red, implement exact corpus and hashing, verify green**

Run before implementation: `pytest -q tests/test_stratified_extraction.py -k scope_corpus`

Expected: missing-function failure.

```python
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

Preserve canonical corpus order, reject duplicate required ids, and require
`seen == required`. Rerun the focused command; expected PASS.

- [ ] **Step 3: Write failing output/manifest test**

```python
def test_write_and_validate_records_relative_paths_hashes_and_counts(tmp_path):
    fixture = selected_fixture(tmp_path)
    summary = write_dataset_output(output_root=tmp_path / "out", **fixture.writer_args)
    assert summary["output_examples"] == "2wiki/2wiki_train.jsonl"
    assert len(summary["output_sha256"]["examples"]) == 64
    audit = validate_dataset_output(tmp_path / "out" / "2wiki",
                                    dataset="2wiki", split="train",
                                    quotas={"inference": 2})
    assert audit["example_count"] == 2
    assert audit["required_corpus_count"] == audit["corpus_count"]
```

- [ ] **Step 4: Implement writer and validator**

Write example JSONL, exact corpus JSONL, and `extract_summary.json` containing
repo-relative sources, output-root-relative outputs, source hashes, exclusions,
quota, indices, qids, counts, and artifact hashes. Validator must reload files
and recalculate every acceptance condition rather than trusting memory.

- [ ] **Step 5: Verify and commit**

```bash
pytest -q tests/test_stratified_extraction.py -k 'corpus or manifest or write_and_validate'
git add src/data_processing/stratified_extraction.py tests/test_stratified_extraction.py
git commit -m "feat: add scoped corpus and extraction manifests"
```

Expected: focused tests pass.

### Task 3: Paired atomic publisher and CLI

**Files:**
- Modify: `src/data_processing/stratified_extraction.py`
- Create: `src/data_processing/extract_stratified_datasets.py`
- Modify: `tests/test_stratified_extraction.py`

**Interfaces:**
- `load_extraction_config(path: Path) -> dict[str, Any]`
- `extract_pair(train_config: dict[str, Any], eval_config: dict[str, Any]) -> dict[str, Any]`
- CLI: `python -m data_processing.extract_stratified_datasets --train-config PATH --eval-config PATH [--dry-run|--audit-existing]`

- [ ] **Step 1: Write failing paired-publication tests**

```python
def test_extract_pair_publishes_both_after_zero_overlap_audit(tmp_path):
    train, evaluation = paired_fixture_configs(tmp_path)
    manifest = extract_pair(train, evaluation)
    assert manifest["overlap_audit"] == {"qid_count": 0, "normalized_question_count": 0}
    assert Path(train["output_root"]).is_dir()
    assert Path(evaluation["output_root"]).is_dir()
    assert not list(tmp_path.glob("*.staging-*"))


def test_extract_pair_refuses_existing_or_overlapping_targets(tmp_path):
    train, evaluation = paired_fixture_configs(tmp_path, overlap_question=True)
    with pytest.raises(ValueError, match="normalized-question overlap"):
        extract_pair(train, evaluation)
    assert not Path(train["output_root"]).exists()
    Path(train["output_root"]).mkdir()
    with pytest.raises(FileExistsError, match="target already exists"):
        extract_pair(train, evaluation)


def test_handled_failure_removes_staging(tmp_path, monkeypatch):
    train, evaluation = paired_fixture_configs(tmp_path)
    def fail_validation(*args, **kwargs):
        raise ValueError("forced validation failure")
    monkeypatch.setattr(
        "data_processing.stratified_extraction.validate_dataset_output",
        fail_validation,
    )
    with pytest.raises(ValueError, match="forced validation failure"):
        extract_pair(train, evaluation)
    assert not list(tmp_path.glob("*.staging-*"))
```

- [ ] **Step 2: Verify red and implement paired staging**

Run: `pytest -q tests/test_stratified_extraction.py -k extract_pair`

Expected: missing-function failure. Select all six groups before writing, audit
cross-split qids/questions, write into unique sibling staging roots, validate
all outputs, then publish with `Path.replace`. Remove staging on handled errors;
report but never resume pre-existing `*.staging-*` paths.

- [ ] **Step 3: Write failing CLI/config tests**

```python
def test_cli_dry_run_writes_nothing(tmp_path, capsys):
    train_path, eval_path = write_pair_configs(tmp_path)
    assert main(["--train-config", str(train_path), "--eval-config", str(eval_path),
                 "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["train"]["total_quota"] == 6
    assert not (tmp_path / "train_out").exists()


def test_config_rejects_wrong_total(tmp_path):
    path = write_config(tmp_path, quotas={"inference": 1}, expected_total=2)
    with pytest.raises(ValueError, match="quota total 1 != expected_total 2"):
        load_extraction_config(path)
```

- [ ] **Step 4: Implement strict YAML loader and thin CLI**

Require `source_root`, `output_root`, `split`, `seed`, `expected_total`, and
quota mappings for exactly the three datasets. Default to the production train
and eval config paths. `--dry-run` validates and prints resolved quotas only;
`--audit-existing` independently reloads and audits published artifacts.

- [ ] **Step 5: Verify and commit**

```bash
pytest -q tests/test_stratified_extraction.py
git add src/data_processing/stratified_extraction.py src/data_processing/extract_stratified_datasets.py tests/test_stratified_extraction.py
git commit -m "feat: add paired atomic extraction CLI"
```

Expected: complete synthetic suite passes.

### Task 4: Production and opt-in downstream configurations

**Files:**
- Create: `config/extract_stratified_train_v2.yml`
- Create: `config/extract_stratified_eval_v2.yml`
- Create: `config/build_retrieval_train_stratified_v2_e5.yml`
- Create: `config/build_retrieval_eval_stratified_v2_e5.yml`
- Create: `config/train_grpo_stratified_v2.yml`
- Create: `config/eval_macorag_stratified_v2.yml`
- Modify: `tests/test_stratified_extraction.py`

**Interfaces:**
- Produces the CLI defaults and four opt-in configs.
- Leaves `config/train_grpo.yml` and `config/eval_macorag.yml` unchanged.

- [ ] **Step 1: Write a failing production-config test**

```python
def test_production_configs_match_approved_contract():
    train = load_extraction_config(Path("config/extract_stratified_train_v2.yml"))
    evaluation = load_extraction_config(Path("config/extract_stratified_eval_v2.yml"))
    assert train["seed"] == evaluation["seed"] == 20260826
    assert (train["split"], evaluation["split"]) == ("train", "dev")
    assert (train["expected_total"], evaluation["expected_total"]) == (2000, 1000)
    assert train["datasets"]["2wiki"]["quotas"] == {
        "compositional": 831, "comparison": 486,
        "bridge_comparison": 440, "inference": 243,
    }
    assert evaluation["datasets"]["musique"]["quotas"] == {
        "2hop": 518, "3hop1": 235, "3hop2": 79,
        "4hop1": 102, "4hop2": 27, "4hop3": 39,
    }
```

- [ ] **Step 2: Verify red**

Run: `pytest -q tests/test_stratified_extraction.py::test_production_configs_match_approved_contract`

Expected: `FileNotFoundError` for the train config.

- [ ] **Step 3: Add exact extraction configs**

```yaml
# config/extract_stratified_train_v2.yml
schema_version: 1
source_root: data/processed
output_root: data/rl_train_2000_stratified_v2
split: train
seed: 20260826
expected_total: 2000
datasets:
  2wiki:
    quotas: {compositional: 831, comparison: 486, bridge_comparison: 440, inference: 243}
  hotpotqa:
    quotas: {hard/bridge: 1596, hard/comparison: 404}
  musique:
    quotas: {2hop: 1036, 3hop1: 470, 3hop2: 159, 4hop1: 203, 4hop2: 53, 4hop3: 79}
```

The evaluation config uses `data/eval_1000_stratified_v2`, split `dev`, total
`1000`, and exact approved eval quotas for all datasets.

- [ ] **Step 4: Add opt-in retrieval configs**

```yaml
command: build
backend: e5_faiss
data_root: data/rl_train_2000_stratified_v2
retrieval_root: data/rl_train_2000_stratified_v2_e5_faiss
datasets: [2wiki, hotpotqa, musique]
embedding_model: intfloat/e5-base-v2
device: cpu
max_length: 512
batch_size: 128
retrieval_top_k: 5
```

The eval retrieval config changes only the two roots to eval v2.

- [ ] **Step 5: Add opt-in runtime configs**

Copy the current live configs, then change exactly these fields:

```yaml
# train_grpo_stratified_v2.yml
rl_data_root: data/rl_train_2000_stratified_v2
retrieval_root: data/rl_train_2000_stratified_v2_e5_faiss
max_samples: 2000

# eval_macorag_stratified_v2.yml
data_root: data/eval_1000_stratified_v2
retrieval_root: data/eval_1000_stratified_v2_e5_faiss
max_samples: null
```

- [ ] **Step 6: Verify and commit**

```bash
pytest -q tests/test_stratified_extraction.py -k 'production_configs or cli_dry_run or config_rejects'
git add config/extract_stratified_train_v2.yml config/extract_stratified_eval_v2.yml config/build_retrieval_train_stratified_v2_e5.yml config/build_retrieval_eval_stratified_v2_e5.yml config/train_grpo_stratified_v2.yml config/eval_macorag_stratified_v2.yml tests/test_stratified_extraction.py
git commit -m "config: define stratified v2 data contracts"
```

Expected: focused tests pass; current default configs remain untouched.

### Task 5: Real extraction and artifact audit

**Files:**
- Generate, do not commit: `data/rl_train_2000_stratified_v2/`
- Generate, do not commit: `data/eval_1000_stratified_v2/`
- Modify code only after adding a failing regression test if canonical data exposes a defect.

**Interfaces:**
- Produces 6,000 train rows, 3,000 dev rows, scoped corpora, summaries, and manifests.

- [ ] **Step 1: Preview production quotas**

```bash
PYTHONPATH=src python -m data_processing.extract_stratified_datasets \
  --train-config config/extract_stratified_train_v2.yml \
  --eval-config config/extract_stratified_eval_v2.yml \
  --dry-run
```

Expected: seed `20260826`, per-dataset totals `2000/1000`, exact strata, no outputs.

- [ ] **Step 2: Refuse accidental overwrite**

```bash
test ! -e data/rl_train_2000_stratified_v2
test ! -e data/eval_1000_stratified_v2
```

Expected: exit 0. If either exists, stop and inspect it; do not delete it.

- [ ] **Step 3: Run paired extraction**

```bash
PYTHONPATH=src python -m data_processing.extract_stratified_datasets \
  --train-config config/extract_stratified_train_v2.yml \
  --eval-config config/extract_stratified_eval_v2.yml
```

Expected: both roots publish only after all six dataset audits pass.

- [ ] **Step 4: Independently audit published artifacts**

```bash
PYTHONPATH=src python -m data_processing.extract_stratified_datasets \
  --train-config config/extract_stratified_train_v2.yml \
  --eval-config config/extract_stratified_eval_v2.yml \
  --audit-existing
```

Expected: train `6000`, eval `3000`, exact strata, zero unusable/flagged rows,
unique qids/questions, zero overlap, exact corpora, and matching hashes.

- [ ] **Step 5: Verify byte determinism in temporary roots**

Create an explicit `mktemp -d`, copy configs while changing only output roots,
run extraction there, compare every JSONL/manifest SHA-256 to production, then
remove only that validated temporary directory. Expected: all digests match.

- [ ] **Step 6: Handle any canonical-data defect with red-green TDD**

Add a focused failing test reproducing the exact row, run it red, make the
minimal fix, run it green, and repeat extraction into new temporary roots.
Never mutate a published v2 root to hide an audit failure.

- [ ] **Step 7: Confirm generated data is not staged**

Run: `git status --short data/rl_train_2000_stratified_v2 data/eval_1000_stratified_v2`

Expected: generated data is untracked or ignored and not staged.

### Task 6: Final verification and retrieval handoff

**Files:**
- No required changes.

**Interfaces:**
- Produces verified extraction artifacts and matching index-build commands.

- [ ] **Step 1: Run extraction tests**

Run: `pytest -q tests/test_stratified_extraction.py`

Expected: zero failures; record exact pass count.

- [ ] **Step 2: Run affected regression suite**

```bash
pytest -q tests/test_io_utils.py tests/test_dataset_builders.py \
  tests/test_schemas.py tests/test_rl_training.py tests/test_evaluation.py \
  tests/test_retrieval_env.py
```

Expected: zero failures; record exact pass count.

- [ ] **Step 3: Re-run `--audit-existing`**

Expected: all real-data acceptance gates still pass.

- [ ] **Step 4: Verify commit/worktree boundary**

```bash
git log --oneline --max-count=6
git status --short
```

Expected: implementation commits contain only planned code/tests/configs;
pre-existing unrelated changes remain; generated data is not committed.

- [ ] **Step 5: Hand off matching E5 builds without launching them**

```bash
PYTHONPATH=src python -m data_processing.retrieval_cli --config config/build_retrieval_train_stratified_v2_e5.yml
PYTHONPATH=src python -m data_processing.retrieval_cli --config config/build_retrieval_eval_stratified_v2_e5.yml
```

Do not use the new runtime configs or claim indexes exist until these commands
are separately authorized, run, and validated.
