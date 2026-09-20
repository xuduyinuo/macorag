from __future__ import annotations

import re
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from .data import (
    load_split, shuffled_training_samples, stratified_validation_samples,
)
from .config import parse_args
from .mappo_types import AgentRole, Episode, MAPPOTransition, RAGState
from .models import (
    FINAL_ANSWER_GUIDED_REGEX, RoleConditionedActor, evidence_guided_regex,
)
from .launch_vllm import _yaml, build_command
from .protocol import (
    PROMPT_CONTRACT,
    build_prompt,
    parse_action,
    role_messages,
    validate_prompt_contract,
)
from .rewards import terminal_reward
from .retrieval import RetrievalEnvironment, RetrievedPassage
from .trainer import MAPPOTrainer, _episode_payload


ROOT = Path(__file__).resolve().parents[2]


def test_v2_fixed_splits_and_shuffle_before_limit() -> None:
    train_path = ROOT / "data_v2/train_rl.jsonl"
    dev_path = ROOT / "data_v2/dev_rl.jsonl"
    train = load_split(train_path)
    dev = load_split(dev_path)
    assert len(train) == 6000
    assert len(dev) == 300
    first = shuffled_training_samples(train_path, seed=42, limit=10)
    again = shuffled_training_samples(train_path, seed=42, limit=10)
    assert [item.qid for item in first] == [item.qid for item in again]
    assert [item.qid for item in first] != [item.qid for item in train[:10]]


def test_validation_limit_is_deterministic_and_proportionally_stratified() -> None:
    path = ROOT / "data_v2/dev_rl.jsonl"
    selected = stratified_validation_samples(path, seed=42, limit=200)
    again = stratified_validation_samples(path, seed=42, limit=200)
    assert len(selected) == 200
    assert [item.qid for item in selected] == [item.qid for item in again]
    assert {
        dataset: sum(item.dataset == dataset for item in selected)
        for dataset in {item.dataset for item in selected}
    } == {"2wiki": 80, "hotpotqa": 80, "musique": 40}
    source_order = {
        item.qid: index for index, item in enumerate(load_split(path))
    }
    assert [source_order[item.qid] for item in selected] == sorted(
        source_order[item.qid] for item in selected
    )


def test_formal_config_enforces_unmodified_vllm_sampling_distribution() -> None:
    config = parse_args(["--config", str(ROOT / "src/rl_v2/train_mappo.yml")])
    assert (config.temperature, config.top_p, config.top_k) == (1.0, 1.0, -1)
    assert config.retrieval_max_length == 128
    assert config.retrieval_batch_size == 256
    assert config.retrieval_use_fp16 is True
    assert config.retrieval_device == "cuda:1"
    assert config.retrieval_top_k == 5
    assert config.retrieval_max_passage_chars == 1200
    assert config.faiss_mmap is False
    assert config.validation_rollout_workers == 32
    assert config.train_rollout_workers == config.rollout_batch_size == 32
    assert config.vllm_max_num_seqs == 16
    assert config.retrieval_batch_wait_ms > 0
    assert config.validation_target_seconds_per_sample == 3.0
    assert config.validation_target_total_minutes == 15.0
    assert config.validation_max_samples == 200
    assert config.max_prompt_length == 1024
    assert config.learner_memory_preflight is True
    config.temperature = 0.8
    with pytest.raises(ValueError, match="unmodified vLLM sampling"):
        config.validate()


def test_vllm_launcher_disables_model_generation_defaults() -> None:
    config_path = ROOT / "src/rl_v2/train_mappo.yml"
    command, _ = build_command(_yaml(config_path), root=ROOT)
    position = command.index("--generation-config")
    assert command[position + 1] == "vllm"


def test_max_samples_is_the_single_post_shuffle_smoke_limit() -> None:
    config = parse_args([
        "--config", str(ROOT / "src/rl_v2/train_mappo.yml"),
        "--max-samples", "7",
    ])
    samples = shuffled_training_samples(
        config.train_file, seed=config.seed, limit=config.max_samples,
    )
    expected = shuffled_training_samples(config.train_file, seed=config.seed)[:7]
    assert [item.qid for item in samples] == [item.qid for item in expected]
    assert len(samples) == 7


def test_retrieval_preserves_faiss_rank_and_eval_passage_limit() -> None:
    environment = RetrievalEnvironment(
        corpus_path="unused", offsets_path="unused", index_path="unused",
        manifest_path="unused", model_path="unused", device="cpu",
        max_length=128, batch_size=256, batch_wait_ms=0, top_k=5,
        mmap=False, use_fp16=False, max_passage_chars=1200,
    )

    class FakeRetriever:
        def search(self, query):
            assert query == "test query"
            return [
                RetrievedPassage("P0", "rank-0", "First", "a" * 1300, 0.9, 10),
                RetrievedPassage("P1", "rank-1", "Second", "b", 0.8, 11),
            ]

    environment._retriever = FakeRetriever()
    result = environment.query("ignored", "test query")
    assert [item["corpus_passage_id"] for item in result["passages"]] == ["rank-0", "rank-1"]
    assert [item["passage_id"] for item in result["passages"]] == [0, 1]
    assert len(result["passages"][0]["text"]) == 1200


def test_prompt_contract_matches_latest_sft_v2_adapter() -> None:
    contract = validate_prompt_contract(
        ROOT / "src/rl_v2/policy_prompts.yml",
        ROOT / "outputs/sft_v2_Qwen2.5-7B/2026-09-18_16-24-28/adapter",
        "macorag-policy-v3",
    )
    assert contract.fingerprint == PROMPT_CONTRACT.fingerprint


def test_policy_messages_keep_sft_v2_few_shots() -> None:
    messages = role_messages(AgentRole.ANSWER, "dynamic", final_round=True)
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "dynamic"}
    assert any("(final round)" in item["content"] for item in messages)


def test_v2_prompt_shapes_and_strict_actions() -> None:
    state = RAGState(
        question="Where?", sub_goal="Find place", round_index=3,
        evidence=[{"title": "T", "text": "Evidence"}],
        retrieval_history=[{"query": "q0"}],
    )
    query = build_prompt(
        AgentRole.QUERY, question=state.question, state=state, max_rounds=4,
    )
    evidence = build_prompt(
        AgentRole.EVIDENCE, question=state.question, state=state,
        observation={"passages": [{"passage_id": 0, "title": "T", "text": "E"}]},
        max_rounds=4,
    )
    answer = build_prompt(
        AgentRole.ANSWER, question=state.question, state=state,
        final_round=True, max_rounds=4,
    )
    assert "selected_evidence" in query
    assert '"passage_id":"P0"' in evidence
    assert "Round: 4 of 4 (final round)" in answer
    assert parse_action(
        '<update-evidence>{"selected_passage_ids":["P0"]}</update-evidence>',
        AgentRole.EVIDENCE,
    ) == {"selected_passage_ids": [0]}
    with pytest.raises(ValueError, match="exactly selected_passage_ids"):
        parse_action(
            '<update-evidence>{"selected_passage_ids":["P0"],"rationale":"x"}</update-evidence>',
            AgentRole.EVIDENCE,
        )


def test_v2_guided_decoding_has_no_rationale() -> None:
    evidence_regex = evidence_guided_regex([0, 1], min_selected=0)
    assert re.fullmatch(
        evidence_regex,
        '<update-evidence>{"selected_passage_ids":[]}</update-evidence>',
    )
    assert re.fullmatch(
        evidence_regex,
        '<update-evidence>{"selected_passage_ids":["P0","P1"]}</update-evidence>',
    )
    assert re.fullmatch(
        FINAL_ANSWER_GUIDED_REGEX,
        '<answer>{"can_answer":true,"answer":"London"}</answer>',
    )
    assert "rationale" not in evidence_regex
    assert "rationale" not in FINAL_ANSWER_GUIDED_REGEX


def test_prompt_truncation_preserves_dynamic_head_and_tail() -> None:
    dynamic = "D" * 100

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            tokens = []
            for index, message in enumerate(messages):
                if message["content"] == dynamic:
                    tokens.extend(range(1000, 1100))
                else:
                    tokens.append(100 + index)
            if kwargs.get("add_generation_prompt"):
                tokens.append(9999)
            return tokens

        def encode(self, text, **kwargs):
            del text, kwargs
            return [7000, 7001]

    actor = RoleConditionedActor.__new__(RoleConditionedActor)
    actor.tokenizer = Tokenizer()
    actor.max_prompt_length = 30
    ids = actor.encode_prompt(AgentRole.QUERY, dynamic)
    assert len(ids) == 30
    assert 1000 in ids
    assert [7000, 7001] == ids[ids.index(7000):ids.index(7000) + 2]
    assert 1099 in ids
    assert ids[-1] == 9999


def test_aliases_participate_in_terminal_answer_reward() -> None:
    _, f1, _ = terminal_reward(
        "NYC", [], ("New York City", "NYC"), (), 1.5, 1.0,
    )
    assert f1 == 1.0


def test_episode_payload_persists_post_initialization_stage_timing() -> None:
    episode = Episode(
        qid="q", dataset="hotpotqa",
        timing={"episode_seconds": 12.5, "retrieval_seconds": 3.0,
                "generation_calls": 6.0},
    )
    assert _episode_payload(episode)["timing"] == episode.timing


def test_cross_sample_rollouts_execute_in_parallel_and_preserve_order() -> None:
    active = 0
    peak = 0
    lock = threading.Lock()

    class Collector:
        def collect(self, sample):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.03)
            with lock:
                active -= 1
            return f"episode-{sample}"

    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.collector = Collector()
    completed = []
    started = time.perf_counter()
    episodes = trainer._collect_episodes(
        list(range(8)), workers=4,
        on_complete=lambda index, sample, episode: completed.append(index),
    )
    elapsed = time.perf_counter() - started
    assert peak == 4
    assert elapsed < 0.18
    assert episodes == [f"episode-{index}" for index in range(8)]
    assert sorted(completed) == list(range(8))


def test_vllm_logprobs_are_replaced_by_local_learner_scores() -> None:
    torch = pytest.importorskip("torch")

    class Actor:
        def score_batch_no_grad(self, sequences, **kwargs):
            del kwargs
            width = max(len(action) for _, action in sequences)
            return torch.full((len(sequences), width), -0.25), torch.zeros(len(sequences), width)

    transition = MAPPOTransition(
        role=AgentRole.QUERY, round_index=0, prompt="p", prompt_ids=[1],
        action_ids=[2, 3], old_token_logprobs=torch.tensor([-9.0, -8.0]),
        central_state={}, next_central_state={},
    )
    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.actor = Actor()
    trainer.torch = torch
    trainer.config = SimpleNamespace(
        use_vllm_generation=True, ppo_old_logprob_source="local_actor",
    )
    trainer._reference_kl_beta = lambda role=None: 0.0
    stats = trainer._align_old_logprobs([transition])
    assert stats["old_logprobs_recomputed"] == 1.0
    assert transition.old_token_logprobs.tolist() == [-0.25, -0.25]
