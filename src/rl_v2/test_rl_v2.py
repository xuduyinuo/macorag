from __future__ import annotations

import re
import threading
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from .data import (
    evenly_spaced_steps, load_split, shuffled_training_samples,
    stratified_epoch_order, stratified_validation_samples,
)
from .config import parse_args
from .mappo_types import (
    AgentRole, Episode, MAPPOTransition, RAGState, RLSample,
)
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
from .rewards import evidence_reward, terminal_reward
from .retrieval import RetrievalEnvironment, RetrievedPassage
from .rollout import RolloutCollector
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
    assert config.local_reward_enabled is True
    assert config.gradient_accumulation_steps == 8
    assert config.actor_learning_rate == pytest.approx(5.0e-7)
    assert config.entropy_anneal_start_step == 31
    assert config.entropy_anneal_end_step == 63
    assert config.answer_local_reward_weight == pytest.approx(0.2)
    assert config.non_final_wait_reward == pytest.approx(0.05)
    assert config.final_answer_bonus == pytest.approx(0.2)
    assert config.evidence_duplicate_penalty == pytest.approx(0.2)
    assert config.num_train_epochs == 1.0
    assert config.validation_steps == 0
    assert config.validation_checks_per_epoch == 10
    assert config.save_on_validation is True
    assert config.early_stopping_patience == 0
    assert config.validation_score_answer_weight == 1.0
    assert config.validation_score_evidence_weight == 1.0
    assert config.validation_score_format_weight == 1.0
    config.temperature = 0.8
    with pytest.raises(ValueError, match="unmodified vLLM sampling"):
        config.validate()


def test_outcome_only_config_changes_only_reward_mode_and_output_root() -> None:
    base = parse_args(["--config", str(ROOT / "src/rl_v2/train_mappo.yml")])
    outcome = parse_args([
        "--config", str(ROOT / "src/rl_v2/train_mappo_outcome_only.yml"),
    ])
    assert outcome.local_reward_enabled is False
    assert outcome.output_root == "outputs/rl_v2_outcome_only_Qwen2.5-7B-Instruct"
    ignored = {"local_reward_enabled", "output_root"}
    assert {
        key: value for key, value in base.to_dict().items() if key not in ignored
    } == {
        key: value for key, value in outcome.to_dict().items() if key not in ignored
    }


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


def test_stratified_epoch_batches_are_deterministic_and_lossless() -> None:
    samples = [
        RLSample(
            qid=f"{dataset}-{index}", dataset=dataset, question="q",
            answer="a", answer_aliases=(),
            supporting_facts=({"title": "t", "text": "x"},),
        )
        for dataset, count in (("2wiki", 40), ("hotpotqa", 40), ("musique", 20))
        for index in range(count)
    ]
    ordered = stratified_epoch_order(samples, seed=42, epoch=0, batch_size=32)
    again = stratified_epoch_order(samples, seed=42, epoch=0, batch_size=32)
    assert [item.qid for item in ordered] == [item.qid for item in again]
    assert Counter(item.qid for item in ordered) == Counter(
        item.qid for item in samples
    )
    for start in range(0, 96, 32):
        counts = Counter(item.dataset for item in ordered[start:start + 32])
        assert counts["2wiki"] in {12, 13}
        assert counts["hotpotqa"] in {12, 13}
        assert counts["musique"] in {6, 7}


def test_validation_schedule_has_ten_checks_and_includes_final_step() -> None:
    assert evenly_spaced_steps(94, 10) == (
        10, 19, 29, 38, 47, 57, 66, 76, 85, 94,
    )
    assert evenly_spaced_steps(4, 10) == (1, 2, 3, 4)


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


def test_evidence_reward_uses_only_new_coverage_and_penalizes_duplicates() -> None:
    first = {"doc_id": "gold-1", "title": "First", "text": "one"}
    second = {"doc_id": "gold-2", "title": "Second", "text": "two"}
    noise = {"doc_id": "noise", "title": "Noise", "text": "irrelevant"}
    gold = (first, second)
    assert evidence_reward(
        [second], [second], gold, 0.2, previous=[first], duplicate_eta=0.2,
    ) == pytest.approx(0.5)
    assert evidence_reward(
        [first], [first], gold, 0.2, previous=[first], duplicate_eta=0.2,
    ) == pytest.approx(-0.2)
    assert evidence_reward(
        [noise], [noise], gold, 0.2, previous=[first], duplicate_eta=0.2,
    ) == pytest.approx(-0.2)


def test_validation_selection_score_adds_three_macro_components() -> None:
    quality = {
        "2wiki": {"answer_f1_mean": 0.5, "evidence_coverage_mean": 0.2},
        "musique": {"answer_f1_mean": 0.7, "evidence_coverage_mean": 0.4},
    }
    protocol = {
        "parse_failure_rate": 0.1,
        "protocol_by_dataset": {
            "2wiki": {"final_compliance_rate": 0.9},
            "musique": {"final_compliance_rate": 1.0},
        },
    }
    config = SimpleNamespace(
        validation_score_answer_weight=1.0,
        validation_score_evidence_weight=1.0,
        validation_score_format_weight=1.0,
        validation_score_parse_penalty=0.0,
    )
    score = MAPPOTrainer._validation_selection_score(
        quality, protocol, config,
    )
    assert score["answer_f1_macro"] == pytest.approx(0.6)
    assert score["evidence_coverage_macro"] == pytest.approx(0.3)
    assert score["format_compliance_macro"] == pytest.approx(0.95)
    assert score["validation_score"] == pytest.approx(1.85)


def test_outcome_only_rollout_has_zero_local_and_unchanged_terminal_reward() -> None:
    torch = pytest.importorskip("torch")
    responses = {
        AgentRole.QUERY: (
            '<query-retriever>{"sub_goal":"find answer","query":"Ada"}'
            '</query-retriever>'
        ),
        AgentRole.EVIDENCE: (
            '<update-evidence>{"selected_passage_ids":["P0"]}'
            '</update-evidence>'
        ),
        AgentRole.ANSWER: '<answer>{"can_answer":true,"answer":"Ada"}</answer>',
    }

    class Actor:
        def generate(self, role, prompt, **kwargs):
            del prompt, kwargs
            return responses[role], [1], [2], torch.tensor([0.0])

    class Critic:
        def __init__(self):
            self.torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    class Retrieval:
        def query(self, dataset, query):
            assert (dataset, query) == ("hotpotqa", "Ada")
            return {"passages": [{
                "passage_id": 0, "title": "Ada", "text": "Ada is the answer.",
            }]}

    config = parse_args([
        "--config", str(ROOT / "src/rl_v2/train_mappo_outcome_only.yml"),
    ])
    config.max_rounds = 1
    config.force_evidence_guided_decoding = False
    collector = RolloutCollector(
        actor=Actor(), critic=Critic(), retrieval=Retrieval(), config=config,
    )
    episode = collector.collect(RLSample(
        qid="q", dataset="hotpotqa", question="Who?", answer="Ada",
        answer_aliases=(),
        supporting_facts=({"title": "Ada", "text": "Ada is the answer."},),
    ))
    assert episode.global_reward == pytest.approx(2.5)
    assert len(episode.transitions) == 3
    assert all(item.local_reward == 0.0 for item in episode.transitions)
    assert all(item.team_reward == pytest.approx(2.5) for item in episode.transitions)
    assert all(item.reward == pytest.approx(2.5) for item in episode.transitions)


def test_outcome_only_invalid_action_has_no_local_penalty() -> None:
    torch = pytest.importorskip("torch")

    class Actor:
        def generate(self, role, prompt, **kwargs):
            del role, prompt, kwargs
            return "invalid", [1], [2], torch.tensor([0.0])

    class Critic:
        def __init__(self):
            self.torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    config = parse_args([
        "--config", str(ROOT / "src/rl_v2/train_mappo_outcome_only.yml"),
    ])
    collector = RolloutCollector(
        actor=Actor(), critic=Critic(), retrieval=None, config=config,
    )
    episode = Episode(qid="q", dataset="hotpotqa")
    parsed, transition = collector._act(
        episode, AgentRole.QUERY, RAGState(question="Who?"),
        observation=None, final_round=False,
    )
    assert parsed is None
    assert transition.valid is False
    assert transition.reward == 0.0


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


def test_update_accumulates_logical_groups_before_optimizer_step(tmp_path) -> None:
    torch = pytest.importorskip("torch")

    class Actor:
        def __init__(self):
            self.model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.model.weight)

        def score_batch(self, sequences):
            width = max(len(action) for _, action in sequences)
            value = self.model.weight.reshape(1, 1)
            logprobs = value.expand(len(sequences), width)
            return logprobs, torch.ones_like(logprobs)

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.zeros(()))

        def forward(self, states):
            return self.value.expand(len(states))

    actor = Actor()
    critic = Critic()
    transitions = []
    for role in AgentRole:
        for index in range(6):
            transitions.append(MAPPOTransition(
                role=role, round_index=index, prompt="p", prompt_ids=[1],
                action_ids=[2], old_token_logprobs=torch.zeros(1),
                central_state={}, next_central_state={}, advantage=1.0,
                return_=1.0,
                optimization_token_mask=(
                    [True] if role is AgentRole.EVIDENCE else None
                ),
            ))

    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.torch = torch
    trainer.actor = actor
    trainer.critic = critic
    trainer.actor_optimizer = torch.optim.SGD(actor.model.parameters(), lr=1.0e-3)
    trainer.critic_optimizer = torch.optim.SGD(critic.parameters(), lr=1.0e-3)
    trainer.global_step = 0
    trainer.run_dir = tmp_path
    trainer.reference_kl_controllers = {}
    trainer.reference_kl_recovery_steps = {role: 0 for role in AgentRole}
    trainer.config = SimpleNamespace(
        use_vllm_generation=False, ppo_old_logprob_source="local_actor",
        reference_kl_beta=0.0, normalize_advantages=False,
        minibatch_size=1, actor_minibatch_mode="role_balanced",
        actor_role_weights={}, gradient_diagnostics_steps=0,
        gradient_diagnostics_max_groups=1, ppo_epochs=1,
        gradient_accumulation_steps=4, entropy_coef=0.0,
        entropy_final_coef=0.0, entropy_anneal_start_step=0,
        entropy_anneal_end_step=0, clip_epsilon=0.2, target_kl=100.0,
        max_grad_norm=10.0, value_clip_epsilon=0.2,
        value_loss_coef=0.5,
    )
    metrics = trainer._update(transitions)
    assert metrics["gradient_accumulation_steps"] == 4.0
    assert metrics["logical_groups_applied"] == 6.0
    assert metrics["policy_updates_applied"] == 2.0
    assert metrics["kl_rejected_updates"] == 0.0
