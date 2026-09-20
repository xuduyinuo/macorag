import json
import os
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest
import torch

from .mappo import (
    AdaptiveReferenceKLController,
    clipped_value_loss,
    compute_gae,
    gradient_conflict_metrics,
    mappo_actor_loss,
    normalize_advantages_by_role,
)
from .data import load_samples, sample_stratum, split_train_validation
from .evaluation import align_evaluation_args
from .models import (
    FINAL_ANSWER_GUIDED_REGEX,
    RoleConditionedActor,
    evidence_action_layout,
    evidence_completion_budget,
    evidence_guided_regex,
)
from .protocol import ProtocolWindowMonitor, build_prompt, parse_action, protocol_metrics
from .rollout import (
    RolloutCollector, _merge_unique_evidence, _without_existing_evidence,
)
from .mappo_types import AgentRole, Episode, MAPPOTransition, RAGState, RLSample
from .trainer import MAPPOTrainer
from .vllm_client import VLLMClient


def _transition(role: AgentRole, reward: float, value: float, done: bool = False) -> MAPPOTransition:
    return MAPPOTransition(
        role=role, round_index=0, prompt="", prompt_ids=[1], action_ids=[2],
        old_token_logprobs=torch.tensor([-0.2]), central_state={},
        next_central_state={}, reward=reward, old_value=value, done=done,
    )


def test_gae_follows_each_agent_timeline() -> None:
    q0 = _transition(AgentRole.QUERY, 1.0, 0.5)
    a0 = _transition(AgentRole.ANSWER, 7.0, 1.0, True)
    q1 = _transition(AgentRole.QUERY, 2.0, 0.25, True)
    episode = Episode(qid="q", dataset="d", transitions=[q0, a0, q1])
    compute_gae(episode, gamma=1.0, gae_lambda=1.0)
    assert q1.advantage == 1.75
    assert q0.advantage == 2.5
    assert a0.advantage == 6.0


def test_actor_clipping_and_mask() -> None:
    old = torch.tensor([[-1.0, -1.0]])
    new = torch.tensor([[-0.5, 9.0]], requires_grad=True)
    loss, stats = mappo_actor_loss(new, old, torch.tensor([1.0]), torch.tensor([[1, 0]]), 0.2)
    assert torch.isclose(loss, torch.tensor(-1.2))
    assert stats["clip_fraction"].item() == 1.0
    loss.backward()
    assert new.grad is not None


def test_gradient_conflict_metrics_reports_cancellation_and_alignment() -> None:
    metrics = gradient_conflict_metrics({
        "answer_generator": [torch.tensor([1.0, 0.0]), None],
        "evidence_updater": [torch.tensor([-1.0, 0.0]), torch.tensor([0.0])],
        "query_retriever": [torch.tensor([0.0, 1.0]), torch.tensor([0.0])],
    })
    assert metrics["pairwise_cosine"][
        "answer_generator__evidence_updater"
    ] == pytest.approx(-1.0)
    assert metrics["pairwise_cosine"][
        "answer_generator__query_retriever"
    ] == pytest.approx(0.0)
    assert metrics["conflicting_pairs"] == 1
    assert metrics["conflict_rate"] == pytest.approx(1.0 / 3.0)
    assert metrics["combined_gradient_norm"] == pytest.approx(1.0)
    assert metrics["gradient_cancellation_ratio"] == pytest.approx(2.0 / 3.0)


def test_mappo_evaluation_args_follow_training_rollout_contract() -> None:
    args = SimpleNamespace(
        model_path="model/Qwen2.5-7B-Instruct",
        max_rounds=99,
        max_prompt_length=4096,
        max_completion_length=256,
        temperature=0.7,
        top_p=0.8,
        retrieval_backend="e5_faiss",
        retrieval_embedding_model="wrong",
        retrieval_device="cuda",
        retrieval_max_length=123,
        retrieval_batch_size=1,
        retrieval_top_k=9,
    )
    config = SimpleNamespace(
        model_path="model/Qwen2.5-7B-Instruct",
        max_rounds=4,
        max_prompt_length=1024,
        max_completion_length=128,
        validation_temperature=0.0,
        top_p=0.95,
        retrieval_backend="e5_faiss",
        retrieval_embedding_model="intfloat/e5-base-v2",
        retrieval_device="cpu",
        retrieval_max_length=512,
        retrieval_batch_size=32,
        retrieval_top_k=5,
    )
    align_evaluation_args(args, config)
    assert (args.max_rounds, args.max_prompt_length, args.max_completion_length) == (
        4, 1024, 128,
    )
    assert args.temperature == 0.0
    assert args.retrieval_embedding_model == "intfloat/e5-base-v2"
    assert args.retrieval_top_k == 5


def test_value_clipping_uses_larger_error() -> None:
    loss = clipped_value_loss(torch.tensor([2.0]), torch.tensor([0.0]), torch.tensor([1.0]), 0.2)
    assert torch.isclose(loss, torch.tensor(0.5))


def test_advantages_are_normalized_per_role_without_erasing_singletons() -> None:
    query_low = _transition(AgentRole.QUERY, 0.0, 0.0)
    query_high = _transition(AgentRole.QUERY, 0.0, 0.0)
    answer_only = _transition(AgentRole.ANSWER, 0.0, 0.0)
    query_low.advantage = 1.0
    query_high.advantage = 3.0
    answer_only.advantage = 7.0
    normalize_advantages_by_role([query_low, query_high, answer_only])
    assert query_low.advantage == -1.0
    assert query_high.advantage == 1.0
    assert answer_only.advantage == 7.0


def test_adaptive_reference_kl_controller_updates_and_restores_state() -> None:
    controller = AdaptiveReferenceKLController(
        beta=0.01, target=0.03, beta_min=0.005, beta_max=0.2,
        ema_decay=0.0, controller_rate=1.0,
        emergency_threshold=0.08, emergency_patience=2,
    )
    first = controller.update(0.09)
    assert first["reference_kl_beta_next"] > first["reference_kl_beta"]
    assert first["reference_kl_emergency_triggered"] == 0.0
    second = controller.update(0.09)
    assert second["reference_kl_emergency_triggered"] == 1.0
    restored = AdaptiveReferenceKLController(
        beta=0.01, target=0.03, beta_min=0.005, beta_max=0.2,
    )
    restored.load_state_dict(controller.state_dict())
    assert restored.state_dict() == controller.state_dict()
    high_beta = restored.beta
    restored.ema_decay = 0.0
    restored.controller_rate = 1.0
    restored.update(0.0)
    assert restored.beta < high_beta


def test_reference_kl_emergency_requires_current_and_ema_violation() -> None:
    controller = AdaptiveReferenceKLController(
        beta=0.01, target=0.03, beta_min=0.005, beta_max=0.2,
        ema_decay=0.9, emergency_threshold=0.08, emergency_patience=2,
    )
    first = controller.update(0.2)
    assert first["reference_kl_emergency_count"] == 1.0
    recovered = controller.update(0.0)
    assert recovered["reference_kl_ema"] > 0.08
    assert recovered["reference_kl_emergency_count"] == 0.0
    assert recovered["reference_kl_emergency_triggered"] == 0.0


def test_evidence_guidance_allows_only_unique_observed_passage_ids() -> None:
    pattern = evidence_guided_regex([7, 2, 7])
    assert re.fullmatch(
        pattern,
        '<update-evidence>{"selected_passage_ids":["P2","P7"],"rationale":"both support"}</update-evidence>',
    )
    assert not re.fullmatch(
        pattern,
        '<update-evidence>{"selected_passage_ids":[],"rationale":"none"}</update-evidence>',
    )
    assert not re.fullmatch(
        pattern,
        '<update-evidence>{"selected_passage_ids":["P7","P7"],"rationale":"duplicate"}</update-evidence>',
    )
    assert not re.fullmatch(
        pattern,
        '<update-evidence>{"selected_passage_ids":["P9"],"rationale":"not observed"}</update-evidence>',
    )
    assert not re.fullmatch(
        pattern,
        '<update-evidence>{"selected_passage_ids":["P2"],"rationale":"'
        + "x" * 65
        + '"}</update-evidence>',
    )
    empty_observation_pattern = evidence_guided_regex([])
    assert re.fullmatch(
        empty_observation_pattern,
        '<update-evidence>{"selected_passage_ids":[],"rationale":"none"}</update-evidence>',
    )


def test_evidence_pointer_layout_optimizes_only_constrained_choice_tokens() -> None:
    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(item) for item in text]

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens
            result = {"input_ids": self.encode(text)}
            if return_offsets_mapping:
                result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
            return result

    response = (
        '<update-evidence>{"selected_passage_ids":["P0","P2"],'
        '"rationale":"both support"}</update-evidence>'
    )
    tokenizer = CharacterTokenizer()
    action_ids = tokenizer.encode(response)
    constraints, optimize, segments = evidence_action_layout(
        tokenizer, response=response, action_ids=action_ids,
        passage_ids=[0, 1, 2], min_selected=1,
    )
    assert any(optimize)
    assert all(constraints[index] is not None for index, active in enumerate(optimize) if active)
    assert not any(active for active, segment in zip(optimize, segments) if segment == "rationale")
    assert "selection" in segments and "rationale" in segments and "format" in segments


def test_evidence_pointer_layout_tokenizes_complete_response_at_bpe_boundary() -> None:
    class BoundaryMergingTokenizer:
        """Mimic a BPE token that crosses the header/rationale boundary."""

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            marker = '"rationale":"s'
            output = []
            index = 0
            while index < len(text):
                if text.startswith(marker, index):
                    output.extend(ord(item) for item in marker[:-2])
                    output.append(100_001)
                    index += len(marker)
                else:
                    output.append(ord(text[index]))
                    index += 1
            return output

        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise NotImplementedError

    response = (
        '<update-evidence>{"selected_passage_ids":["P1"],'
        '"rationale":"support"}</update-evidence>'
    )
    tokenizer = BoundaryMergingTokenizer()
    action_ids = tokenizer.encode(response)
    constraints, optimize, segments = evidence_action_layout(
        tokenizer, response=response, action_ids=action_ids,
        passage_ids=[0, 1, 2], min_selected=1,
    )
    assert any(optimize)
    assert len(constraints) == len(action_ids)
    assert not any(active for active, segment in zip(optimize, segments) if segment == "rationale")


def test_evidence_pointer_layout_accepts_trailing_special_stop_token() -> None:
    class CharacterTokenizer:
        all_special_ids = [0]

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(item) for item in text]

        def __call__(self, *args, **kwargs):
            del args, kwargs
            raise NotImplementedError

    response = (
        '<update-evidence>{"selected_passage_ids":["P0"],'
        '"rationale":"support"}</update-evidence>'
    )
    tokenizer = CharacterTokenizer()
    action_ids = [*tokenizer.encode(response), 0]
    constraints, optimize, segments = evidence_action_layout(
        tokenizer, response=response, action_ids=action_ids,
        passage_ids=[0, 1], min_selected=1,
    )
    assert len(constraints) == len(action_ids)
    assert constraints[-1] is None and not optimize[-1] and segments[-1] == "format"


def test_evidence_pointer_parser_normalizes_labels_to_local_ids() -> None:
    parsed = parse_action(
        '<update-evidence>{"selected_passage_ids":["P1","P4"],'
        '"rationale":"support"}</update-evidence>',
        AgentRole.EVIDENCE,
    )
    assert parsed["selected_passage_ids"] == [1, 4]


def test_evidence_completion_budget_rejects_token_overflow() -> None:
    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return list(text)

    with pytest.raises(ValueError, match="token budget is too small"):
        evidence_completion_budget(
            CharacterTokenizer(), passage_ids=list(range(5)),
            rationale_max_chars=64, max_completion_length=32,
        )


def test_role_balanced_minibatches_group_one_microbatch_per_role() -> None:
    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.config = SimpleNamespace(minibatch_size=1, actor_minibatch_mode="role_balanced")
    transitions = [
        *[_transition(AgentRole.QUERY, 0.0, 0.0) for _ in range(2)],
        _transition(AgentRole.EVIDENCE, 0.0, 0.0),
        *[_transition(AgentRole.ANSWER, 0.0, 0.0) for _ in range(3)],
    ]
    groups = trainer._actor_minibatch_groups(transitions)
    assert len(groups) == 3
    assert {batch[0].role for batch in groups[0]} == set(AgentRole)
    assert all(len({item.role for item in batch}) == 1 for group in groups for batch in group)
    assert sum(len(batch) for group in groups for batch in group) == len(transitions)


def test_protocol_is_strict() -> None:
    parsed = parse_action(
        '<query-retriever>{"sub_goal":"birth place","query":"Ada Lovelace born"}</query-retriever>',
        AgentRole.QUERY,
    )
    assert parsed["query"] == "Ada Lovelace born"


def test_final_round_prompt_has_consistent_forced_answer_example() -> None:
    from .mappo_types import RAGState
    prompt = build_prompt(
        AgentRole.ANSWER, question="q", state=RAGState(question="q"), final_round=True,
    )
    assert '"can_answer":true' in prompt
    assert "fallback_guess:" in prompt
    assert '"can_answer":false' not in prompt


def test_non_final_answer_prompt_has_positive_and_negative_few_shots() -> None:
    prompt = build_prompt(
        AgentRole.ANSWER, question="q", state=RAGState(question="q"), final_round=False,
    )
    assert "evidence is insufficient" in prompt
    assert "evidence is sufficient" in prompt
    assert '"can_answer":false' in prompt
    assert '"can_answer":true' in prompt
    assert prompt.count("<answer>{") == 2
    assert prompt.index("<output-contract>") > prompt.index("</state>")
    assert prompt.index("Example when evidence is insufficient") > prompt.index("<output-contract>")


def test_prompt_compaction_bounds_state_and_preserves_multi_hop_edges() -> None:
    evidence = [
        {"passage_id": index, "title": f"title-{index}", "text": "Z" * 2000}
        for index in range(20)
    ]
    history = [
        {"query": f"query-{index}-" + "Q" * 500, "sub_goal": "S" * 500,
         "passage_ids": list(range(20))}
        for index in range(10)
    ]
    prompt = build_prompt(
        AgentRole.ANSWER,
        question="Which answer?",
        state=RAGState(
            question="Which answer?", evidence=evidence,
            retrieval_history=history, round_index=3,
        ),
        max_evidence_items=4,
        max_history_items=2,
        evidence_text_chars=80,
    )
    assert '"evidence_count":20' in prompt
    assert '"retrieval_count":10' in prompt
    assert all(f'"passage_id":{index}' in prompt for index in (0, 1, 18, 19))
    assert '"passage_id":2' not in prompt
    assert "Z" * 2000 not in prompt
    assert "...[truncated]" in prompt
    assert prompt.index("<output-contract>") > prompt.index("</state>")
    assert prompt.endswith("</output-contract>")

    one_item_prompt = build_prompt(
        AgentRole.ANSWER,
        question="q",
        state=RAGState(question="q", evidence=evidence),
        max_evidence_items=1,
    )
    assert '"passage_id":0' in one_item_prompt
    assert one_item_prompt.count('"passage_id":') == 1


def test_evidence_prompt_compacts_observation_but_keeps_ids() -> None:
    observation = {"query": "q", "passages": [
        {"passage_id": index, "title": f"t-{index}", "text": "X" * 1000}
        for index in range(12)
    ]}
    prompt = build_prompt(
        AgentRole.EVIDENCE,
        question="q",
        state=RAGState(question="q"),
        observation=observation,
        max_evidence_items=3,
        observation_text_chars=64,
    )
    assert "<passage_count>12</passage_count>" in prompt
    assert all(f"<P{index}>" in prompt for index in (0, 1, 2))
    assert "<P3>" not in prompt
    assert "X" * 1000 not in prompt
    assert '"selected_passage_ids":["P0"]' in prompt
    assert "an empty selection is not allowed" in prompt
    assert "within 64 characters" in prompt
    assert prompt.index("<output-contract>") > prompt.index("</observation>")


def test_overflow_keeps_system_prefix_and_tail_output_contract() -> None:
    class Model(torch.nn.Module):
        def forward(self, input_ids, **kwargs):
            del kwargs
            return SimpleNamespace(logits=torch.zeros((*input_ids.shape, 8)))

    class CharacterTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, add_generation_prompt, tokenize):
            del tokenize
            rendered = "".join(
                f"<{message['role']}>{message['content']}</{message['role']}>"
                for message in messages
            )
            if add_generation_prompt:
                rendered += "<assistant>"
            return [ord(char) for char in rendered]

        def decode(self, ids, skip_special_tokens=True):
            del skip_special_tokens
            return "".join(chr(item) for item in ids)

    tokenizer = CharacterTokenizer()
    actor = RoleConditionedActor(
        Model(), tokenizer, max_prompt_length=900, max_completion_length=128,
        temperature=0.8, top_p=0.95, top_k=5, device=torch.device("cpu"),
    )
    prompt = build_prompt(
        AgentRole.ANSWER,
        question="q",
        state=RAGState(
            question="q",
            evidence=[{"passage_id": 0, "title": "t", "text": "Y" * 5000}],
        ),
        evidence_text_chars=5000,
    )
    encoded = actor.encode_prompt(AgentRole.ANSWER, prompt)
    decoded = tokenizer.decode(encoded)
    assert len(encoded) == 900
    assert decoded.startswith("<system>You are the answer-generator role")
    assert "<output-contract>" in decoded
    assert "Example when evidence is insufficient" in decoded
    assert "Example when evidence is sufficient" in decoded
    assert decoded.endswith("</output-contract></user><assistant>")


def test_stratified_validation_split_is_deterministic_and_disjoint() -> None:
    samples = [
        RLSample(
            qid=f"{dataset}-{index}", dataset=dataset, question="q", answer="a",
            answer_aliases=(), supporting_facts=(),
        )
        for dataset in ("a", "b") for index in range(5)
    ]
    first = split_train_validation(samples, validation_ratio=0.4, seed=42)
    second = split_train_validation(samples, validation_ratio=0.4, seed=42)
    assert [[item.qid for item in part] for part in first] == [[item.qid for item in part] for part in second]
    train, validation = first
    assert len(train) == 6 and len(validation) == 4
    assert {(x.dataset, x.qid) for x in train}.isdisjoint(
        {(x.dataset, x.qid) for x in validation}
    )


def test_validation_split_is_stratified_by_musique_hop() -> None:
    samples = [
        RLSample(
            qid=f"{hop}__{index}", dataset="musique", question="q", answer="a",
            answer_aliases=(), supporting_facts=(),
        )
        for hop in ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3")
        for index in range(10)
    ]
    train, validation = split_train_validation(
        samples, validation_ratio=0.1, validation_samples_per_dataset=12, seed=42,
    )
    assert len(train) == 48 and len(validation) == 12
    counts = {
        hop: sum(sample_stratum(item) == hop for item in validation)
        for hop in ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3")
    }
    assert counts == {hop: 2 for hop in counts}


def test_musique_rare_hop_sampling_is_weighted_without_duplicate_qids(tmp_path) -> None:
    data_dir = tmp_path / "musique"
    data_dir.mkdir()
    rows = []
    for hop, count in (("2hop", 80), ("4hop2", 10), ("4hop3", 10)):
        for index in range(count):
            rows.append({
                "qid": f"{hop}__{index}", "dataset": "musique",
                "question": "q", "answer": "a", "supporting_facts": [],
            })
    (data_dir / "musique_train.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
    )
    natural = load_samples(
        tmp_path, max_per_dataset=50, max_total=None, seed=42,
        musique_rare_hop_oversample_factor=1.0,
    )
    weighted = load_samples(
        tmp_path, max_per_dataset=50, max_total=None, seed=42,
        musique_rare_hop_oversample_factor=2.0,
    )
    natural_rare = sum(sample_stratum(item) in {"4hop2", "4hop3"} for item in natural)
    weighted_rare = sum(sample_stratum(item) in {"4hop2", "4hop3"} for item in weighted)
    assert weighted_rare > natural_rare
    assert len({item.qid for item in weighted}) == 50


def test_protocol_metrics_and_window_monitor_detect_failures() -> None:
    valid = Episode(qid="ok", dataset="d", final_answer="answer")
    invalid = Episode(
        qid="bad", dataset="d", parse_errors=["Missing required tag: answer"],
    )
    metrics = protocol_metrics(
        [valid, invalid], max_parse_failure_rate=0.1,
        max_missing_answer_tag_rate=0.1, min_final_compliance_rate=0.9,
    )
    assert metrics["parse_failure_rate"] == 0.5
    assert metrics["missing_answer_tag_rate"] == 0.5
    assert metrics["checkpoint_eligible"] is False
    monitor = ProtocolWindowMonitor(window_size=2, max_parse_failure_rate=0.1, bad_windows_to_warn=1)
    assert monitor.add([valid, invalid])["should_warn"] is True


def test_validation_protocol_uses_overall_gate_and_keeps_dataset_diagnostics() -> None:
    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.config = SimpleNamespace(
        max_protocol_parse_failure_rate=0.01,
        max_validation_missing_answer_tag_rate=0.005,
        min_validation_final_compliance_rate=0.99,
    )
    episodes = [
        Episode(qid=f"a-{index}", dataset="a", final_answer="answer")
        for index in range(49)
    ] + [
        Episode(
            qid="a-invalid", dataset="a", final_answer="answer",
            parse_errors=["Invalid JSON"],
        )
    ] + [
        Episode(qid=f"b-{index}", dataset="b", final_answer="answer")
        for index in range(50)
    ]
    metrics = trainer._validation_protocol(episodes)
    assert metrics["parse_failure_rate"] == 0.01
    assert metrics["protocol_by_dataset"]["a"]["parse_failure_rate"] == 0.02
    assert metrics["protocol_by_dataset"]["a"]["checkpoint_eligible"] is False
    assert metrics["protocol_by_dataset"]["b"]["checkpoint_eligible"] is True
    assert metrics["checkpoint_eligible"] is True
    assert metrics["checkpoint_eligibility_scope"] == "overall"


def test_rollout_assigns_local_and_team_rewards_to_all_agents() -> None:
    class Actor:
        responses = iter([
            '<query-retriever>{"sub_goal":"birth","query":"Ada birth"}</query-retriever>',
            '<update-evidence>{"selected_passage_ids":[0],"rationale":"match"}</update-evidence>',
            '<answer>{"can_answer":true,"answer":"London","rationale":"supported"}</answer>',
        ])

        def generate(self, role, prompt):
            response = next(self.responses)
            return response, [1], [2], torch.tensor([-0.1])

    class Critic:
        torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    class Retrieval:
        def query(self, dataset, query):
            return {"query": query, "passages": [{
                "passage_id": 0, "title": "Ada Lovelace", "text": "Born in London",
            }]}

    class Config:
        max_rounds = 2
        eta_query = 0.2
        eta_evidence = 0.2
        answer_local_reward_weight = 1.0
        omega_answer = 1.0
        omega_evidence = 1.0
        terminal_reward_weight = 1.0

    sample = RLSample(
        qid="q", dataset="d", question="Where was Ada born?", answer="London",
        answer_aliases=(), supporting_facts=({"title": "Ada Lovelace", "text": "Born in London"},),
    )
    episode = RolloutCollector(
        actor=Actor(), critic=Critic(), retrieval=Retrieval(), config=Config(),
    ).collect(sample)
    assert episode.global_reward == 2.0
    assert len(episode.transitions) == 3
    assert all(item.done for item in episode.transitions)
    assert all(item.parsed_action is not None for item in episode.transitions)
    assert all(item.team_reward == 2.0 for item in episode.transitions)
    assert [item.reward for item in episode.transitions] == [3.0, 3.0, 3.0]


def test_rollout_passes_dynamic_guidance_to_evidence_actor() -> None:
    class Actor:
        guidance = None

        def generate(self, role, prompt, *, guided_regex=None, max_tokens=None):
            del prompt
            assert role is AgentRole.EVIDENCE
            assert max_tokens == 192
            self.guidance = guided_regex
            response = (
                '<update-evidence>{"selected_passage_ids":["P3"],'
                '"rationale":"supported"}</update-evidence>'
            )
            return response, [1], [2], torch.tensor([-0.1])

    class Critic:
        torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    actor = Actor()
    collector = RolloutCollector(
        actor=actor, critic=Critic(), retrieval=None,
        config=SimpleNamespace(
            max_rounds=2, force_evidence_guided_decoding=True,
            format_reward_weight=0.1,
        ),
    )
    parsed, transition = collector._act(
        Episode(qid="q", dataset="d"), AgentRole.EVIDENCE,
        RAGState(question="q"),
        observation={"passages": [{"passage_id": 3}, {"passage_id": 8}]},
        final_round=False,
    )
    assert parsed is not None and transition.valid
    assert actor.guidance is not None
    assert re.fullmatch(actor.guidance, transition.response)
    assert "9" not in actor.guidance


def test_rollout_uses_answer_specific_completion_budget() -> None:
    class Actor:
        seen_max_tokens = None

        def generate(self, role, prompt, *, max_tokens=None):
            del prompt
            assert role is AgentRole.ANSWER
            self.seen_max_tokens = max_tokens
            response = (
                '<answer>{"can_answer":true,"answer":"Ada",'
                '"rationale":"supported"}</answer>'
            )
            return response, [1], [2], torch.tensor([-0.1])

    class Critic:
        torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    actor = Actor()
    collector = RolloutCollector(
        actor=actor, critic=Critic(), retrieval=None,
        config=SimpleNamespace(
            max_rounds=2, force_evidence_guided_decoding=False,
            answer_max_completion_length=192, format_reward_weight=0.1,
        ),
    )
    parsed, transition = collector._act(
        Episode(qid="q", dataset="d"), AgentRole.ANSWER,
        RAGState(question="q"), observation=None, final_round=True,
    )
    assert parsed["answer"] == "Ada" and transition.valid
    assert actor.seen_max_tokens == 192


def test_cross_round_evidence_candidates_and_state_are_deduplicated() -> None:
    existing = [{
        "dataset": "d", "chunk_id": "chunk-a", "passage_id": 0,
        "title": "A", "text": "existing text",
    }]
    retrieved = [
        {
            "dataset": "d", "chunk_id": "chunk-a", "passage_id": 4,
            "title": "A", "text": "existing text",
        },
        {
            "dataset": "d", "chunk_id": "chunk-b", "passage_id": 1,
            "title": "B", "text": "new text",
        },
        {
            "dataset": "d", "chunk_id": "chunk-b", "passage_id": 2,
            "title": "B duplicate", "text": "new text",
        },
    ]
    candidates = _without_existing_evidence(retrieved, existing)
    assert [item["chunk_id"] for item in candidates] == ["chunk-b"]
    merged = _merge_unique_evidence(existing, retrieved)
    assert [item["chunk_id"] for item in merged] == ["chunk-a", "chunk-b"]


def test_recoverable_evidence_json_is_repaired_and_rescored() -> None:
    class Actor:
        def generate(self, role, prompt, *, guided_regex=None, max_tokens=None):
            del role, prompt, guided_regex, max_tokens
            response = (
                '<update-evidence>{"selected_passage_ids":["P3"],'
                '"rationale":"Dallol\\\'s record"}</update-evidence>'
            )
            return response, [1], [2], torch.tensor([-0.1])

        def rescore_response(self, prompt_ids, response):
            assert prompt_ids == [1]
            assert "\\'" not in response and "Dallol's" in response
            return [9], torch.tensor([-0.2])

    class Critic:
        torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    collector = RolloutCollector(
        actor=Actor(), critic=Critic(), retrieval=None,
        config=SimpleNamespace(
            max_rounds=2, force_evidence_guided_decoding=True,
            format_reward_weight=0.1,
        ),
    )
    parsed, transition = collector._act(
        Episode(qid="q", dataset="d"), AgentRole.EVIDENCE,
        RAGState(question="q"),
        observation={"passages": [{"passage_id": 3}]}, final_round=False,
    )
    assert parsed["rationale"] == "Dallol's record"
    assert transition.valid and transition.format_recovery == "repaired"
    assert transition.action_ids == [9]


def test_noncanonical_guided_evidence_tokens_are_canonicalized() -> None:
    class Tokenizer:
        all_special_ids = [0]

        def encode(self, text, add_special_tokens=False):
            del add_special_tokens
            return [ord(item) for item in text]

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            del add_special_tokens
            result = {"input_ids": self.encode(text)}
            if return_offsets_mapping:
                result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
            return result

    class Actor:
        tokenizer = Tokenizer()

        def generate(self, role, prompt, *, guided_regex=None, max_tokens=None):
            del role, prompt, guided_regex, max_tokens
            response = (
                '<update-evidence>{"selected_passage_ids":["P3"],'
                '"rationale":"supported"}</update-evidence>'
            )
            # Same legal text was produced by a non-canonical guided-decoding
            # token path, represented minimally here by unrelated IDs.
            return response, [1], [999, 998], torch.tensor([-0.1, -0.2])

        def rescore_response(self, prompt_ids, response):
            assert prompt_ids == [1]
            canonical = self.tokenizer.encode(response)
            return canonical, torch.full((len(canonical),), -0.3)

    class Critic:
        torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    collector = RolloutCollector(
        actor=Actor(), critic=Critic(), retrieval=None,
        config=SimpleNamespace(
            max_rounds=2, force_evidence_guided_decoding=True,
            evidence_min_selected_passages=1, format_reward_weight=0.1,
            evidence_max_completion_length=512,
            ppo_old_logprob_source="local_actor",
        ),
    )
    parsed, transition = collector._act(
        Episode(qid="q", dataset="d"), AgentRole.EVIDENCE,
        RAGState(question="q"),
        observation={"passages": [{"passage_id": 3}, {"passage_id": 8}]},
        final_round=False,
    )
    assert parsed["selected_passage_ids"] == [3]
    assert transition.valid
    assert transition.tokenization_recovery == "canonicalized"
    assert transition.action_ids == Actor.tokenizer.encode(transition.response)
    assert torch.count_nonzero(transition.old_token_logprobs) == 0
    assert transition.token_constraints is not None
    assert any(transition.optimization_token_mask)


def test_unrepairable_format_error_is_retried_once() -> None:
    class Actor:
        responses = iter([
            "missing tag",
            '<query-retriever>{"sub_goal":"birth","query":"Ada birth"}</query-retriever>',
        ])

        def generate(self, role, prompt):
            del role, prompt
            return next(self.responses), [1], [2], torch.tensor([-0.1])

    class Critic:
        torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    collector = RolloutCollector(
        actor=Actor(), critic=Critic(), retrieval=None,
        config=SimpleNamespace(max_rounds=2, format_reward_weight=0.1),
    )
    parsed, transition = collector._act(
        Episode(qid="q", dataset="d"), AgentRole.QUERY,
        RAGState(question="q"), observation=None, final_round=False,
    )
    assert parsed["query"] == "Ada birth"
    assert transition.valid and transition.format_recovery == "retried"


def test_rollout_adds_format_reward_to_valid_actions() -> None:
    class Actor:
        responses = iter([
            '<query-retriever>{"sub_goal":"birth","query":"Ada birth"}</query-retriever>',
            '<update-evidence>{"selected_passage_ids":[0],"rationale":"match"}</update-evidence>',
            '<answer>{"can_answer":true,"answer":"London","rationale":"supported"}</answer>',
        ])
        def generate(self, role, prompt):
            return next(self.responses), [1], [2], torch.tensor([-0.1])
    class Critic:
        torch = torch
        def __call__(self, states): return torch.zeros(len(states))
    class Retrieval:
        def query(self, dataset, query):
            return {"query": query, "passages": [{"passage_id": 0, "title": "Ada", "text": "London"}]}
    config = SimpleNamespace(
        max_rounds=1, eta_query=0.2, eta_evidence=0.2,
        answer_local_reward_weight=1.0, omega_answer=1.0, omega_evidence=1.0,
        terminal_reward_weight=1.0, format_reward_weight=0.2,
        invalid_action_penalty=-1.0,
    )
    sample = RLSample(
        qid="q", dataset="d", question="Where?", answer="London", answer_aliases=(),
        supporting_facts=({"title": "Ada", "text": "London"},),
    )
    episode = RolloutCollector(actor=Actor(), critic=Critic(), retrieval=Retrieval(), config=config).collect(sample)
    assert [round(item.reward, 6) for item in episode.transitions] == [3.2, 3.2, 3.2]


def test_final_refusal_is_penalized_and_cannot_collect_terminal_evidence_reward() -> None:
    class Actor:
        responses = iter([
            '<query-retriever>{"sub_goal":"birth","query":"Ada birth"}</query-retriever>',
            '<update-evidence>{"selected_passage_ids":[0],"rationale":"match"}</update-evidence>',
            '<answer>{"can_answer":false,"answer":null,"rationale":"wait"}</answer>',
        ])

        def generate(self, role, prompt):
            return next(self.responses), [1], [2], torch.tensor([-0.1])

    class Critic:
        torch = torch

        def __call__(self, states):
            return torch.zeros(len(states))

    class Retrieval:
        def query(self, dataset, query):
            return {"query": query, "passages": [{
                "passage_id": 0, "title": "Ada", "text": "Born in London",
            }]}

    config = SimpleNamespace(
        max_rounds=1, eta_query=0.2, eta_evidence=0.2,
        answer_local_reward_weight=1.0, omega_answer=1.0, omega_evidence=1.0,
        terminal_reward_weight=1.0, format_reward_weight=0.1,
        invalid_action_penalty=-1.0, final_answer_invalid_penalty=-3.0,
        non_final_wait_reward=0.2, final_answer_bonus=1.0,
        gate_terminal_evidence_on_valid_answer=True,
    )
    sample = RLSample(
        qid="q", dataset="d", question="Where?", answer="London",
        answer_aliases=(), supporting_facts=({"title": "Ada", "text": "Born in London"},),
    )
    episode = RolloutCollector(
        actor=Actor(), critic=Critic(), retrieval=Retrieval(), config=config,
    ).collect(sample)
    assert episode.final_answer is None
    assert episode.evidence_coverage == 1.0
    assert episode.global_reward == 0.0
    assert episode.transitions[-1].valid is False
    assert episode.transitions[-1].reward == -3.0


def test_entropy_coefficient_anneals_to_zero() -> None:
    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.config = SimpleNamespace(
        entropy_coef=0.002, entropy_final_coef=0.0,
        entropy_anneal_start_step=100, entropy_anneal_end_step=150,
    )
    trainer.global_step = 50
    assert trainer._entropy_coefficient() == 0.002
    trainer.global_step = 125
    assert trainer._entropy_coefficient() == 0.001
    trainer.global_step = 150
    assert trainer._entropy_coefficient() == 0.0


def test_actor_scores_only_required_tail_logits() -> None:
    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
            del attention_mask, use_cache
            logits = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:, :]
            return SimpleNamespace(logits=logits.requires_grad_())

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 7

    actor = RoleConditionedActor(
        Model(), Tokenizer(), max_prompt_length=8, max_completion_length=2,
        temperature=0.8, top_p=0.95, top_k=5, device=torch.device("cpu"),
    )
    logp, entropy = actor.score_batch([([1, 2], [3, 4])])
    assert logp.shape == entropy.shape == (1, 2)
    assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()


def test_actor_constrained_scoring_renormalizes_same_legal_token_set() -> None:
    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
            del attention_mask, use_cache
            logits = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:, :]
            return SimpleNamespace(logits=logits.requires_grad_())

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 7

    actor = RoleConditionedActor(
        Model(), Tokenizer(), max_prompt_length=8, max_completion_length=2,
        temperature=0.8, top_p=0.95, top_k=5, device=torch.device("cpu"),
    )
    constrained, _, raw = actor.score_batch_detailed(
        [([1, 2], [3, 4])], token_constraints=[[None, [4]]],
    )
    assert constrained[0, 1].item() == 0.0
    assert raw[0, 1].item() < 0.0


def test_actor_no_grad_scoring_restores_training_mode() -> None:
    class Model(torch.nn.Module):
        def forward(self, input_ids, attention_mask, use_cache, logits_to_keep=0):
            del attention_mask, use_cache
            logits = torch.nn.functional.one_hot(input_ids, num_classes=8).float()
            if logits_to_keep:
                logits = logits[:, -logits_to_keep:, :]
            return SimpleNamespace(logits=logits)

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 7

    model = Model()
    model.train()
    actor = RoleConditionedActor(
        model, Tokenizer(), max_prompt_length=8, max_completion_length=2,
        temperature=0.8, top_p=0.95, top_k=5, device=torch.device("cpu"),
    )
    logp, entropy = actor.score_batch_no_grad([([1, 2], [3, 4])])
    assert model.training is True
    assert not logp.requires_grad and not entropy.requires_grad


def test_trainer_replaces_vllm_logprobs_with_local_actor_scores() -> None:
    transition = _transition(AgentRole.QUERY, reward=0.0, value=0.0)
    transition.old_token_logprobs = torch.tensor([-4.0])

    class Actor:
        def score_batch_no_grad(self, sequences):
            assert sequences == [([1], [2])]
            return torch.tensor([[-0.25]]), torch.tensor([[0.5]])

    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.torch = torch
    trainer.actor = Actor()
    trainer.config = SimpleNamespace(
        use_vllm_generation=True, ppo_old_logprob_source="local_actor",
    )
    stats = trainer._align_old_logprobs([transition])
    assert torch.equal(transition.old_token_logprobs, torch.tensor([-0.25]))
    assert stats["old_logprobs_recomputed"] == 1.0
    assert stats["behavior_logprob_abs_diff"] == 3.75
    assert stats["behavior_approx_kl"] > 0.0


def test_trainer_records_fixed_sft_reference_logprobs() -> None:
    transition = _transition(AgentRole.QUERY, reward=0.0, value=0.0)

    class Actor:
        def score_reference_batch_no_grad(self, sequences):
            assert sequences == [([1], [2])]
            return torch.tensor([[-0.75]]), torch.tensor([[0.0]])

    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.torch = torch
    trainer.actor = Actor()
    trainer.config = SimpleNamespace(
        use_vllm_generation=False, ppo_old_logprob_source="local_actor",
        reference_kl_beta=0.01,
    )
    stats = trainer._align_old_logprobs([transition])
    assert torch.equal(transition.reference_token_logprobs, torch.tensor([-0.75]))
    assert stats["reference_logprobs_computed"] == 1.0


def test_kl_guard_rejects_update_before_parameter_mutation(tmp_path) -> None:
    class Actor:
        device = torch.device("cpu")

        def __init__(self):
            self.model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.model.weight)

        def score_batch(self, sequences):
            rows = len(sequences)
            logp = self.model.weight.reshape(1, 1).expand(rows, 1)
            return logp, torch.zeros_like(logp)

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, states):
            return self.value.expand(len(states))

    actor = Actor()
    critic = Critic()
    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.torch = torch
    trainer.actor = actor
    trainer.critic = critic
    trainer.run_dir = tmp_path
    trainer.global_step = 0
    trainer.config = SimpleNamespace(
        use_vllm_generation=False, ppo_old_logprob_source="local_actor",
        normalize_advantages=False, ppo_epochs=1, minibatch_size=1,
        clip_epsilon=0.2, entropy_coef=0.0, target_kl=0.02,
        max_grad_norm=1.0, value_clip_epsilon=0.2, value_loss_coef=0.5,
    )
    trainer.actor_optimizer = torch.optim.AdamW(actor.model.parameters(), lr=1.0)
    trainer.critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=1.0)
    transition = _transition(AgentRole.QUERY, reward=1.0, value=0.0)
    transition.old_token_logprobs = torch.tensor([-4.0])
    transition.advantage = 1.0
    transition.return_ = 1.0
    before = actor.model.weight.detach().clone()
    stats = trainer._update([transition])
    assert torch.equal(actor.model.weight.detach(), before)
    assert stats["policy_updates_applied"] == 0.0
    assert stats["kl_rejected_updates"] == 1.0
    assert stats["ppo_early_stop"] == 1.0


def test_role_balanced_actor_update_accumulates_then_steps_once(tmp_path) -> None:
    class Actor:
        device = torch.device("cpu")

        def __init__(self):
            self.model = torch.nn.Linear(1, 1, bias=False)
            torch.nn.init.zeros_(self.model.weight)

        def score_batch(self, sequences):
            rows = len(sequences)
            logp = self.model.weight.reshape(1, 1).expand(rows, 1)
            return logp, torch.zeros_like(logp)

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, states):
            return self.value.expand(len(states))

    actor, critic = Actor(), Critic()
    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.torch, trainer.actor, trainer.critic = torch, actor, critic
    trainer.run_dir, trainer.global_step = tmp_path, 0
    trainer.config = SimpleNamespace(
        use_vllm_generation=False, ppo_old_logprob_source="local_actor",
        normalize_advantages=False, ppo_epochs=1, minibatch_size=1,
        actor_minibatch_mode="role_balanced", clip_epsilon=0.2,
        entropy_coef=0.0, target_kl=0.02, max_grad_norm=1.0,
        value_clip_epsilon=0.2, value_loss_coef=0.5,
        actor_role_weights={
            "query_retriever": 1.0,
            "evidence_updater": 0.5,
            "answer_generator": 1.0,
        },
        gradient_diagnostics_steps=1,
        gradient_diagnostics_max_groups=1,
        reference_kl_recovery_role_scale=0.25,
    )
    trainer.reference_kl_recovery_steps = {
        AgentRole.QUERY: 0, AgentRole.EVIDENCE: 2, AgentRole.ANSWER: 0,
    }
    trainer.actor_optimizer = torch.optim.SGD(actor.model.parameters(), lr=0.1)
    trainer.critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.1)
    transitions = [_transition(role, reward=1.0, value=0.0) for role in AgentRole]
    for item in transitions:
        item.old_token_logprobs = torch.tensor([0.0])
        item.advantage = item.raw_advantage = item.return_ = 1.0
    stats = trainer._update(transitions)
    assert stats["policy_updates_applied"] == 1.0
    assert actor.model.weight.item() > 0.0
    assert set(stats["role_metrics"]) == {role.value for role in AgentRole}
    assert all(
        metrics["optimizer_microbatches"] == 1.0
        for metrics in stats["role_metrics"].values()
    )
    assert stats["role_metrics"]["evidence_updater"]["configured_actor_role_weight"] == 0.5
    assert stats["role_metrics"]["evidence_updater"]["recovery_role_scale_applied"] == 0.25
    assert trainer.reference_kl_recovery_steps[AgentRole.EVIDENCE] == 1
    assert len(stats["gradient_diagnostics"]) == 1
    assert stats["gradient_diagnostics"][0]["conflict_rate"] == 0.0
    assert (tmp_path / "gradient_diagnostics.jsonl").is_file()


def test_validation_exports_best_actor_and_restores_temperature(tmp_path) -> None:
    class Saver:
        def save_pretrained(self, path, **kwargs):
            del kwargs
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "saved").write_text("ok", encoding="utf-8")

    class Actor:
        temperature = 0.8
        model = Saver()
        tokenizer = Saver()

    class Collector:
        def __init__(self, actor):
            self.actor = actor

        def collect(self, sample):
            assert self.actor.temperature == 0.0
            return Episode(
                qid=sample.qid, dataset=sample.dataset, final_answer="answer",
                global_reward=1.5, answer_f1=1.0, evidence_coverage=0.5,
            )

    sample = RLSample(
        qid="validation-q", dataset="d", question="q", answer="answer",
        answer_aliases=(), supporting_facts=(),
    )
    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.actor = Actor()
    trainer.collector = Collector(trainer.actor)
    trainer.validation_samples = [sample]
    trainer.run_dir = tmp_path
    trainer.global_step = 0
    trainer.best_validation_score = float("-inf")
    trainer.best_validation_step = 0
    trainer.baseline_validation_score = None
    trainer.bad_validation_count = 0
    trainer.last_validation_step = -1
    trainer.stopped_early = False
    sft_adapter = tmp_path / "sft_adapter"
    sft_adapter.mkdir()
    (sft_adapter / "prompt_contract.json").write_text(
        '{"prompt_contract_version":"test"}\n', encoding="utf-8",
    )
    trainer.config = SimpleNamespace(
        validation_temperature=0.0, validation_score_parse_penalty=1.0,
        validation_min_delta=0.001, early_stopping_patience=3,
        max_protocol_parse_failure_rate=0.05,
        max_validation_missing_answer_tag_rate=0.02,
        min_validation_final_compliance_rate=0.95,
        resume_from_checkpoint="", sft_adapter_path=str(sft_adapter),
    )
    result = trainer._validate()
    assert trainer.actor.temperature == 0.8
    assert result["improved"] is True and result["checkpoint_eligible"] is True
    assert result["validation_kind"] == "baseline"
    assert (tmp_path / "best_actor" / "saved").is_file()
    assert (tmp_path / "best_answer_actor" / "saved").is_file()
    assert (tmp_path / "best_protocol_actor" / "saved").is_file()
    exported_contract = json.loads(
        (tmp_path / "best_actor" / "prompt_contract.json").read_text(encoding="utf-8")
    )
    assert exported_contract["prompt_contract_version"] == "test"
    assert exported_contract["evidence_pointer_format"] == "P{local_passage_id}"
    assert exported_contract["evidence_rationale_in_policy_loss"] is False
    assert (tmp_path / "baseline_validation.json").is_file()
    assert (tmp_path / "best_validation.json").is_file()
    assert (tmp_path / "validation_metrics.jsonl").is_file()
    assert (tmp_path / "validation_episodes.jsonl").is_file()


def test_protocol_ineligible_validation_cannot_replace_best_actor(tmp_path) -> None:
    class Saver:
        def save_pretrained(self, path, **kwargs):
            del kwargs
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "saved").write_text("ok", encoding="utf-8")

    class Actor:
        temperature = 0.8
        model = Saver()
        tokenizer = Saver()

    class Collector:
        def __init__(self):
            self.invalid = False

        def collect(self, sample):
            return Episode(
                qid=sample.qid, dataset=sample.dataset, final_answer="answer",
                global_reward=2.0 if self.invalid else 0.5,
                answer_f1=1.0 if self.invalid else 0.25,
                evidence_coverage=1.0 if self.invalid else 0.25,
                parse_errors=["Invalid JSON"] if self.invalid else [],
            )

    sft_adapter = tmp_path / "sft_adapter"
    sft_adapter.mkdir()
    (sft_adapter / "prompt_contract.json").write_text(
        '{"prompt_contract_version":"test"}\n', encoding="utf-8",
    )
    trainer = MAPPOTrainer.__new__(MAPPOTrainer)
    trainer.actor = Actor()
    trainer.collector = Collector()
    trainer.validation_samples = [RLSample(
        qid="validation-q", dataset="musique", question="q", answer="answer",
        answer_aliases=(), supporting_facts=(),
    )]
    trainer.run_dir = tmp_path
    trainer.global_step = 0
    trainer.best_validation_score = float("-inf")
    trainer.best_validation_step = 0
    trainer.baseline_validation_score = None
    trainer.bad_validation_count = 0
    trainer.last_validation_step = -1
    trainer.stopped_early = False
    trainer.config = SimpleNamespace(
        validation_temperature=0.0, validation_score_parse_penalty=0.1,
        validation_score_answer_weight=1.5,
        validation_score_evidence_weight=1.0,
        validation_min_delta=0.001, early_stopping_patience=3,
        early_stopping_min_steps=300,
        max_protocol_parse_failure_rate=0.01,
        max_validation_missing_answer_tag_rate=0.005,
        min_validation_final_compliance_rate=0.99,
        resume_from_checkpoint="", sft_adapter_path=str(sft_adapter),
    )
    eligible = trainer._validate()
    assert eligible["improved"] is True
    assert trainer.best_validation_score == 0.25

    trainer.global_step = 1
    trainer.collector.invalid = True
    ineligible = trainer._validate()
    assert ineligible["score_improved"] is True
    assert ineligible["checkpoint_eligible"] is False
    assert ineligible["improved"] is False
    assert trainer.best_validation_score == 0.25
    assert trainer.best_validation_step == 0
    # Protocol eligibility controls exports, while early-stop patience tracks
    # actual answer-quality stagnation independently.
    assert ineligible["quality_improved"] is True
    assert trainer.bad_validation_count == 0
    assert json.loads(
        (tmp_path / "best_validation.json").read_text(encoding="utf-8")
    )["global_step"] == 0

    trainer.collector.invalid = False
    trainer.bad_validation_count = 2
    early_score = trainer.best_early_stopping_score
    trainer.global_step = 2
    emergency = trainer._validate(kind="kl_emergency")
    assert emergency["affects_early_stopping"] is False
    assert emergency["early_stopping_improved"] is None
    assert trainer.bad_validation_count == 2
    assert trainer.best_early_stopping_score == early_score
    assert trainer.stopped_early is False
    trainer.global_step = 3
    final = trainer._validate(kind="final")
    assert final["affects_early_stopping"] is False
    assert trainer.bad_validation_count == 2
    assert trainer.best_early_stopping_score == early_score

    trainer.global_step = 299
    trainer._validate()
    assert trainer.stopped_early is False
    trainer.global_step = 300
    trainer._validate()
    assert trainer.stopped_early is True


def test_vllm_client_uses_token_ids_and_behavior_logprobs() -> None:
    class Response:
        status_code = 200
        text = "ok"

        def __init__(self, payload):
            self.payload = payload

        def json(self):
            return self.payload

    class Session:
        trust_env = True

        def __init__(self):
            self.payload = None

        def request(self, method, url, json, timeout):
            del method, timeout
            self.payload = json
            assert url.endswith("/v1/completions")
            return Response({"choices": [{
                "text": "answer", "logprobs": {
                    "tokens": ["token_id:3", "token_id:4"],
                    "token_logprobs": [-0.25, -0.5],
                },
            }]})

    session = Session()
    client = VLLMClient(
        host="127.0.0.1", port=8002, model_name="base",
        lora_name="policy", timeout=1.0, attempts=1, backoff=0.0,
        session=session,
    )
    assert session.trust_env is False
    output = client.generate(
        [1, 2], max_tokens=8, temperature=0.8, top_p=0.95,
        top_k=5, seed=42, guided_regex=FINAL_ANSWER_GUIDED_REGEX,
    )
    assert output.token_ids == [3, 4]
    assert output.token_logprobs == [-0.25, -0.5]
    assert session.payload["guided_regex"] == FINAL_ANSWER_GUIDED_REGEX


def test_vllm_lora_sync_publishes_versioned_snapshot(tmp_path) -> None:
    class Response:
        status_code = 200
        text = "ok"

        def __init__(self, payload=None):
            self.payload = payload or {}

        def json(self):
            return self.payload

    class Session:
        trust_env = True

        def __init__(self):
            self.calls = []

        def request(self, method, url, json, timeout):
            del timeout
            self.calls.append((method, url, json))
            if url.endswith("/v1/models"):
                return Response({"data": [{"id": "base"}, {"id": "policy"}]})
            return Response()

    class Saver:
        def save_pretrained(self, path, **kwargs):
            del kwargs
            (path / "adapter_config.json").write_text("{}", encoding="utf-8")

    class Tokenizer:
        def save_pretrained(self, path):
            (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    initial = tmp_path / "initial"
    initial.mkdir()
    session = Session()
    client = VLLMClient(
        host="127.0.0.1", port=8002, model_name="base",
        lora_name="policy", timeout=1.0, attempts=1, backoff=0.0,
        initial_adapter_path=initial, session=session,
    )
    snapshot = client.sync_lora(
        Saver(), Tokenizer(), sync_root=tmp_path / "sync", step=3, keep=2,
    )
    assert snapshot.name == "step-00000003"
    assert (snapshot / "READY").is_file()
    assert [call[1].rsplit("/", 1)[-1] for call in session.calls] == [
        "unload_lora_adapter", "load_lora_adapter", "models",
    ]


def test_shell_launcher_runs_from_rl_mappo_directory() -> None:
    directory = Path(__file__).resolve().parent
    environment = dict(os.environ)
    environment["MAPPO_LAUNCH_DRY_RUN"] = "1"
    completed = subprocess.run(
        ["bash", "run_mappo.sh"], cwd=directory, env=environment,
        text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "partially initialized module 'types'" not in completed.stderr
    assert "[mappo-trainer]" in completed.stdout
