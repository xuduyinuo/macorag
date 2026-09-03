from copy import deepcopy
from types import SimpleNamespace

import pytest

from answer_metrics import calculate_f1
from rl_training.checkpointing import fingerprint_config
from rl_training.config import parse_args
from rl_training.rewards import compute_action_rewards, compute_answer_f1, compute_rl_rewards
from rl_training.trainer import assign_action_advantages


@pytest.mark.parametrize("prediction,gold,aliases", [
    ("7531", "7,531", []),
    ("30--60%", "30% to 65%", []),
    ("Scott Carson", "Scott Paul Carson", ["Scott Carson"]),
    ("Jessie Woodrow Wilson Sayre", "Jessie Woodrow Wilson", ["Jessie Woodrow Wilson Sayre"]),
    (None, "Ada", []), ("", "", []), ("0", 0, []),
    ("the david arquette!", "David Arquette", []),
])
def test_training_correctness_exactly_matches_fixed_evaluator(prediction, gold, aliases):
    sample = {"answer": gold, "answer_aliases": aliases, "supporting_facts": []}
    rollout = {
        "final_answer": prediction, "parse_errors": [],
        "trajectory": [{"round": 0, "generated_roles": ["answer_generator"],
                        "answer": {"can_answer": True, "answer": prediction}}],
    }
    expected = calculate_f1(prediction, gold)
    assert compute_answer_f1(prediction, gold, aliases) == expected
    assert compute_rl_rewards(rollout=rollout, sample=sample)["answer_f1"] == expected
    credit = compute_action_rewards(rollout=rollout, sample=sample)
    assert credit["terminal_reward"] == 2 * expected
    assert credit["action_rewards"][0]["components"]["answer_correctness"] == 2 * expected


def _answer_wait_pair():
    passage = {"passage_id": 0, "doc_id": "d1", "title": "Ada", "text": "Ada"}
    sample = {"answer": "Ada", "supporting_facts": [passage]}
    turn = {
        "round": 0, "generated_roles": ["query_retriever", "evidence_updater", "answer_generator"],
        "query_retriever": {"query": "who is Ada", "sub_goal": "identify Ada"},
        "observation": {"passages": [passage]},
        "update_evidence": {"selected_passage_ids": [0]},
        "answer": {"can_answer": True, "answer": "Ada"},
    }
    immediate = {"final_answer": "Ada", "parse_errors": [], "trajectory": [turn]}
    waited = deepcopy(immediate)
    waited["trajectory"][0]["answer"] = {"can_answer": False, "answer": None}
    waited["trajectory"].append({"round": 1, "generated_roles": ["answer_generator"],
                                  "answer": {"can_answer": True, "answer": "Ada"}})
    return sample, immediate, waited


def test_ablation_removes_local_stop_preference_without_changing_terminal_or_other_roles():
    sample, immediate, waited = _answer_wait_pair()
    arms = {}
    for weight in (1.0, 0.0):
        group = []
        for rollout in (immediate, waited):
            credit = compute_action_rewards(rollout=rollout, sample=sample, answer_local_reward_weight=weight)
            assert credit["terminal_reward"] == 3.0
            action = SimpleNamespace(role="answer_generator", round_index=0)
            group.append({**credit, "actions": [action]})
        assign_action_advantages(group, global_weights={"answer_generator": 7 / 3},
                                 granularity="role_round", epsilon=1e-6,
                                 degenerate_bucket_fallback_weight=0.2)
        arms[weight] = group
    assert arms[1.0][0]["actions"][0].advantage > 0.99
    assert arms[1.0][1]["actions"][0].advantage < -0.99
    assert [g["actions"][0].advantage for g in arms[0.0]] == [0.0, 0.0]
    for control, ablated in zip(arms[1.0], arms[0.0]):
        assert control["action_rewards"][:2] == ablated["action_rewards"][:2]


def test_ablation_preserves_format_penalty_and_partial_scaling():
    sample, immediate, _ = _answer_wait_pair()
    immediate["trajectory"][0]["parse_error_role"] = "answer_generator"
    immediate["parse_errors"] = ["bad answer"]
    values = [compute_action_rewards(rollout=immediate, sample=sample, answer_local_reward_weight=w)
              for w in (0.0, 0.25, 1.0)]
    assert [v["terminal_reward"] for v in values] == [2.0, 2.0, 2.0]
    assert [v["action_rewards"][-1]["local_reward"] for v in values] == [-1.0, -0.5, 1.0]


@pytest.mark.parametrize("weight", ["-0.1", "1.1", "nan", "inf"])
def test_invalid_local_weight_rejected(weight):
    with pytest.raises(SystemExit):
        parse_args(["--answer-local-reward-weight", weight])
    with pytest.raises(ValueError):
        compute_action_rewards(rollout={}, sample={}, answer_local_reward_weight=float(weight))


def test_reward_variant_changes_resume_identity(monkeypatch):
    import rl_training.checkpointing as checkpointing
    args = parse_args([])
    original = fingerprint_config(args)
    args.answer_local_reward_weight = 0.0
    assert fingerprint_config(args) != original
    args.answer_local_reward_weight = 1.0
    monkeypatch.setattr(checkpointing, "ANSWER_F1_CONTRACT", "old_reward_metric")
    assert fingerprint_config(args) != original


def test_training_entrypoint_passes_local_weight_to_reward_computation():
    from rl_training.train_grpo_macorag import _score_rollout_candidates
    sample, immediate, _ = _answer_wait_pair()
    reward_sample = SimpleNamespace(to_reward_sample=lambda: sample)
    args = parse_args(["--answer-local-reward-weight", "0"])
    immediate["actions"] = []
    _score_rollout_candidates(args=args, sample=reward_sample, rollouts=[immediate])
    answer = immediate["action_rewards"][-1]
    assert answer["role"] == "answer_generator"
    assert answer["local_reward"] == 0.0
    assert immediate["terminal_reward"] == 3.0
