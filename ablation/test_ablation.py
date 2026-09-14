from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

import pytest

from ablation.ablation_cli import adapter_model_identity
from ablation.select_gpu import select_gpu
from ablation.train_grpo_ablation import _trajectory_credit, _without_local_rewards


class Role(str, Enum):
    QUERY = "query_retriever"
    ANSWER = "answer_generator"


@dataclass
class Action:
    role: Role
    round_index: int
    local_reward: float = 99.0
    terminal_reward: float = 99.0
    decision_return: float = 99.0
    primary_advantage: float = 99.0
    fallback_advantage: float = 99.0
    advantage: float = 99.0


def test_trajectory_credit_broadcasts_one_advantage_to_every_action() -> None:
    first = [Action(Role.QUERY, 0), Action(Role.ANSWER, 0)]
    second = [Action(Role.QUERY, 0), Action(Role.ANSWER, 0)]
    stats = _trajectory_credit(
        [
            {"terminal_reward": 1.0, "actions": first},
            {"terminal_reward": 3.0, "actions": second},
        ],
        global_weights={},
    )
    assert first[0].advantage == pytest.approx(-1.0)
    assert first[1].advantage == pytest.approx(first[0].advantage)
    assert second[0].advantage == pytest.approx(1.0)
    assert second[1].advantage == pytest.approx(second[0].advantage)
    assert stats["trajectory"]["action_count"] == 4


def test_without_local_rewards_zeroes_every_role_and_component() -> None:
    def original(**_: object) -> dict[str, object]:
        return {
            "terminal_reward": 2.5,
            "action_rewards": [
                {"role": "query_retriever", "local_reward": 1.2, "components": {"novelty": 0.2}},
                {"role": "evidence_updater", "local_reward": -0.5, "components": {"valid": -0.5}},
                {"role": "answer_generator", "local_reward": 3.0, "components": {"f1": 3.0}},
            ],
        }

    result = _without_local_rewards(original)(rollout={}, sample={})
    assert result["terminal_reward"] == 2.5
    assert all(item["local_reward"] == 0.0 for item in result["action_rewards"])
    assert all(set(item["components"].values()) == {0.0} for item in result["action_rewards"])


def test_adapter_model_identity_accepts_equivalent_path_spellings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real_model = tmp_path / "real-model"
    real_model.mkdir()
    model_alias = tmp_path / "model-alias"
    model_alias.symlink_to(real_model, target_is_directory=True)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": str(real_model)}),
        encoding="utf-8",
    )

    adapter_model_identity(adapter, model_alias)

    assert capsys.readouterr().out.strip() == str(real_model)


def test_select_gpu_uses_most_free_memory_and_enforces_threshold() -> None:
    assert select_gpu([(0, 24000), (1, 3000)], min_free_mib=18000) == 0
    with pytest.raises(RuntimeError, match="No GPU has the required"):
        select_gpu([(0, 12000), (1, 3000)], min_free_mib=18000)
