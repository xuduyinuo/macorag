from __future__ import annotations

import pytest
import torch

from rl_training.scheduling import build_cosine_scheduler


def test_cosine_schedule_reaches_warmup_peak_and_floor() -> None:
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([parameter], lr=1.0e-5)
    scheduler = build_cosine_scheduler(
        optimizer,
        total_updates=1000,
        warmup_ratio=0.03,
        min_lr_ratio=0.1,
    )

    values = []
    for _ in range(1000):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])

    assert max(values[:30]) == pytest.approx(1.0e-5)
    assert values[-1] == pytest.approx(1.0e-6)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"total_updates": 0, "warmup_ratio": 0.03, "min_lr_ratio": 0.1},
        {"total_updates": 10, "warmup_ratio": 1.0, "min_lr_ratio": 0.1},
        {"total_updates": 10, "warmup_ratio": 0.03, "min_lr_ratio": 1.1},
    ],
)
def test_cosine_schedule_rejects_invalid_contract(kwargs) -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=1.0)
    with pytest.raises(ValueError):
        build_cosine_scheduler(optimizer, **kwargs)
