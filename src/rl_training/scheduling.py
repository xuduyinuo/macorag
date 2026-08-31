from __future__ import annotations

import math
from typing import Any


def build_cosine_scheduler(
    optimizer: Any,
    *,
    total_updates: int,
    warmup_ratio: float,
    min_lr_ratio: float,
) -> Any:
    if int(total_updates) <= 0:
        raise ValueError("total_updates must be positive")
    if not 0.0 <= float(warmup_ratio) < 1.0:
        raise ValueError("warmup_ratio must satisfy 0 <= value < 1")
    if not 0.0 <= float(min_lr_ratio) <= 1.0:
        raise ValueError("min_lr_ratio must satisfy 0 <= value <= 1")

    from torch.optim.lr_scheduler import LambdaLR

    total = int(total_updates)
    warmup = int(math.ceil(total * float(warmup_ratio)))
    floor = float(min_lr_ratio)

    def multiplier(completed_updates: int) -> float:
        update = max(0, min(int(completed_updates), total))
        if warmup > 0 and update <= warmup:
            return update / warmup
        decay_updates = max(1, total - warmup)
        progress = min(1.0, max(0.0, (update - warmup) / decay_updates))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return floor + ((1.0 - floor) * cosine)

    return LambdaLR(optimizer, lr_lambda=multiplier)
