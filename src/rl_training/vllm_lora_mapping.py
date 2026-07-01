from __future__ import annotations

import re
from typing import Any


_LORA_RE = re.compile(
    r"^(?:base_model\.model\.)?(?P<base>model\.layers\.\d+\.(?:self_attn|mlp)\."
    r"(?:q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj))\."
    r"(?P<side>lora_[AB])\.default\.weight$"
)


def normalize_peft_lora_name(name: str) -> str | None:
    match = _LORA_RE.match(name)
    if match is None:
        return None
    return f"{match.group('base')}.{match.group('side')}.weight"


def collect_lora_named_tensors(model: Any, device: Any | None = None) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    for name, parameter in model.named_parameters():
        mapped = normalize_peft_lora_name(name)
        if mapped is None:
            continue
        tensor = parameter.detach().float()
        if device is not None:
            tensor = tensor.to(device=device)
        else:
            tensor = tensor.cpu()
        tensors[mapped] = tensor
    return tensors
