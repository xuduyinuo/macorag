"""Isolated MACORAG RL-v2 MAPPO implementation.

The package owns its data adapter, prompt contract, unified retriever, vLLM
rollout client, MAPPO learner, checkpointing, configuration and launch script.
"""

from .mappo import compute_gae, mappo_actor_loss, clipped_value_loss
from .mappo_types import AgentRole, Episode, MAPPOTransition, RLSample

__all__ = [
    "AgentRole",
    "Episode",
    "MAPPOTransition",
    "RLSample",
    "clipped_value_loss",
    "compute_gae",
    "mappo_actor_loss",
]
