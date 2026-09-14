"""Self-contained MAPPO training package for MACORAG.

The package deliberately does not import project-local modules outside
``src.rl_mappo``.  Query, evidence and answer roles are cooperative agents;
they share a role-conditioned language-model actor and use a centralized
critic during training.
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
