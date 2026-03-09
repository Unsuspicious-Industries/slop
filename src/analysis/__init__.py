"""Analysis utilities: flow field construction, attractor detection, and more."""
from .flow_fields import compute_flow_field, compute_flow_field_3d
from .attractors import find_attractors, find_repellers

__all__ = [
    "compute_flow_field",
    "compute_flow_field_3d",
    "find_attractors",
    "find_repellers",
]
