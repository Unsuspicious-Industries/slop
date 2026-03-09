"""Core types for physics utilities.

`Field` is a thin wrapper around a callable that maps coordinates to values.
Coordinates and values are numpy arrays.

The callable should support batched evaluation, for example:
    coords: (B, ...)
    values: (B, ...)
"""

from __future__ import annotations

from typing import Callable

import numpy as np


Tensor = np.ndarray
ScalarField = np.ndarray
VectorField = np.ndarray
Vector = np.ndarray


class Field:
    """Callable field wrapper.

    This is intentionally minimal. Higher level field types in this repo
    compose behavior by wrapping `Field` instances.
    """

    def __init__(self, field_fn: Callable[[np.ndarray], np.ndarray]):
        self.field_fn = field_fn

    def __call__(self, coords: np.ndarray) -> np.ndarray:
        return self.field_fn(coords)
