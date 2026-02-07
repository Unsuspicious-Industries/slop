"""Numpy array <-> JSON-safe dict conversion.

Arrays are compressed with zlib, base64-encoded, and stored as::

    {"dtype": "float32", "shape": [50, 50, 2], "data": "base64...", "encoding": "b64"}

This ensures strict JSON compatibility (no raw bytes), removing the need for
msgpack or other non-standard libraries.
"""

from __future__ import annotations

import zlib
import base64
from typing import Any, Dict, Optional

import numpy as np


def pack_array(
    arr: np.ndarray,
    *,
    compress: bool = True,
    half: bool = False,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a numpy array to a JSON-safe dict.

    Parameters
    ----------
    arr      : the array
    compress : zlib-compress the raw bytes (default True)
    half     : downcast float64/float32 to float16 first (saves bandwidth)
    name     : optional label stored alongside the data

    Returns
    -------
    dict with keys: dtype, shape, data, compressed, encoding, name
    """
    if half and arr.dtype in (np.float32, np.float64):
        arr = arr.astype(np.float16)

    raw = arr.tobytes()
    compressed = False
    if compress:
        raw = zlib.compress(raw, level=6)
        compressed = True

    # Base64 encode for JSON safety
    b64_data = base64.b64encode(raw).decode('ascii')

    result: Dict[str, Any] = {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": b64_data,
        "compressed": compressed,
        "encoding": "base64"
    }
    if name:
        result["name"] = name
    return result


def unpack_array(d: Dict[str, Any]) -> np.ndarray:
    """Reconstruct a numpy array from a packed dict.

    Parameters
    ----------
    d : dict produced by ``pack_array``

    Returns
    -------
    numpy array with the original dtype and shape
    """
    raw_str = d["data"]
    
    # Handle base64 decoding
    if d.get("encoding") == "base64":
        raw = base64.b64decode(raw_str)
    else:
        # Fallback for old/direct bytes (shouldn't happen with JSON)
        raw = raw_str

    if d.get("compressed", False):
        raw = zlib.decompress(raw)

    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"])
