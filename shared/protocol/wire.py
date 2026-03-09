"""Wire format: length-prefixed JSON over byte streams.

Used to communicate between client and server processes connected
via SSH (stdin/stdout of a subprocess). No HTTP, no sockets, no
firewall rules needed.

Frame layout (all integers big-endian):

    ┌──────────────┬────────────────────────┐
    │ 4 bytes: len │ len bytes: JSON blob   │
    └──────────────┴────────────────────────┘

The payload is always a UTF-8 encoded JSON object.
Binary data (like numpy arrays) must be base64-encoded within the JSON structure
(see serialization.py).

This avoids dependencies on 'msgpack' or other non-standard libraries,
making deployment to constrained HPC environments trivial.
"""

from __future__ import annotations

import io
import json
import struct
from typing import Any, BinaryIO, Dict, Optional

_HEADER = struct.Struct(">I")  # 4-byte big-endian unsigned int
_MAX_FRAME = 2 * 1024 * 1024 * 1024  # 2 GiB sanity cap


class FrameError(Exception):
    """Raised when a frame cannot be read or written."""


def write_message(stream: BinaryIO, msg: Dict[str, Any]) -> None:
    """Serialize *msg* and write a length-prefixed frame to *stream*.

    Parameters
    ----------
    stream : writable binary stream (e.g. ``proc.stdin``)
    msg    : dict — must be JSON-serializable
    """
    # Serialize to JSON bytes
    payload = json.dumps(msg).encode("utf-8")
    
    if len(payload) > _MAX_FRAME:
        raise FrameError(f"payload too large: {len(payload)} bytes")
        
    stream.write(_HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def read_message(stream: BinaryIO) -> Optional[Dict[str, Any]]:
    """Read one length-prefixed frame from *stream* and return the dict.

    Blocks until a full frame is available.  Returns ``None`` if
    the stream reaches EOF before the header is read (clean shutdown).

    Raises
    ------
    FrameError
        If the header is partial or the payload is truncated.
    """
    header = _read_exact(stream, _HEADER.size)
    if len(header) == 0:
        return None  # clean EOF
    if len(header) < _HEADER.size:
        raise FrameError("truncated frame header")

    (length,) = _HEADER.unpack(header)
    if length > _MAX_FRAME:
        raise FrameError(f"frame too large: {length} bytes")

    payload = _read_exact(stream, length)
    if len(payload) < length:
        raise FrameError(f"truncated payload: expected {length}, got {len(payload)}")

    return json.loads(payload.decode("utf-8"))


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    """Read exactly *n* bytes, handling partial reads."""
    buf = bytearray()
    while len(buf) < n:
        # Read in chunks (e.g. 512KB) to avoid single massive read calls
        # which can trigger OS-level timeouts or buffer weirdness on pipes
        to_read = min(n - len(buf), 512 * 1024)
        chunk = stream.read(to_read)
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)
