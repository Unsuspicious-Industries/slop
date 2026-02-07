"""Wire protocol for server-client communication over SSH.

Messages are length-prefixed msgpack blobs sent over stdin/stdout of an SSH
subprocess.  This avoids any need for HTTP, open ports, or firewall rules
on shared HPC clusters.

Frame format (big-endian):
    [4 bytes: payload length N][N bytes: msgpack payload]

Every payload is a dict with at least a "kind" key that routes it to the
right handler.
"""

from .messages import (
    MessageKind,
    Request,
    Response,
    InferenceRequest,
    EncodeRequest,
    IntrospectRequest,
    JobResult,
    ServerInfo,
    ErrorResponse,
)
from .wire import write_message, read_message, FrameError
from .serialization import pack_array, unpack_array

__all__ = [
    "MessageKind",
    "Request",
    "Response",
    "InferenceRequest",
    "EncodeRequest",
    "IntrospectRequest",
    "JobResult",
    "ServerInfo",
    "ErrorResponse",
    "write_message",
    "read_message",
    "FrameError",
    "pack_array",
    "unpack_array",
]
