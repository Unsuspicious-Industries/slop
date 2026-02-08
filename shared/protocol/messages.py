"""Message types for the SLOP inference protocol.

Every request/response is a plain dict serializable to msgpack.
We use dataclass-like structures for clarity but convert to dicts for wire.
"""

from __future__ import annotations

import enum
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


class MessageKind(str, enum.Enum):
    """Top-level message discriminator sent on the wire."""

    # Client -> Server
    PING = "ping"
    SERVER_INFO = "server_info"
    INFERENCE = "inference"          # run diffusion, get full introspection
    ENCODE = "encode"                # encode text/image -> embeddings
    INTROSPECT = "introspect"        # model metadata / architecture dump
    SHUTDOWN = "shutdown"

    # Server -> Client
    PONG = "pong"
    RESULT = "result"
    ERROR = "error"
    INFO = "info"
    PROGRESS = "progress"


# ---------------------------------------------------------------------------
# Requests (client -> server)
# ---------------------------------------------------------------------------

@dataclass
class Request:
    """Base request envelope."""
    kind: str
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Request":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class InferenceRequest(Request):
    """Run diffusion generation with full trajectory capture.

    The server will return:
      - final image (PNG bytes)
      - per-step latents (compressed numpy)
      - per-step noise predictions
      - prompt embeddings (text encoder hidden states)
      - attention maps (cross-attention, if captured)
      - scheduler state (sigmas / timesteps)
      - model config metadata
    """
    kind: str = MessageKind.INFERENCE.value

    model_id: str = "stabilityai/stable-diffusion-2-1"
    prompt: str = ""
    negative_prompt: str = ""
    num_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: int = -1                    # -1 = random

    # Introspection knobs
    capture_latents: bool = True
    capture_noise_pred: bool = True
    capture_prompt_embeds: bool = True
    capture_attention: bool = False   # expensive, off by default
    capture_timesteps: bool = True    # capture scheduler timestep values
    latent_sample_rate: int = 1       # capture every N steps
    compress_latents: bool = True     # float16 + zlib

    # Embedding overrides — bypass text encoder, inject embeddings directly.
    # Values are packed-array dicts from shared.protocol.serialization.pack_array.
    # When set, the server skips text encoding and uses these embeddings for the UNet.
    prompt_embeds_override: Optional[Dict[str, Any]] = None
    negative_prompt_embeds_override: Optional[Dict[str, Any]] = None


@dataclass
class EncodeRequest(Request):
    """Encode text or image through an encoder model."""
    kind: str = MessageKind.ENCODE.value

    model_id: str = "openai/clip-vit-large-patch14"
    modality: str = "text"            # "text" | "image"
    inputs: List[str] = field(default_factory=list)  # texts or base64-encoded images
    return_hidden_states: bool = False


@dataclass
class IntrospectRequest(Request):
    """Ask the server to dump model architecture / config info."""
    kind: str = MessageKind.INTROSPECT.value
    model_id: str = ""


# ---------------------------------------------------------------------------
# Responses (server -> client)
# ---------------------------------------------------------------------------

@dataclass
class Response:
    """Base response envelope."""
    kind: str
    job_id: str = ""
    timestamp: float = field(default_factory=time.time)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Response":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class JobResult(Response):
    """Successful inference / encode result.

    Binary blobs (images, arrays) are stored as sub-dicts with keys:
        dtype, shape, data (bytes, zlib-compressed)
    Use shared.protocol.serialization.unpack_array to reconstruct.
    """
    kind: str = MessageKind.RESULT.value

    # What was requested
    request_kind: str = ""
    model_id: str = ""
    prompt: str = ""

    # Payload — the actual data
    # Keys vary by request type; always JSON-safe except for "arrays"
    # which is a dict[str, packed_array_dict].
    payload: Dict[str, Any] = field(default_factory=dict)
    arrays: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ServerInfo(Response):
    """Server capability advertisement."""
    kind: str = MessageKind.INFO.value

    hostname: str = ""
    gpu_name: str = ""
    gpu_memory_mb: int = 0
    cuda_version: str = ""
    torch_version: str = ""
    loaded_models: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)


@dataclass
class ErrorResponse(Response):
    """Something went wrong."""
    kind: str = MessageKind.ERROR.value
    error: str = ""
    traceback: str = ""
