"""Minimal wire protocol for remote inference primitives."""

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
    EMBED = "embed"                  # encode prompts -> embeddings only
    ENCODE = "encode"                # encode text/image -> embeddings
    DECODE = "decode"                # VAE-decode raw latents -> images
    INTROSPECT = "introspect"        # model metadata / architecture dump
    CLEANUP = "cleanup"              # trigger VRAM / memory cleanup
    TRAIN = "train"                  # run training on remote
    JOB_LIST = "job_list"            # list autonomous jobs
    JOB_ATTACH = "job_attach"        # fetch job status/progress (poll)
    JOB_KILL = "job_kill"            # terminate a job
    DATASET_STATS = "dataset_stats"  # summarize a manifest dataset
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
    kind: str = MessageKind.INFERENCE.value

    model_id: str = "stabilityai/stable-diffusion-2-1"
    prompt: str = ""
    negative_prompt: str = ""
    num_steps: int = 50
    guidance_scale: float = 7.5
    height: int = 512
    width: int = 512
    seed: int = -1                    # -1 = random
    batch_size: int = 1

    capture_latents: bool = True
    capture_noise_pred: bool = True
    capture_prompt_embeds: bool = True
    capture_timesteps: bool = True
    compress_latents: bool = True
    decode_latents: bool = True        # if False, skip VAE decode (just return latents)

    prompt_embeds_override: Optional[Dict[str, Any]] = None
    negative_prompt_embeds_override: Optional[Dict[str, Any]] = None
    base_prompt_embeds_override: Optional[Dict[str, Any]] = None
    base_prompt: str = ""

    latent_override: Optional[Dict[str, Any]] = None

    score_only: bool = False
    delta_probe: bool = False
    delta_sample: bool = False        # if True, run sampling in differential field
    probe_timestep: int = 500


@dataclass
class EncodeRequest(Request):
    """Encode text or image through an encoder model."""
    kind: str = MessageKind.ENCODE.value

    model_id: str = "openai/clip-vit-large-patch14"
    modality: str = "text"            # "text" | "image"
    inputs: List[str] = field(default_factory=list)  # texts or base64-encoded images
    return_hidden_states: bool = False


@dataclass
class EmbedRequest(Request):
    """Encode prompts into embeddings for later use with sample."""
    kind: str = MessageKind.EMBED.value

    model_id: str = "stabilityai/stable-diffusion-2-1"
    prompt: str = ""
    negative_prompt: str = ""
    return_prompt_embeds: bool = True
    return_negative_prompt_embeds: bool = True


@dataclass
class DecodeRequest(Request):
    kind: str = MessageKind.DECODE.value
    model_id: str = "runwayml/stable-diffusion-v1-5"
    latents: Optional[Dict[str, Any]] = None   # packed array


@dataclass
class IntrospectRequest(Request):
    """Ask the server to dump model architecture / config info."""
    kind: str = MessageKind.INTROSPECT.value
    model_id: str = ""


@dataclass
class CleanupRequest(Request):
    """Trigger VRAM and memory cleanup on the server."""
    kind: str = MessageKind.CLEANUP.value
    clear_model: bool = False  # if True, delete the loaded model from memory too


@dataclass
class TrainRequest(Request):
    """Run training on the server on collected teacher samples."""
    kind: str = MessageKind.TRAIN.value
    
    model_id: str = "runwayml/stable-diffusion-v1-5"
    manifest_path: str = ""  # path to manifest.jsonl on server
    output_dir: str = ""    # where to save checkpoints
    
    batch_size: int = 2
    epochs: int = 1
    learning_rate: float = 5e-5
    lora_rank: int = 16
    save_every: int = 50


@dataclass
class ListJobsRequest(Request):
    kind: str = MessageKind.JOB_LIST.value
    limit: int = 50


@dataclass
class AttachJobRequest(Request):
    kind: str = MessageKind.JOB_ATTACH.value
    target_job_id: str = ""
    since_line: int = 0
    max_lines: int = 200


@dataclass
class KillJobRequest(Request):
    kind: str = MessageKind.JOB_KILL.value
    target_job_id: str = ""
    signal: str = "term"  # "term" | "kill"


@dataclass
class DatasetStatsRequest(Request):
    kind: str = MessageKind.DATASET_STATS.value
    manifest_path: str = ""  # path on server
    max_records: int = 100000
    sample_images: int = 16  # open first N images to get dimensions


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
class ProgressResponse(Response):
    """Training progress update from server to client."""
    kind: str = MessageKind.PROGRESS.value

    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    message: str = ""
    checkpoint_path: str = ""  # if a checkpoint was just saved


@dataclass
class ErrorResponse(Response):
    """Something went wrong."""
    kind: str = MessageKind.ERROR.value
    error: str = ""
    traceback: str = ""
