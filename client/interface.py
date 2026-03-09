from __future__ import annotations

import base64
import io
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
from PIL import Image

from client.config import ProviderConfig
from client.transport import SSHTransport
from shared.protocol.messages import (
    AttachJobRequest,
    CleanupRequest,
    DatasetStatsRequest,
    DecodeRequest,
    EmbedRequest,
    EncodeRequest,
    ErrorResponse,
    InferenceRequest,
    JobResult,
    KillJobRequest,
    ListJobsRequest,
    MessageKind,
    Request,
    ServerInfo,
    TrainRequest,
)
from shared.protocol.serialization import pack_array, unpack_array


@dataclass
class Result:
    image: Optional[bytes] = None
    images: list[bytes] = field(default_factory=list)
    points: Optional[np.ndarray] = None
    forces: Optional[np.ndarray] = None
    timesteps: Optional[np.ndarray] = None
    embeddings: Optional[np.ndarray] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_job(cls, job: JobResult) -> "Result":
        result = cls()

        encoded_images = job.payload.get("images", [])
        if encoded_images:
            result.images = [base64.b64decode(image) for image in encoded_images]
            result.image = result.images[0]
        elif isinstance(job.payload.get("image"), str) and job.payload["image"]:
            result.image = base64.b64decode(job.payload["image"])
            result.images = [result.image]

        for name, packed in job.arrays.items():
            result.arrays[name] = unpack_array(packed)

        result.points = result.arrays.get("latents")
        result.forces = result.arrays.get("noise_preds")
        result.timesteps = result.arrays.get("timesteps")
        result.embeddings = result.arrays.get("prompt_embeds")
        result.metadata = {
            "job_id": job.job_id,
            "model_id": job.model_id,
            "prompt": job.prompt,
            "elapsed_s": job.elapsed_s,
            "request_kind": job.request_kind,
            **job.payload,
        }
        return result


def _packed(value: np.ndarray, *, half: bool, compress: bool = True) -> dict[str, Any]:
    """Pack a numeric array for the wire protocol."""
    arr = np.asarray(value, dtype=np.float32)
    return pack_array(arr, compress=compress, half=half)


def _probe_request(
    *,
    model_id: str,
    points: np.ndarray,
    timestep: int,
    guidance_scale: float,
    prompt: str = "",
    negative_prompt: str = "",
    prompt_embeds: Optional[np.ndarray] = None,
    negative_prompt_embeds: Optional[np.ndarray] = None,
    base_prompt: str = "",
    base_prompt_embeds: Optional[np.ndarray] = None,
    delta_probe: bool = False,
) -> InferenceRequest:
    """Build a probe request for one batched UNet evaluation."""
    req = InferenceRequest(
        model_id=model_id,
        prompt=prompt,
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        latent_override=_packed(points, half=True),
        score_only=True,
        delta_probe=delta_probe,
        probe_timestep=timestep,
        num_steps=1,
        seed=-1,
        base_prompt=base_prompt,
    )
    if prompt_embeds is not None:
        req.prompt_embeds_override = _packed(prompt_embeds, half=False)
    if negative_prompt_embeds is not None:
        req.negative_prompt_embeds_override = _packed(negative_prompt_embeds, half=False)
    if base_prompt_embeds is not None:
        req.base_prompt_embeds_override = _packed(base_prompt_embeds, half=False)
    return req


class SlopClient:
    def __init__(self, config: ProviderConfig):
        self.config = config
        self.transport = SSHTransport(config)
        self._connected = False

    def connect(self) -> None:
        if not self._connected:
            self.transport.connect()
            self._connected = True

    def close(self) -> None:
        self.transport.close()
        self._connected = False

    def __enter__(self) -> "SlopClient":
        self.connect()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()

    def _job(self, req: Request) -> JobResult:
        """Send one request and require a successful job result."""
        self.connect()
        resp = self.transport.send_request(req)
        if isinstance(resp, ErrorResponse):
            raise RuntimeError(resp.error if not resp.traceback else f"{resp.error}\n{resp.traceback}")
        if not isinstance(resp, JobResult):
            raise RuntimeError(f"unexpected response: {type(resp)}")
        return resp

    def info(self) -> ServerInfo:
        """Return server metadata."""
        self.connect()
        resp = self.transport.send_request(Request(kind=MessageKind.SERVER_INFO.value))
        if isinstance(resp, ErrorResponse):
            raise RuntimeError(resp.error)
        if not isinstance(resp, ServerInfo):
            raise RuntimeError(f"unexpected response: {type(resp)}")
        return resp

    def cleanup(self, clear_model: bool = False, timeout_s: float = 60.0) -> dict:
        """Trigger memory cleanup on the server."""
        self.connect()
        req = CleanupRequest(kind=MessageKind.CLEANUP.value, clear_model=clear_model)
        resp = self.transport.send_request(req, timeout_s=timeout_s)
        if isinstance(resp, ErrorResponse):
            raise RuntimeError(resp.error)
        return resp.payload if hasattr(resp, "payload") and resp.payload else {}

    def train(
        self,
        manifest_path: str,
        output_dir: str,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        batch_size: int = 2,
        epochs: int = 1,
        learning_rate: float = 5e-5,
        lora_rank: int = 16,
        save_every: int = 50,
        timeout_s: float = 60.0,
    ) -> Result:
        """Start an autonomous training job on the remote server.
        
        Args:
            manifest_path: Path to manifest.jsonl on the REMOTE server
            output_dir: Where to save checkpoints on the REMOTE server
            model_id: Base model to train
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate
            lora_rank: LoRA rank (for LoRA training)
            save_every: Save checkpoint every N steps
            timeout_s: Timeout for training job
            
        Returns:
            Result with training stats in metadata
        """
        self.connect()
        req = TrainRequest(
            model_id=model_id,
            manifest_path=manifest_path,
            output_dir=output_dir,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            lora_rank=lora_rank,
            save_every=save_every,
        )
        # This should return quickly (it starts a detached worker)
        resp = self.transport.send_request(req, timeout_s=timeout_s)
        if isinstance(resp, ErrorResponse):
            raise RuntimeError(resp.error)
        if not isinstance(resp, JobResult):
            raise RuntimeError(f"unexpected response: {type(resp)}")
        
        # Return a Result-like object with the job start metadata
        result = Result()
        result.metadata = {
            "job_id": resp.job_id,
            "elapsed_s": resp.elapsed_s,
            **resp.payload,
        }
        return result

    def list_jobs(self, limit: int = 50) -> list[dict[str, Any]]:
        """List autonomous jobs on the remote server."""
        job = self._job(ListJobsRequest(limit=limit))
        return list(job.payload.get("jobs", []))

    def attach_job(self, job_id: str, since_line: int = 0, max_lines: int = 200) -> dict[str, Any]:
        """Fetch job status and progress events since a line offset."""
        job = self._job(AttachJobRequest(target_job_id=job_id, since_line=since_line, max_lines=max_lines))
        return dict(job.payload)

    def kill_job(self, job_id: str, signal: str = "term") -> dict[str, Any]:
        """Terminate a running job."""
        job = self._job(KillJobRequest(target_job_id=job_id, signal=signal))
        return dict(job.payload)

    def dataset_stats(self, manifest_path: str, sample_images: int = 16, max_records: int = 100000) -> dict[str, Any]:
        """Get a dataset summary for a manifest on the remote server."""
        job = self._job(DatasetStatsRequest(manifest_path=manifest_path, sample_images=sample_images, max_records=max_records))
        return dict(job.payload)

    def encode(self, texts: str | list[str], model_id: str = "runwayml/stable-diffusion-v1-5") -> np.ndarray:
        """Encode text strings into prompt embeddings."""
        inputs = [texts] if isinstance(texts, str) else list(texts)
        job = self._job(EncodeRequest(model_id=model_id, modality="text", inputs=inputs))
        return unpack_array(job.arrays["prompt_embeds"])

    def embed(
        self,
        prompt: str = "",
        negative_prompt: str = "",
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode a prompt pair into (prompt_embeds, negative_prompt_embeds).

        Returns embeddings of shape (1, 77, 768) each.
        These can be composed arithmetically before passing to sample_from_embeds().
        """
        job = self._job(EmbedRequest(
            model_id=model_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            return_prompt_embeds=True,
            return_negative_prompt_embeds=True,
        ))
        return (
            unpack_array(job.arrays["prompt_embeds"]),
            unpack_array(job.arrays["negative_prompt_embeds"]),
        )

    def sample(
        self,
        prompt: str = "",
        num_steps: int = 50,
        seed: int = -1,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        negative_prompt: str = "",
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> Result:
        """Run sampling and return latents (no rendering). Use render() to decode latents to images."""
        req = InferenceRequest(
            model_id=model_id,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            batch_size=batch_size,
            height=height,
            width=width,
            capture_latents=True,
            capture_noise_pred=True,
            capture_timesteps=True,
            capture_prompt_embeds=True,
            decode_latents=False,
        )
        return Result.from_job(self._job(req))

    def sample_from_embeds(
        self,
        prompt_embeds: np.ndarray,
        negative_prompt_embeds: np.ndarray,
        initial_latents: Optional[np.ndarray] = None,
        # backward-compatible alias (some notebooks call `initial_latent`)
        initial_latent: Optional[np.ndarray] = None,
        num_steps: int = 50,
        seed: int = -1,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> Result:
        """Run sampling from pre-computed embeddings. Returns latents — use render() to decode.

        Embeddings can be composed arithmetically before calling:
            embeds_a, neg_a = client.embed("a person running")
            embeds_b, neg_b = client.embed("an arab person running")
            identity_vec = embeds_b - embeds_a
            result = client.sample_from_embeds(embeds_a + identity_vec, neg_a)
        
        You can also pass `initial_latents` to start diffusion from a specific latent
        tensor (shape: (B, C, H, W)). When provided the sampler will initialize
        its latents to this value instead of sampling random noise.
        """
        req = InferenceRequest(
            model_id=model_id,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            batch_size=batch_size,
            height=height,
            width=width,
            capture_latents=True,
            capture_noise_pred=True,
            capture_timesteps=True,
            capture_prompt_embeds=True,
            decode_latents=False,
            prompt_embeds_override=_packed(prompt_embeds, half=False),
            negative_prompt_embeds_override=_packed(negative_prompt_embeds, half=False),
        )
        # Accept either `initial_latents` (plural) or the historical alias
        # `initial_latent` (singular). Prefer the explicit plural if both are set.
        init = initial_latents if initial_latents is not None else initial_latent
        if init is not None:
            req.latent_override = _packed(init, half=True)
        return Result.from_job(self._job(req))

    def sample_delta(
        self,
        target_embeds: np.ndarray,
        base_embeds: np.ndarray,
        negative_prompt_embeds: np.ndarray,
        num_steps: int = 50,
        seed: int = -1,
        batch_size: int = 1,
        height: int = 512,
        width: int = 512,
        guidance_scale: float = 7.5,
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> Result:
        """Run sampling in a differential field: force = score(target) - score(base).

        This allows following the semantic direction isolated by the difference of two prompts.
        """
        req = InferenceRequest(
            model_id=model_id,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            batch_size=batch_size,
            height=height,
            width=width,
            capture_latents=True,
            capture_noise_pred=True,
            capture_timesteps=True,
            decode_latents=False,
            delta_sample=True,
            prompt_embeds_override=_packed(target_embeds, half=False),
            base_prompt_embeds_override=_packed(base_embeds, half=False),
            negative_prompt_embeds_override=_packed(negative_prompt_embeds, half=False),
        )
        return Result.from_job(self._job(req))

    def probe_at(
        self,
        points: np.ndarray,
        prompt_embeds: np.ndarray,
        negative_prompt_embeds: np.ndarray,
        timestep: int = 500,
        guidance_scale: float = 7.5,
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> np.ndarray:
        """Evaluate score model at batched latent points using pre-computed embeddings.

        points: (B, C, H, W) — batch of latent vectors
        Returns forces: (B, C, H, W)
        """
        job = self._job(_probe_request(
            model_id=model_id,
            points=points,
            timestep=timestep,
            guidance_scale=guidance_scale,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
        ))
        return unpack_array(job.arrays["noise_preds"])

    def render(self, latents: np.ndarray, model_id: str = "runwayml/stable-diffusion-v1-5") -> list[Image.Image]:
        """Decode a batch of latents into images."""
        job = self._job(DecodeRequest(model_id=model_id, latents=_packed(latents, half=True)))
        images = [Image.open(io.BytesIO(base64.b64decode(image))).convert("RGB") for image in job.payload.get("images", [])]
        return images

    def probe(
        self,
        points: np.ndarray,
        *,
        prompt: str = "",
        negative_prompt: str = "",
        prompt_embeds: Optional[np.ndarray] = None,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        timestep: int = 500,
        guidance_scale: float = 7.5,
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> np.ndarray:
        """Evaluate the score model on a batch of points at one timestep."""
        job = self._job(
            _probe_request(
                model_id=model_id,
                points=points,
                timestep=timestep,
                guidance_scale=guidance_scale,
                prompt=prompt,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
            )
        )
        return unpack_array(job.arrays["noise_preds"])

    def probe_delta(
        self,
        points: np.ndarray,
        *,
        prompt_a: str,
        prompt_b: str,
        negative_prompt: str = "",
        timestep: int = 500,
        guidance_scale: float = 7.5,
        model_id: str = "runwayml/stable-diffusion-v1-5",
    ) -> np.ndarray:
        """Return one batched delta probe from two prompts."""
        job = self._job(
            _probe_request(
                model_id=model_id,
                points=points,
                timestep=timestep,
                guidance_scale=guidance_scale,
                prompt=prompt_b,
                negative_prompt=negative_prompt,
                base_prompt=prompt_a,
                delta_probe=True,
            )
        )
        return unpack_array(job.arrays["delta_noise_preds"])


__all__ = ["Result", "SlopClient"]
