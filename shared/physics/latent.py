"""Latent-space field wrappers.

These wrappers batch many small `(B, C, H, W)` probe calls into fewer larger
batches. This reduces transport overhead and makes better use of the remote GPU.
"""

from __future__ import annotations

import asyncio
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from .types import Field


class AsyncLatentBatcher:
    """Batch synchronous functions in a background thread."""

    def __init__(
        self,
        fn: Callable[[np.ndarray], np.ndarray],
        *,
        max_batch: int = 16,
        flush_ms: float = 2.0,
    ) -> None:
        self._fn = fn
        self._max_batch = int(max_batch)
        self._flush_s = float(flush_ms) / 1000.0
        self._q: queue.Queue[tuple[np.ndarray, Future] | None] = queue.Queue()
        self._closed = False
        self._worker = threading.Thread(target=self._run, name="latent-batcher", daemon=True)
        self._worker.start()

    def close(self) -> None:
        self._closed = True
        try:
            self._q.put_nowait(None)
        except Exception:
            pass

    def submit(self, points: np.ndarray) -> Future:
        if self._closed:
            raise RuntimeError("AsyncLatentBatcher is closed")
        fut: Future = Future()
        self._q.put((np.asarray(points), fut))
        return fut

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.submit(points).result()

    async def acall(self, points: np.ndarray) -> np.ndarray:
        fut = self.submit(points)
        return await asyncio.wrap_future(fut)

    def _run(self) -> None:
        pending: list[tuple[np.ndarray, Future]] = []
        last_flush = time.time()
        while True:
            timeout = max(0.0, self._flush_s - (time.time() - last_flush))
            try:
                item = self._q.get(timeout=timeout)
                if item is None:
                    break
                pending.append(item)
            except queue.Empty:
                pass

            if not pending:
                last_flush = time.time()
                continue

            # Decide whether to flush. Batch size is measured in B.
            total = sum(int(p.shape[0]) for p, _ in pending)
            if total < self._max_batch and (time.time() - last_flush) < self._flush_s:
                continue

            # Take up to max_batch worth of pending work so we don't build
            # unbounded batches under high concurrency.
            take: list[tuple[np.ndarray, Future]] = []
            remaining: list[tuple[np.ndarray, Future]] = []
            used = 0
            for pts, fut in pending:
                b = int(pts.shape[0])
                if used == 0 or used + b <= self._max_batch:
                    take.append((pts, fut))
                    used += b
                else:
                    remaining.append((pts, fut))
            pending = remaining

            try:
                stacked = np.concatenate([p for p, _ in take], axis=0)
            except Exception as exc:
                for _, fut in take:
                    if not fut.done():
                        fut.set_exception(exc)
                last_flush = time.time()
                continue

            try:
                out = np.asarray(self._fn(stacked))
            except Exception as exc:
                for _, fut in take:
                    if not fut.done():
                        fut.set_exception(exc)
                last_flush = time.time()
                continue

            if out.shape != stacked.shape:
                exc = ValueError(f"Expected fn output shape {stacked.shape}, got {out.shape}")
                for _, fut in take:
                    if not fut.done():
                        fut.set_exception(exc)
                last_flush = time.time()
                continue

            offset = 0
            for pts, fut in take:
                b = int(pts.shape[0])
                chunk = out[offset : offset + b]
                offset += b
                if not fut.done():
                    fut.set_result(chunk)
            last_flush = time.time()

        # Cancel anything still pending.
        exc = RuntimeError("AsyncLatentBatcher is closed")
        for _, fut in pending:
            if not fut.done():
                fut.set_exception(exc)


def _normalize_latent_batch(points: np.ndarray) -> tuple[np.ndarray, tuple[int, ...]]:
    """Return `(points_4d, leading_batch_shape)`.

    Accepts any array shaped like `(*batch, C, H, W)` or `(C, H, W)`.
    """
    p = np.asarray(points)
    if p.ndim < 3:
        raise ValueError(f"Expected latent points with ndim>=3, got shape {p.shape}")
    if p.ndim == 3:
        p = p[None, ...]
    if p.ndim < 4:
        raise ValueError(f"Expected latent points ending in (C,H,W), got shape {p.shape}")
    batch_shape = tuple(p.shape[:-3])
    c, h, w = p.shape[-3:]
    flat_b = int(np.prod(batch_shape) or 1)
    p4 = p.reshape(flat_b, c, h, w)
    return p4, batch_shape


def _restore_latent_batch(values: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
    v = np.asarray(values)
    if v.ndim != 4:
        raise ValueError(f"Expected (B,C,H,W) output, got shape {v.shape}")
    if batch_shape == ():
        return v
    return v.reshape(*batch_shape, *v.shape[-3:])


@dataclass(frozen=True)
class LatentFieldConfig:
    timestep: int = 500
    guidance_scale: float = 7.5
    model_id: str = "runwayml/stable-diffusion-v1-5"
    max_batch: int = 16
    flush_ms: float = 1.0


class LatentField(Field):
    """Field backed by batched `client.probe_at` calls."""

    def __init__(
        self,
        client: Any,
        *,
        prompt_embeds: np.ndarray,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        timestep: int = 500,
        guidance_scale: float = 7.5,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        max_batch: int = 16,
        flush_ms: float = 1.0,
    ) -> None:
        self.client = client
        self.prompt_embeds = prompt_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.timestep = int(timestep)
        self.guidance_scale = float(guidance_scale)
        self.model_id = str(model_id)

        def _probe(points: np.ndarray) -> np.ndarray:
            p4, batch_shape = _normalize_latent_batch(points)
            out = client.probe_at(
                p4,
                prompt_embeds,
                negative_prompt_embeds,
                timestep=self.timestep,
                guidance_scale=self.guidance_scale,
                model_id=self.model_id,
            )
            return _restore_latent_batch(out, batch_shape)

        self._batcher = AsyncLatentBatcher(_probe, max_batch=max_batch, flush_ms=flush_ms)
        super().__init__(self._batcher)

    async def acall(self, coords: np.ndarray) -> np.ndarray:
        p4, batch_shape = _normalize_latent_batch(coords)
        out = await self._batcher.acall(p4)
        return _restore_latent_batch(out, batch_shape)

    def close(self) -> None:
        self._batcher.close()


class LatentDiffField(Field):
    """Field backed by a single batched delta probe per call."""

    def __init__(
        self,
        client: Any,
        *,
        target_embeds: np.ndarray,
        base_embeds: np.ndarray,
        negative_prompt_embeds: Optional[np.ndarray] = None,
        timestep: int = 500,
        guidance_scale: float = 7.5,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        max_batch: int = 16,
        flush_ms: float = 1.0,
    ) -> None:
        self.client = client
        self.target_embeds = target_embeds
        self.base_embeds = base_embeds
        self.negative_prompt_embeds = negative_prompt_embeds
        self.timestep = int(timestep)
        self.guidance_scale = float(guidance_scale)
        self.model_id = str(model_id)

        def _probe(points: np.ndarray) -> np.ndarray:
            p4, batch_shape = _normalize_latent_batch(points)
            out = client.probe_delta_at(
                p4,
                target_embeds=target_embeds,
                base_embeds=base_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                timestep=self.timestep,
                guidance_scale=self.guidance_scale,
                model_id=self.model_id,
            )
            return _restore_latent_batch(out, batch_shape)

        self._batcher = AsyncLatentBatcher(_probe, max_batch=max_batch, flush_ms=flush_ms)
        super().__init__(self._batcher)

    async def acall(self, coords: np.ndarray) -> np.ndarray:
        p4, batch_shape = _normalize_latent_batch(coords)
        out = await self._batcher.acall(p4)
        return _restore_latent_batch(out, batch_shape)

    def close(self) -> None:
        self._batcher.close()


__all__ = [
    "LatentField",
    "LatentDiffField",
    "LatentFieldConfig",
]
