"""Benchmarks for latent-field operators.

Run:
    python -m shared.physics.bench_operators

This is self-contained. It does not call the remote server.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .operators import divergence, gradient, laplacian
from .types import Field


@dataclass
class CallStats:
    calls: int = 0
    points: int = 0


def counting_field(fn: Callable[[np.ndarray], np.ndarray], *, sleep_s: float = 0.0) -> tuple[Field, CallStats]:
    stats = CallStats()

    def wrapped(x: np.ndarray) -> np.ndarray:
        stats.calls += 1
        stats.points += int(np.asarray(x).shape[0]) if np.asarray(x).ndim >= 1 else 1
        if sleep_s:
            time.sleep(sleep_s)
        return fn(x)

    return Field(wrapped), stats


def _latent_points(batch: int, shape: tuple[int, int, int]) -> np.ndarray:
    c, h, w = shape
    return np.random.randn(batch, c, h, w).astype(np.float32)


def _flat_n(shape: tuple[int, ...]) -> int:
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _report(name: str, t_s: float, stats: CallStats) -> None:
    print(f"{name:28s}  {t_s*1000:9.2f} ms  calls={stats.calls:3d}  total_B={stats.points}")


def main() -> None:
    np.random.seed(0)

    # ------------------------------------------------------------------
    # Accuracy checks against analytic ground truths
    # ------------------------------------------------------------------
    latent_shape = (4, 16, 16)
    N = _flat_n(latent_shape)
    x = _latent_points(1, latent_shape)

    # Vector field V(x) = a*x => div(V) = a*N
    a = 0.5
    V = Field(lambda z: a * z)
    div_est = float(np.asarray(divergence(V, x, dx=1e-3, estimator="hutchinson", n_probes=32, point_ndim=3)).item())
    print("== Ground Truth Checks ==")
    print(f"div(a*x) truth={a*N:.3f}  est={div_est:.3f}  err={div_est - a*N:+.3f}")

    # Scalar phi(x) = sum x^2 => grad = 2x, lap = 2N
    # Use float64 accumulation so the *exact* finite-difference Laplacian is meaningful.
    phi = Field(lambda z: np.sum((z.astype(np.float64) ** 2), axis=(1, 2, 3)))
    grad_est = np.asarray(gradient(phi, x, dx=1e-3, method="central", point_ndim=3))
    grad_truth = 2.0 * x
    rel = float(np.linalg.norm(grad_est - grad_truth) / (np.linalg.norm(grad_truth) + 1e-12))
    print(f"grad(sum x^2) rel_l2_err={rel:.3e}")

    lap_est = float(np.asarray(laplacian(phi, x, dx=1e-3, estimator="hutchinson", n_probes=32, point_ndim=3)).item())
    print(f"lap(sum x^2) truth={2*N:.3f}  est={lap_est:.3f}  err={lap_est - 2*N:+.3f}")

    # Exact per-coordinate Laplacian is numerically fragile for float32-valued
    # fields; this uses float64 phi() so it reflects the operator math.
    lap_exact = float(np.asarray(laplacian(phi, x, dx=1e-3, estimator="exact", method="central", point_ndim=3)).item())
    print(f"lap(sum x^2) exact={lap_exact:.3f}  err={lap_exact - 2*N:+.3f}")

    # ------------------------------------------------------------------
    # Batching behavior: compare naive probe loop vs batched implementation
    # ------------------------------------------------------------------
    print("\n== Batching Benchmarks (simulated call overhead) ==")
    sleep_s = 0.005
    batch = 1
    x = _latent_points(batch, (4, 32, 32))
    N = _flat_n((4, 32, 32))

    # Use a cheap identity field; cost is dominated by per-call sleep.
    V_counted, st_batched = counting_field(lambda z: z, sleep_s=sleep_s)

    t0 = time.perf_counter()
    _ = divergence(V_counted, x, dx=1e-3, estimator="hutchinson", n_probes=8, point_ndim=3)
    t1 = time.perf_counter()
    _report("div(hutchinson) batched", t1 - t0, st_batched)

    # Naive baseline: 2 calls per probe (x+eps v, x-eps v).
    V_naive, st_naive = counting_field(lambda z: z, sleep_s=sleep_s)
    eps = 1e-3
    probes = np.random.choice((-1.0, 1.0), size=(8, batch, N)).astype(np.float32)
    x_flat = x.reshape(batch, N)
    t0 = time.perf_counter()
    acc = 0.0
    for p in probes:
        vp = (x_flat + eps * p).reshape(batch, 4, 32, 32)
        vm = (x_flat - eps * p).reshape(batch, 4, 32, 32)
        Vp = V_naive(vp).reshape(batch, N)
        Vm = V_naive(vm).reshape(batch, N)
        acc += (np.sum(p * (Vp - Vm), axis=-1) / (2.0 * eps)).item()
    _ = acc / probes.shape[0]
    t1 = time.perf_counter()
    _report("div(hutch) naive loop", t1 - t0, st_naive)


if __name__ == "__main__":
    main()
