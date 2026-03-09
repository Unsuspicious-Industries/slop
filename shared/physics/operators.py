"""Finite-difference operators over `Field`.

In this repo, fields are often defined over diffusion latents where both the
input coordinates and the output values are latent tensors, for example
`(B, 4, 64, 64)`. These operators treat a latent tensor as a single point in
R^N where N is the product of the latent shape.

Curl is only defined for 2D and 3D coordinate-vector fields.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Literal, Sequence, cast

from .types import Field


_DiffMethod = Literal["forward", "backward", "central"]
_Estimator = Literal["auto", "exact", "hutchinson"]

# When an operator needs many field evaluations at nearby points, batch them
# into stacked calls to reduce request overhead.
_BATCH_EVAL_CHUNK = 16


def _as_coords(coords: np.ndarray) -> np.ndarray:
    c = np.asarray(coords)
    if c.ndim == 0:
        raise ValueError(f"Expected coords with shape (..., dim); got scalar with shape {c.shape}")
    if c.dtype.kind not in ("f", "i", "u"):
        raise ValueError(f"Expected numeric coords array; got dtype {c.dtype}")
    if c.dtype.kind in ("i", "u"):
        c = c.astype(np.float32)
    return c


def _infer_point_ndim(coords: np.ndarray) -> int:
    """Infer how many trailing axes represent a single point.

    Heuristic:
    - If coords is a coordinate vector (shape (..., dim)), use point_ndim=1.
      This commonly shows up as 1D coords, or 2D with last axis 2/3.
    - Otherwise (diffusion latents), assume point_ndim=3 (C, H, W).
    """

    c = _as_coords(coords)
    if c.ndim == 1:
        return 1
    if c.ndim == 2 and c.shape[-1] in (2, 3):
        return 1
    if c.ndim >= 3:
        return 3
    return 1


def _split_batch_point(coords: np.ndarray, point_ndim: int | None) -> tuple[tuple[int, ...], tuple[int, ...]]:
    c = _as_coords(coords)
    pnd = _infer_point_ndim(c) if point_ndim is None else int(point_ndim)
    if pnd <= 0 or pnd > c.ndim:
        raise ValueError(f"Invalid point_ndim={pnd} for coords with shape {c.shape}")
    batch_shape = tuple(c.shape[:-pnd])
    point_shape = tuple(c.shape[-pnd:])
    if any(d <= 0 for d in point_shape):
        raise ValueError(f"Invalid point_shape inferred from coords: {point_shape}")
    return batch_shape, point_shape


def _flatten_point(coords: np.ndarray, point_ndim: int | None) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
    c = _as_coords(coords)
    batch_shape, point_shape = _split_batch_point(c, point_ndim)
    n = int(np.prod(point_shape))
    flat = c.reshape(*batch_shape, n)
    return flat, batch_shape, point_shape


def _ensure_scalar_output(phi_val: np.ndarray, batch_shape: tuple[int, ...]) -> np.ndarray:
    v = np.asarray(phi_val)
    # Allow scalar for non-batched evaluation.
    if batch_shape == () and v.shape == ():
        return v
    if v.shape != batch_shape:
        raise ValueError(f"Expected scalar field output with shape {batch_shape}, got shape {v.shape}")
    return v


def _ensure_vector_output(V_val: np.ndarray, coords_shape: tuple[int, ...]) -> np.ndarray:
    v = np.asarray(V_val)
    if v.shape != coords_shape:
        raise ValueError(f"Expected vector field output with same shape as coords {coords_shape}, got {v.shape}")
    return v


def _maybe_item(x: np.ndarray) -> np.ndarray | float:
    arr = np.asarray(x)
    return arr.item() if arr.size == 1 else arr


def _to_spacing_vector(
    dim: int,
    *,
    spacing: float | Sequence[float] | np.ndarray | None,
    dx: float,
    dy: float | None,
    dz: float | None,
) -> np.ndarray:
    if spacing is not None:
        # Avoid np.isscalar() for type narrowing; be explicit.
        if isinstance(spacing, (int, float, np.floating)):
            # Cast keeps static type-checkers happy; runtime is already narrowed.
            h = np.full((dim,), cast(float, spacing), dtype=float)
        else:
            arr = np.asarray(cast(Any, spacing), dtype=float).reshape(-1)
            if arr.shape != (dim,):
                raise ValueError(f"Expected spacing of length {dim}, got shape {arr.shape}")
            h = arr
        if np.any(h == 0):
            raise ValueError("All spacing values must be non-zero")
        return h

    # Back-compat path: dx/dy/dz. For high-dimensional latent space,
    # `dx` is interpreted as a uniform step for all coordinates.
    if dy is None and dz is None:
        if dx == 0:
            raise ValueError("Step size must be non-zero")
        return np.full((dim,), float(dx), dtype=float)

    parts: list[float] = [float(dx)]
    if dy is not None:
        parts.append(float(dy))
    if dz is not None:
        parts.append(float(dz))

    if any(p == 0 for p in parts):
        raise ValueError("Step size must be non-zero")
    if len(parts) == dim:
        return np.asarray(parts, dtype=float)
    if len(parts) == 1:
        return np.full((dim,), parts[0], dtype=float)
    raise ValueError(
        f"Got dx/dy/dz of length {len(parts)} but inferred dim={dim}. "
        "Pass `spacing=` to specify per-dimension steps."
    )


def _shift(coords: np.ndarray, coord_index: int, amount: float) -> np.ndarray:
    delta = np.zeros((coords.shape[-1],), dtype=float)
    delta[coord_index] = amount
    return coords + delta


def _shift_flattened(
    coords_flat: np.ndarray,
    point_shape: tuple[int, ...],
    index: int,
    amount: float,
) -> np.ndarray:
    x = np.array(coords_flat, copy=True)
    x[..., index] = x[..., index] + amount
    return x.reshape(*coords_flat.shape[:-1], *point_shape)


def _first_derivative(
    f: Field,
    coords: np.ndarray,
    coord_index: int,
    h: float,
    method: _DiffMethod,
) -> np.ndarray:
    if h == 0:
        raise ValueError("Step size must be non-zero")

    if method == "forward":
        return (f(_shift(coords, coord_index, h)) - f(coords)) / h
    if method == "backward":
        return (f(coords) - f(_shift(coords, coord_index, -h))) / h
    if method == "central":
        return (f(_shift(coords, coord_index, h)) - f(_shift(coords, coord_index, -h))) / (2.0 * h)

    raise ValueError(f"Unknown differencing method: {method}")


def _second_derivative(
    f: Field,
    coords: np.ndarray,
    coord_index: int,
    h: float,
    method: _DiffMethod,
) -> np.ndarray:
    if h == 0:
        raise ValueError("Step size must be non-zero")

    if method == "central":
        fp = f(_shift(coords, coord_index, h))
        f0 = f(coords)
        fm = f(_shift(coords, coord_index, -h))
        return (fp - 2.0 * f0 + fm) / (h * h)

    # Forward/backward second derivative using a one-sided stencil.
    if method == "forward":
        f0 = f(coords)
        f1 = f(_shift(coords, coord_index, h))
        f2 = f(_shift(coords, coord_index, 2.0 * h))
        return (f2 - 2.0 * f1 + f0) / (h * h)
    if method == "backward":
        f0 = f(coords)
        f1 = f(_shift(coords, coord_index, -h))
        f2 = f(_shift(coords, coord_index, -2.0 * h))
        return (f2 - 2.0 * f1 + f0) / (h * h)

    raise ValueError(f"Unknown differencing method: {method}")


def gradient(
    phi: Field,
    coords: np.ndarray | None = None,
    dx: float = 1e-3,
    dy: float | None = None,
    dz: float | None = None,
    *,
    spacing: float | Sequence[float] | np.ndarray | None = None,
    method: _DiffMethod = "central",
    point_ndim: int | None = None,
) -> Field | np.ndarray:
    """Gradient of a scalar field in high-dimensional latent space.

    For diffusion latents, this returns a tensor with the same shape as the
    input coords (per batch element), corresponding to dphi/dx_i for each
    latent coordinate x_i.

    `point_ndim` controls how many trailing axes belong to a single point; for
    SD-style latents (C, H, W) use 3. If omitted, a heuristic is used.
    """

    def _grad(coords: np.ndarray) -> np.ndarray:
        c = _as_coords(coords)
        x_flat, batch_shape, point_shape = _flatten_point(c, point_ndim)
        n = x_flat.shape[-1]
        h = _to_spacing_vector(int(n), spacing=spacing, dx=dx, dy=dy, dz=dz)
        base = _ensure_scalar_output(np.asarray(phi(c)), batch_shape)

        out = np.empty((*batch_shape, n), dtype=np.asarray(base).dtype)

        # Batch evaluations across coordinates to reduce call overhead.
        batch_prod = int(np.prod(batch_shape) or 1)
        x2 = x_flat.reshape(batch_prod, int(n))
        base2 = np.asarray(base).reshape(batch_prod)

        for start in range(0, int(n), _BATCH_EVAL_CHUNK):
            idx = np.arange(start, min(start + _BATCH_EVAL_CHUNK, int(n)), dtype=int)
            k = int(idx.size)
            r = np.arange(k)[:, None]
            b = np.arange(batch_prod)[None, :]
            cidx = idx[:, None]

            if method == "forward":
                xp = np.repeat(x2[None, :, :], k, axis=0)
                xp[r, b, cidx] += h[idx][:, None]
                xp_pts = xp.reshape(k * batch_prod, *point_shape)
                fp = np.asarray(phi(xp_pts)).reshape(k, batch_prod)
                out2 = ((fp - base2[None, :]) / h[idx][:, None]).astype(out.dtype, copy=False)
            elif method == "backward":
                xm = np.repeat(x2[None, :, :], k, axis=0)
                xm[r, b, cidx] -= h[idx][:, None]
                xm_pts = xm.reshape(k * batch_prod, *point_shape)
                fm = np.asarray(phi(xm_pts)).reshape(k, batch_prod)
                out2 = ((base2[None, :] - fm) / h[idx][:, None]).astype(out.dtype, copy=False)
            elif method == "central":
                xp = np.repeat(x2[None, :, :], k, axis=0)
                xm = np.repeat(x2[None, :, :], k, axis=0)
                xp[r, b, cidx] += h[idx][:, None]
                xm[r, b, cidx] -= h[idx][:, None]
                xp_pts = xp.reshape(k * batch_prod, *point_shape)
                xm_pts = xm.reshape(k * batch_prod, *point_shape)
                stacked = np.concatenate([xp_pts, xm_pts], axis=0)
                vals = np.asarray(phi(stacked)).reshape(2, k, batch_prod)
                fp = vals[0]
                fm = vals[1]
                out2 = ((fp - fm) / (2.0 * h[idx][:, None])).astype(out.dtype, copy=False)
            else:
                raise ValueError(f"Unknown differencing method: {method}")

            # Write chunk back (transpose to (batch_prod, k) then scatter)
            out_2d = out.reshape(batch_prod, int(n))
            out_2d[:, idx] = out2.T

        return out.reshape(*batch_shape, *point_shape)

    op = Field(_grad)
    return op if coords is None else op(coords)


def divergence(
    V: Field,
    coords: np.ndarray | None = None,
    dx: float = 1e-3,
    dy: float | None = None,
    dz: float | None = None,
    *,
    spacing: float | Sequence[float] | np.ndarray | None = None,
    method: _DiffMethod = "central",
    estimator: _Estimator = "auto",
    n_probes: int = 4,
    point_ndim: int | None = None,
) -> Field | np.ndarray | float:
    """Divergence (trace of Jacobian) of a diffusion-style vector field.

    Assumes V(coords) has the same shape as coords.

    For high-dimensional latents, computing the exact divergence is expensive
    (O(N) field evaluations). By default (`estimator="auto"`), this switches to
    a Hutchinson trace estimator when N is large.
    """

    def _div(coords: np.ndarray) -> np.ndarray:
        c = _as_coords(coords)
        x_flat, batch_shape, point_shape = _flatten_point(c, point_ndim)
        n = int(x_flat.shape[-1])
        h = _to_spacing_vector(n, spacing=spacing, dx=dx, dy=dy, dz=dz)

        use = estimator
        if use == "auto":
            use = "hutchinson" if n > 2048 else "exact"

        if use == "exact":
            # Only evaluate V(x) when needed for one-sided stencils.
            V0_flat: np.ndarray | None = None
            if method in ("forward", "backward"):
                V0 = _ensure_vector_output(np.asarray(V(c)), c.shape)
                V0_flat = V0.reshape(*batch_shape, n)

            dtype0 = (V0_flat.dtype if V0_flat is not None else x_flat.dtype)
            total = np.zeros(batch_shape, dtype=dtype0)

            # Batch evaluations across coordinates to reduce call overhead.
            batch_prod = int(np.prod(batch_shape) or 1)
            x2 = x_flat.reshape(batch_prod, n)
            V0_2d = V0_flat.reshape(batch_prod, n) if V0_flat is not None else None

            for start in range(0, n, _BATCH_EVAL_CHUNK):
                idx = np.arange(start, min(start + _BATCH_EVAL_CHUNK, n), dtype=int)
                k = int(idx.size)
                r = np.arange(k)[:, None]
                b = np.arange(batch_prod)[None, :]
                cidx = idx[:, None]

                if method == "central":
                    xp = np.repeat(x2[None, :, :], k, axis=0)
                    xm = np.repeat(x2[None, :, :], k, axis=0)
                    xp[r, b, cidx] += h[idx][:, None]
                    xm[r, b, cidx] -= h[idx][:, None]
                    xp_pts = xp.reshape(k * batch_prod, *point_shape)
                    xm_pts = xm.reshape(k * batch_prod, *point_shape)
                    stacked = np.concatenate([xp_pts, xm_pts], axis=0)
                    V_stacked = _ensure_vector_output(np.asarray(V(stacked)), (2 * k * batch_prod, *point_shape))
                    V_stacked = V_stacked.reshape(2, k, batch_prod, n)
                    Vp = V_stacked[0]
                    Vm = V_stacked[1]
                    Vp_diag = Vp[np.arange(k)[:, None], np.arange(batch_prod)[None, :], idx[:, None]]
                    Vm_diag = Vm[np.arange(k)[:, None], np.arange(batch_prod)[None, :], idx[:, None]]
                    chunk = np.sum((Vp_diag - Vm_diag) / (2.0 * h[idx][:, None]), axis=0)
                elif method == "forward":
                    if V0_2d is None:
                        raise RuntimeError("internal error: V0_2d is None for forward differencing")
                    xp = np.repeat(x2[None, :, :], k, axis=0)
                    xp[r, b, cidx] += h[idx][:, None]
                    xp_pts = xp.reshape(k * batch_prod, *point_shape)
                    Vp = _ensure_vector_output(np.asarray(V(xp_pts)), (k * batch_prod, *point_shape)).reshape(k, batch_prod, n)
                    Vp_diag = Vp[np.arange(k)[:, None], np.arange(batch_prod)[None, :], idx[:, None]]
                    V0_diag = V0_2d[:, idx].T
                    chunk = np.sum((Vp_diag - V0_diag) / h[idx][:, None], axis=0)
                elif method == "backward":
                    if V0_2d is None:
                        raise RuntimeError("internal error: V0_2d is None for backward differencing")
                    xm = np.repeat(x2[None, :, :], k, axis=0)
                    xm[r, b, cidx] -= h[idx][:, None]
                    xm_pts = xm.reshape(k * batch_prod, *point_shape)
                    Vm = _ensure_vector_output(np.asarray(V(xm_pts)), (k * batch_prod, *point_shape)).reshape(k, batch_prod, n)
                    Vm_diag = Vm[np.arange(k)[:, None], np.arange(batch_prod)[None, :], idx[:, None]]
                    V0_diag = V0_2d[:, idx].T
                    chunk = np.sum((V0_diag - Vm_diag) / h[idx][:, None], axis=0)
                else:
                    raise ValueError(f"Unknown differencing method: {method}")

                total = total + chunk.reshape(batch_shape)
            return total

        if use == "hutchinson":
            if n_probes <= 0:
                raise ValueError("n_probes must be >= 1")
            # Rademacher probes (+1/-1) yield an unbiased trace estimator.
            # Implementation detail: this is structured to call V() once on a
            # stacked batch of points (x+eps*v and x-eps*v).
            # Use uniform step if provided as dx; per-coordinate h is not used with probes.
            eps = np.asarray(h[0], dtype=x_flat.dtype).item()

            probes = np.random.choice((-1.0, 1.0), size=(int(n_probes), *batch_shape, n)).astype(
                x_flat.dtype, copy=False
            )
            xp = (x_flat[None, ...] + eps * probes).reshape(int(n_probes) * int(np.prod(batch_shape) or 1), *point_shape)
            xm = (x_flat[None, ...] - eps * probes).reshape(int(n_probes) * int(np.prod(batch_shape) or 1), *point_shape)

            stacked = np.concatenate([xp, xm], axis=0)
            V_stacked = np.asarray(V(stacked))
            V_stacked = V_stacked.reshape(2, int(n_probes), *batch_shape, *point_shape)
            Vp = V_stacked[0].reshape(int(n_probes), *batch_shape, n)
            Vm = V_stacked[1].reshape(int(n_probes), *batch_shape, n)

            # v^T J v ≈ v · (V(x+eps v) - V(x-eps v)) / (2 eps)
            # Accumulate in float64 to reduce cancellation.
            per_probe = np.sum(
                (probes * (Vp - Vm)).astype(np.float64, copy=False),
                axis=-1,
            ) / (2.0 * float(eps))
            out = np.mean(per_probe, axis=0)
            return out.astype(Vp.dtype, copy=False)

        raise ValueError(f"Unknown estimator: {estimator}")

    op = Field(_div)
    return op if coords is None else _maybe_item(op(coords))


def curl(
    V: Field,
    coords: np.ndarray | None = None,
    dx: float = 1.0,
    dy: float | None = None,
    dz: float | None = None,
    *,
    spacing: float | Sequence[float] | np.ndarray | None = None,
    method: _DiffMethod = "central",
    point_ndim: int | None = None,
) -> Field | np.ndarray:
    """Curl of a vector field.

    - If V is 2D (output (...,2)), returns scalar curl_z.
    - If V is 3D (output (...,3)), returns vector curl.

    Dimension is inferred from the field output at the evaluation coordinates.
    """

    def _curl(coords: np.ndarray) -> np.ndarray:
        # Curl is only meaningful for 2D/3D coordinate-vector style fields.
        c = _as_coords(coords)
        if point_ndim not in (None, 1):
            raise ValueError("curl() only supports point_ndim=1 coordinate-vector fields")
        if c.shape[-1] not in (2, 3):
            raise ValueError(f"Curl is only implemented for last-dim 2 or 3; got coords shape {c.shape}")

        n = int(c.shape[-1])
        h = _to_spacing_vector(n, spacing=spacing, dx=dx, dy=dy, dz=dz)

        def comp(ii: int) -> Field:
            return Field(lambda cc, j=ii: np.asarray(V(cc))[..., j])

        if n == 2:
            dVy_dx = _first_derivative(comp(1), c, 0, float(h[0]), method)
            dVx_dy = _first_derivative(comp(0), c, 1, float(h[1]), method)
            return dVy_dx - dVx_dy

        dVz_dy = _first_derivative(comp(2), c, 1, float(h[1]), method)
        dVy_dz = _first_derivative(comp(1), c, 2, float(h[2]), method)
        dVx_dz = _first_derivative(comp(0), c, 2, float(h[2]), method)
        dVz_dx = _first_derivative(comp(2), c, 0, float(h[0]), method)
        dVy_dx = _first_derivative(comp(1), c, 0, float(h[0]), method)
        dVx_dy = _first_derivative(comp(0), c, 1, float(h[1]), method)
        return np.stack([dVz_dy - dVy_dz, dVx_dz - dVz_dx, dVy_dx - dVx_dy], axis=-1)

    op = Field(_curl)
    return op if coords is None else op(coords)


def laplacian(
    phi: Field,
    coords: np.ndarray | None = None,
    dx: float = 1e-3,
    dy: float | None = None,
    dz: float | None = None,
    *,
    spacing: float | Sequence[float] | np.ndarray | None = None,
    method: _DiffMethod = "central",
    estimator: _Estimator = "auto",
    n_probes: int = 4,
    point_ndim: int | None = None,
) -> Field | np.ndarray | float:
    """Laplacian (trace of Hessian) of a scalar field.

    For high-dimensional latents, the exact Laplacian is expensive. With
    `estimator="auto"`, this uses a Hutchinson estimator for large N:

        tr(H) = E_v [ v^T H v ]
             approx (phi(x+eps v) - 2 phi(x) + phi(x-eps v)) / eps^2

    where v is a Rademacher probe.
    """

    def _lap(coords: np.ndarray) -> np.ndarray:
        c = _as_coords(coords)
        x_flat, batch_shape, point_shape = _flatten_point(c, point_ndim)
        n = int(x_flat.shape[-1])
        h = _to_spacing_vector(n, spacing=spacing, dx=dx, dy=dy, dz=dz)
        base = _ensure_scalar_output(np.asarray(phi(c)), batch_shape)

        use = estimator
        if use == "auto":
            use = "hutchinson" if n > 2048 else "exact"

        if use == "exact":
            total = np.zeros(batch_shape, dtype=np.asarray(base).dtype)

            batch_prod = int(np.prod(batch_shape) or 1)
            x2 = x_flat.reshape(batch_prod, n)
            base2 = np.asarray(base).reshape(batch_prod)

            for start in range(0, n, _BATCH_EVAL_CHUNK):
                idx = np.arange(start, min(start + _BATCH_EVAL_CHUNK, n), dtype=int)
                k = int(idx.size)
                r = np.arange(k)[:, None]
                b = np.arange(batch_prod)[None, :]
                cidx = idx[:, None]

                if method == "central":
                    xp = np.repeat(x2[None, :, :], k, axis=0)
                    xm = np.repeat(x2[None, :, :], k, axis=0)
                    xp[r, b, cidx] += h[idx][:, None]
                    xm[r, b, cidx] -= h[idx][:, None]
                    xp_pts = xp.reshape(k * batch_prod, *point_shape)
                    xm_pts = xm.reshape(k * batch_prod, *point_shape)
                    stacked = np.concatenate([xp_pts, xm_pts], axis=0)
                    vals = np.asarray(phi(stacked)).reshape(2, k, batch_prod)
                    fp = vals[0]
                    fm = vals[1]
                    chunk = np.sum((fp - 2.0 * base2[None, :] + fm) / (h[idx][:, None] ** 2), axis=0)
                elif method == "forward":
                    x1 = np.repeat(x2[None, :, :], k, axis=0)
                    x2s = np.repeat(x2[None, :, :], k, axis=0)
                    x1[r, b, cidx] += h[idx][:, None]
                    x2s[r, b, cidx] += (2.0 * h[idx])[:, None]
                    p1 = x1.reshape(k * batch_prod, *point_shape)
                    p2 = x2s.reshape(k * batch_prod, *point_shape)
                    stacked = np.concatenate([p1, p2], axis=0)
                    vals = np.asarray(phi(stacked)).reshape(2, k, batch_prod)
                    f1 = vals[0]
                    f2 = vals[1]
                    chunk = np.sum((f2 - 2.0 * f1 + base2[None, :]) / (h[idx][:, None] ** 2), axis=0)
                elif method == "backward":
                    x1 = np.repeat(x2[None, :, :], k, axis=0)
                    x2s = np.repeat(x2[None, :, :], k, axis=0)
                    x1[r, b, cidx] -= h[idx][:, None]
                    x2s[r, b, cidx] -= (2.0 * h[idx])[:, None]
                    p1 = x1.reshape(k * batch_prod, *point_shape)
                    p2 = x2s.reshape(k * batch_prod, *point_shape)
                    stacked = np.concatenate([p1, p2], axis=0)
                    vals = np.asarray(phi(stacked)).reshape(2, k, batch_prod)
                    f1 = vals[0]
                    f2 = vals[1]
                    chunk = np.sum((f2 - 2.0 * f1 + base2[None, :]) / (h[idx][:, None] ** 2), axis=0)
                else:
                    raise ValueError(f"Unknown differencing method: {method}")

                total = total + chunk.reshape(batch_shape)
            return total

        if use == "hutchinson":
            if n_probes <= 0:
                raise ValueError("n_probes must be >= 1")
            eps = np.asarray(h[0], dtype=x_flat.dtype).item()
            probes = np.random.choice((-1.0, 1.0), size=(int(n_probes), *batch_shape, n)).astype(
                x_flat.dtype, copy=False
            )
            xp = (x_flat[None, ...] + eps * probes).reshape(int(n_probes) * int(np.prod(batch_shape) or 1), *point_shape)
            xm = (x_flat[None, ...] - eps * probes).reshape(int(n_probes) * int(np.prod(batch_shape) or 1), *point_shape)

            stacked = np.concatenate([xp, xm], axis=0)
            vals = np.asarray(phi(stacked))
            vals = vals.reshape(2, int(n_probes), *batch_shape)
            fp = vals[0]
            fm = vals[1]
            # Compute in float64 to reduce catastrophic cancellation when
            # base is large compared to the finite-difference curvature term.
            per_probe = (
                fp.astype(np.float64, copy=False)
                - 2.0 * np.asarray(base, dtype=np.float64)[None, ...]
                + fm.astype(np.float64, copy=False)
            ) / (float(eps) * float(eps))
            out = np.mean(per_probe, axis=0)
            return out.astype(np.asarray(base).dtype, copy=False)

        raise ValueError(f"Unknown estimator: {estimator}")

    op = Field(_lap)
    return op if coords is None else _maybe_item(op(coords))


def gradient_3d(
    phi: Field,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    method: _DiffMethod = "central",
) -> Field | np.ndarray:
    # Back-compat wrapper.
    return gradient(phi, dx=dx, dy=dy, dz=dz, method=method)


def divergence_3d(
    V: Field,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    method: _DiffMethod = "central",
) -> Field | np.ndarray | float:
    # Back-compat wrapper.
    return divergence(V, dx=dx, dy=dy, dz=dz, method=method)


def curl_3d(
    V: Field,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    method: _DiffMethod = "central",
) -> Field | np.ndarray:
    # Back-compat wrapper.
    return curl(V, dx=dx, dy=dy, dz=dz, method=method)


def laplacian_3d(
    phi: Field,
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    method: _DiffMethod = "central",
) -> Field | np.ndarray | float:
    # Back-compat wrapper.
    return laplacian(phi, dx=dx, dy=dy, dz=dz, method=method)
