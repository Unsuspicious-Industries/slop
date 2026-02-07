"""Colonial visual grammar analysis.

Core idea:
    1. Generate N variations per prompt category (neutral / Arab / Palestinian / etc.)
    2. Capture the full latent trajectory for *every seed*
    3. Compute metrics **per-seed first**, then aggregate with mean ± SEM,
       bootstrap CIs, and permutation tests — so every number comes with an
       uncertainty estimate and a significance level.
    4. Average trajectories only for visualization; never use averages as the
       sole input to a statistical claim.

Statistical design:
    - "Seed" = one generation run (fixed prompt, fixed model, one random seed).
    - Each category has N_VARIATIONS seeds.
    - Divergence between categories A and B is computed for all N² (or matched N)
      seed pairs, giving a *distribution* of divergence curves, not one curve.
    - Lock-in step, channel diffs, flow diffs — all reported as mean ± CI.
    - Permutation test: shuffle category labels, recompute metric, build null
      distribution → p-value.

Shapes reference (SD v1.5 @ 512×512):
    latent:     (steps+1, batch, 4, 64, 64)   batch=2 for CFG (uncond+cond)
    noise_pred: (steps, batch, 4, 64, 64)
    flow:       (steps, batch, 4, 64, 64)    — velocity = latent[t+1] - latent[t]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _flatten_step(latents: np.ndarray) -> np.ndarray:
    """Flatten spatial dims at each step: (T, batch, C, H, W) → (T, D).

    Averages over the batch dim first (handles CFG uncond+cond).
    """
    per_step = latents.mean(axis=1)          # (T, C, H, W)
    return per_step.reshape(per_step.shape[0], -1)  # (T, D)


# ---------------------------------------------------------------------------
# Trajectory averaging (for visualization — NOT for statistics)
# ---------------------------------------------------------------------------

def average_trajectories(
    latent_stacks: List[np.ndarray],
) -> np.ndarray:
    """Average multiple latent trajectories element-wise.

    Each entry should have shape (steps+1, batch, C, H, W).
    All entries must share the same shape.

    NOTE: Use this for visualizing the "mean trajectory".  For any statistical
    claim, compute the metric per-seed and then aggregate (see batch_* funcs).

    Returns:
        Averaged trajectory, same shape as individual entries.
    """
    if not latent_stacks:
        raise ValueError("Need at least one trajectory")

    ref_shape = latent_stacks[0].shape
    for i, t in enumerate(latent_stacks):
        if t.shape != ref_shape:
            raise ValueError(
                f"Shape mismatch: trajectory 0 has {ref_shape}, "
                f"trajectory {i} has {t.shape}"
            )

    return np.mean(latent_stacks, axis=0)


# ---------------------------------------------------------------------------
# Flow fields
# ---------------------------------------------------------------------------

def compute_flow_field(latents: np.ndarray) -> np.ndarray:
    """Compute per-step velocity field from a latent trajectory.

    flow[t] = latent[t+1] − latent[t]

    Args:
        latents: (steps+1, batch, C, H, W)

    Returns:
        Flow field of shape (steps, batch, C, H, W)
    """
    return np.diff(latents, axis=0)


def average_flow_field(
    latent_stacks: List[np.ndarray],
) -> np.ndarray:
    """Compute the mean flow field across multiple trajectory sets.

    1. Compute flow for each trajectory
    2. Average element-wise

    Returns:
        Average flow, shape (steps, batch, C, H, W)
    """
    flows = [compute_flow_field(t) for t in latent_stacks]
    return np.mean(flows, axis=0)


def flow_difference(
    flow_a: np.ndarray,
    flow_b: np.ndarray,
) -> np.ndarray:
    """Element-wise difference between two averaged flow fields.

    Positive regions = flow_a pushes harder in that direction.
    """
    n = min(flow_a.shape[0], flow_b.shape[0])
    return flow_a[:n] - flow_b[:n]


def flow_magnitude(flow: np.ndarray) -> np.ndarray:
    """Per-pixel flow magnitude across channels.

    Args:
        flow: (steps, batch, C, H, W)

    Returns:
        Magnitude map (steps, batch, H, W)
    """
    return np.sqrt(np.sum(flow ** 2, axis=2))


# =====================================================================
#  BATCH-AWARE STATISTICAL INFRASTRUCTURE
# =====================================================================

def _bootstrap_ci(
    samples: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Bootstrap confidence interval for the mean of `samples`.

    Args:
        samples: (N, ...) — first axis is the sample axis
        n_boot: Number of bootstrap resamples
        ci: Confidence level (e.g. 0.95 for 95%)

    Returns:
        (lower, upper) each shaped like samples[0]
    """
    if rng is None:
        rng = np.random.default_rng(42)
    n = samples.shape[0]
    alpha = (1.0 - ci) / 2.0

    # Bootstrap resamples of the mean
    boot_means = np.empty((n_boot, *samples.shape[1:]), dtype=samples.dtype)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = samples[idx].mean(axis=0)

    lower = np.percentile(boot_means, 100 * alpha, axis=0)
    upper = np.percentile(boot_means, 100 * (1 - alpha), axis=0)
    return lower, upper


def _sem(samples: np.ndarray) -> np.ndarray:
    """Standard error of the mean along axis 0."""
    return np.std(samples, axis=0, ddof=1) / np.sqrt(samples.shape[0])


# ---------------------------------------------------------------------------
# Batch divergence: the core metric
# ---------------------------------------------------------------------------

@dataclass
class BatchDivergenceResult:
    """Per-step divergence between two categories, with full statistics.

    All arrays have shape (steps+1,) or (steps,).
    """
    mean: np.ndarray          # Mean L2 distance per step
    std: np.ndarray           # Std across seeds
    sem: np.ndarray           # Standard error of the mean
    ci_lower: np.ndarray      # Lower bound of 95% CI
    ci_upper: np.ndarray      # Upper bound of 95% CI
    per_seed: np.ndarray      # (N_pairs, steps+1) — all per-seed curves
    lock_in_mean: float       # Mean lock-in step across seeds
    lock_in_std: float        # Std of lock-in step
    p_value: float            # Permutation test p-value
    n_seeds_a: int
    n_seeds_b: int


def _single_divergence_curve(lat_a: np.ndarray, lat_b: np.ndarray) -> np.ndarray:
    """L2 distance at each step between two single-seed trajectories.

    Args:
        lat_a, lat_b: (steps+1, batch, C, H, W)

    Returns:
        (steps+1,) — L2 distance per step
    """
    n = min(lat_a.shape[0], lat_b.shape[0])
    a_flat = _flatten_step(lat_a[:n])  # (T, D)
    b_flat = _flatten_step(lat_b[:n])  # (T, D)
    return np.linalg.norm(a_flat - b_flat, axis=1)


def _find_lock_in(distances: np.ndarray, threshold: float = 0.5) -> int:
    """Step where cumulative distance first exceeds `threshold` of final."""
    target = distances[-1] * threshold
    candidates = np.where(distances >= target)[0]
    return int(candidates[0]) if len(candidates) > 0 else -1


def batch_divergence(
    stacks_a: List[np.ndarray],
    stacks_b: List[np.ndarray],
    mode: str = "matched",
    n_perms: int = 1000,
    ci: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> BatchDivergenceResult:
    """Batch-averaged trajectory divergence with full statistics.

    Computes divergence for every seed pair, then aggregates.

    Args:
        stacks_a: List of trajectories for category A, each (steps+1, batch, C, H, W)
        stacks_b: List of trajectories for category B
        mode: "matched" — pair seeds by index (requires len(a)==len(b)).
              "all_pairs" — every a × b combination (N² pairs, slower).
        n_perms: Number of permutation test iterations
        ci: Confidence interval level (0.95 → 95% CI)

    Returns:
        BatchDivergenceResult with mean, std, CI, per-seed curves, p-value
    """
    if rng is None:
        rng = np.random.default_rng(42)

    na, nb = len(stacks_a), len(stacks_b)

    # ── Build per-seed divergence curves ──
    if mode == "matched":
        n_pairs = min(na, nb)
        per_seed = np.stack([
            _single_divergence_curve(stacks_a[i], stacks_b[i])
            for i in range(n_pairs)
        ])  # (N, steps+1)
    else:  # all_pairs
        curves = []
        for i in range(na):
            for j in range(nb):
                curves.append(_single_divergence_curve(stacks_a[i], stacks_b[j]))
        per_seed = np.stack(curves)  # (N*M, steps+1)

    mean_curve = per_seed.mean(axis=0)
    std_curve = per_seed.std(axis=0, ddof=1) if per_seed.shape[0] > 1 else np.zeros_like(mean_curve)
    sem_curve = _sem(per_seed)

    # Bootstrap CI on the mean curve
    ci_lower, ci_upper = _bootstrap_ci(per_seed, ci=ci, rng=rng)

    # Per-seed lock-in steps
    lock_ins = np.array([_find_lock_in(curve) for curve in per_seed])
    valid_lock_ins = lock_ins[lock_ins >= 0]
    lock_in_mean = float(valid_lock_ins.mean()) if len(valid_lock_ins) > 0 else -1.0
    lock_in_std = float(valid_lock_ins.std(ddof=1)) if len(valid_lock_ins) > 1 else 0.0

    # ── Permutation test ──
    # Null hypothesis: category labels don't matter.
    # Pool all trajectories, shuffle labels, recompute mean final divergence.
    pooled = stacks_a + stacks_b
    observed_stat = mean_curve[-1]  # Mean final-step divergence

    n_exceed = 0
    for _ in range(n_perms):
        rng.shuffle(pooled)  # in-place shuffle of the list
        perm_a = pooled[:na]
        perm_b = pooled[na:na + nb]
        n_matched = min(len(perm_a), len(perm_b))
        perm_final = np.mean([
            _single_divergence_curve(perm_a[i], perm_b[i])[-1]
            for i in range(n_matched)
        ])
        if perm_final >= observed_stat:
            n_exceed += 1
    # Restore original order (shuffle was in-place)
    # The lists stacks_a/stacks_b are not mutated because we copied into pooled

    p_value = (n_exceed + 1) / (n_perms + 1)  # +1 for continuity correction

    return BatchDivergenceResult(
        mean=mean_curve,
        std=std_curve,
        sem=sem_curve,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        per_seed=per_seed,
        lock_in_mean=lock_in_mean,
        lock_in_std=lock_in_std,
        p_value=p_value,
        n_seeds_a=na,
        n_seeds_b=nb,
    )


# ---------------------------------------------------------------------------
# Batch flow difference
# ---------------------------------------------------------------------------

@dataclass
class BatchFlowResult:
    """Per-step flow difference magnitude with statistics."""
    mean: np.ndarray           # (steps,) — mean |Δflow| per step
    std: np.ndarray            # (steps,)
    sem: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    per_seed: np.ndarray       # (N, steps) — per-seed curves


def batch_flow_difference(
    stacks_a: List[np.ndarray],
    stacks_b: List[np.ndarray],
    ci: float = 0.95,
) -> BatchFlowResult:
    """Per-seed flow difference magnitude, then aggregate.

    For each matched pair (seed_i_a, seed_i_b):
        1. Compute flow for each
        2. Compute |flow_a - flow_b| averaged over spatial dims
        3. This gives a scalar per step

    Returns:
        BatchFlowResult with mean, std, CI, per-seed curves
    """
    n = min(len(stacks_a), len(stacks_b))
    per_seed_curves = []

    for i in range(n):
        fa = compute_flow_field(stacks_a[i])
        fb = compute_flow_field(stacks_b[i])
        n_steps = min(fa.shape[0], fb.shape[0])
        diff = fa[:n_steps] - fb[:n_steps]
        mag = flow_magnitude(diff).mean(axis=(1, 2, 3))  # (steps,)
        per_seed_curves.append(mag)

    per_seed = np.stack(per_seed_curves)   # (N, steps)
    mean_curve = per_seed.mean(axis=0)
    std_curve = per_seed.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean_curve)
    sem_curve = _sem(per_seed)
    ci_lower, ci_upper = _bootstrap_ci(per_seed, ci=ci)

    return BatchFlowResult(
        mean=mean_curve,
        std=std_curve,
        sem=sem_curve,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        per_seed=per_seed,
    )


# ---------------------------------------------------------------------------
# Batch channel difference
# ---------------------------------------------------------------------------

@dataclass
class BatchChannelResult:
    """Per-channel difference with seed-level statistics."""
    mean: np.ndarray          # (C,) — mean per-channel diff
    std: np.ndarray           # (C,)
    sem: np.ndarray
    ci_lower: np.ndarray      # (C,)
    ci_upper: np.ndarray
    per_seed: np.ndarray      # (N, C) — per-seed channel diffs


def batch_channel_difference(
    stacks_a: List[np.ndarray],
    stacks_b: List[np.ndarray],
    step: int = -1,
    ci: float = 0.95,
) -> BatchChannelResult:
    """Per-seed per-channel absolute difference at a given step.

    For each seed pair:
        diff_c = mean(|a[step, :, c, :, :] - b[step, :, c, :, :]|)  per channel c

    Returns:
        BatchChannelResult with (C,) arrays for mean, std, CI
    """
    n = min(len(stacks_a), len(stacks_b))
    per_seed = []

    for i in range(n):
        a = stacks_a[i][step].mean(axis=0)  # (C, H, W)
        b = stacks_b[i][step].mean(axis=0)
        ch_diff = np.mean(np.abs(a - b), axis=(1, 2))  # (C,)
        per_seed.append(ch_diff)

    per_seed = np.stack(per_seed)  # (N, C)
    mean = per_seed.mean(axis=0)
    std = per_seed.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    sem = _sem(per_seed)
    ci_lower, ci_upper = _bootstrap_ci(per_seed, ci=ci)

    return BatchChannelResult(
        mean=mean, std=std, sem=sem,
        ci_lower=ci_lower, ci_upper=ci_upper,
        per_seed=per_seed,
    )


# ---------------------------------------------------------------------------
# Batch spatial heatmap — per-seed variance of spatial differences
# ---------------------------------------------------------------------------

@dataclass
class BatchSpatialResult:
    """Spatial heatmaps with seed-level statistics."""
    mean: np.ndarray          # (H, W) — mean spatial diff
    std: np.ndarray           # (H, W) — std across seeds
    snr: np.ndarray           # (H, W) — signal-to-noise ratio = mean / std
    per_seed: np.ndarray      # (N, H, W) — individual heatmaps


def batch_spatial_heatmap(
    stacks_a: List[np.ndarray],
    stacks_b: List[np.ndarray],
    step: int = -1,
) -> BatchSpatialResult:
    """Per-seed spatial difference heatmaps, then compute mean/std/SNR.

    SNR > 1 means the spatial difference at that pixel is reliable (signal > noise).

    Returns:
        BatchSpatialResult with mean, std, SNR heatmaps
    """
    n = min(len(stacks_a), len(stacks_b))
    per_seed = []

    for i in range(n):
        a = stacks_a[i][step].mean(axis=0)  # (C, H, W)
        b = stacks_b[i][step].mean(axis=0)
        diff = a - b
        heatmap = np.sqrt(np.sum(diff ** 2, axis=0))  # (H, W)
        per_seed.append(heatmap)

    per_seed = np.stack(per_seed)  # (N, H, W)
    mean = per_seed.mean(axis=0)
    std = per_seed.std(axis=0, ddof=1) if n > 1 else np.ones_like(mean)
    snr = np.where(std > 1e-10, mean / std, 0.0)

    return BatchSpatialResult(mean=mean, std=std, snr=snr, per_seed=per_seed)


# ---------------------------------------------------------------------------
# Batch PCA — fit on ALL seeds, not just averages
# ---------------------------------------------------------------------------

def batch_inter_category_pca(
    category_stacks: Dict[str, List[np.ndarray]],
    n_components: int = 4,
    sample_steps: int = 10,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], PCA, np.ndarray]:
    """PCA on individual seed trajectories (not averages) for robust separation.

    Fits PCA on sampled timesteps from ALL seeds across ALL categories.
    Then projects each seed's full trajectory onto the PC axes.

    Returns:
        avg_projections: {cat: (steps+1, n_components)} — mean projection per category
        per_seed_projections: {cat: (N_seeds, steps+1, n_components)}
        pca: Fitted PCA object
        explained_variance_ratio: array
    """
    # ── Collect training points: sampled steps from every seed ──
    all_points = []
    for name, stacks in category_stacks.items():
        for lat in stacks:
            n_steps = lat.shape[0]
            step_idx = np.linspace(0, n_steps - 1, min(sample_steps, n_steps), dtype=int)
            for s in step_idx:
                all_points.append(lat[s].mean(axis=0).flatten())  # (D,)

    all_points = np.stack(all_points)
    pca = PCA(n_components=min(n_components, all_points.shape[0], all_points.shape[1]))
    pca.fit(all_points)

    # ── Project each seed's full trajectory ──
    per_seed_projections: Dict[str, np.ndarray] = {}
    avg_projections: Dict[str, np.ndarray] = {}

    for name, stacks in category_stacks.items():
        seed_projs = []
        for lat in stacks:
            flat = _flatten_step(lat)  # (T, D)
            proj = pca.transform(flat)  # (T, n_components)
            seed_projs.append(proj)
        per_seed_projections[name] = np.stack(seed_projs)  # (N, T, n_comp)
        avg_projections[name] = per_seed_projections[name].mean(axis=0)  # (T, n_comp)

    return avg_projections, per_seed_projections, pca, pca.explained_variance_ratio_


# =====================================================================
#  DIVERGENCE RATE ANALYSIS
# =====================================================================
# The absolute divergence from neutral isn't interesting on its own —
# related prompts ("Arab in Jaffa", "Jesus in Jaffa") will all diverge.
# What matters is the *rate profile*: d(divergence)/dt at each step,
# which categories accelerate earliest, and where rates differ.

@dataclass
class DivergenceRateResult:
    """Per-step divergence rate (derivative) with batch statistics.

    Rates are computed as first differences of the divergence curve,
    so rate[t] ≈ divergence[t+1] - divergence[t].
    """
    mean: np.ndarray           # (steps,) — mean rate per step
    std: np.ndarray            # (steps,)
    sem: np.ndarray
    ci_lower: np.ndarray
    ci_upper: np.ndarray
    per_seed: np.ndarray       # (N, steps) — per-seed rate curves
    peak_step_mean: float      # Mean step of peak divergence rate
    peak_step_std: float
    accel_half_mean: float     # Mean step where cumulative rate reaches 50%
    accel_half_std: float


def batch_divergence_rate(
    stacks_a: List[np.ndarray],
    stacks_b: List[np.ndarray],
    smooth_window: int = 3,
    ci: float = 0.95,
) -> DivergenceRateResult:
    """Per-seed divergence *rate* (velocity of separation), then aggregate.

    For each matched seed pair:
        1. Compute L2 divergence curve
        2. Take first difference → rate
        3. Optionally smooth with a running mean

    The rate tells you *when* the model is actively pushing categories apart,
    not just how far apart they ended up.

    Args:
        stacks_a, stacks_b: Per-seed trajectory lists
        smooth_window: Running-mean window for rate smoothing (1 = no smoothing)
        ci: Confidence interval level

    Returns:
        DivergenceRateResult with rate curves, peak step, acceleration midpoint
    """
    n = min(len(stacks_a), len(stacks_b))
    per_seed_rates = []
    peak_steps = []
    accel_halves = []

    for i in range(n):
        # Divergence curve for this seed pair
        div_curve = _single_divergence_curve(stacks_a[i], stacks_b[i])  # (T,)
        # Rate = first difference
        rate = np.diff(div_curve)  # (T-1,)

        # Smooth
        if smooth_window > 1 and len(rate) >= smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            rate = np.convolve(rate, kernel, mode="same")

        per_seed_rates.append(rate)

        # Peak: step of maximum divergence rate
        peak_steps.append(int(np.argmax(rate)))

        # Acceleration midpoint: step where cumulative rate reaches 50% of total
        cum_rate = np.cumsum(np.maximum(rate, 0))  # only positive (diverging) rates
        if cum_rate[-1] > 0:
            half_target = cum_rate[-1] * 0.5
            half_idx = np.searchsorted(cum_rate, half_target)
            accel_halves.append(int(min(half_idx, len(rate) - 1)))
        else:
            accel_halves.append(-1)

    per_seed = np.stack(per_seed_rates)  # (N, T-1)
    mean = per_seed.mean(axis=0)
    std = per_seed.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean)
    sem = _sem(per_seed)
    ci_lower, ci_upper = _bootstrap_ci(per_seed, ci=ci)

    peak_arr = np.array(peak_steps, dtype=float)
    accel_arr = np.array([a for a in accel_halves if a >= 0], dtype=float)

    return DivergenceRateResult(
        mean=mean,
        std=std,
        sem=sem,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        per_seed=per_seed,
        peak_step_mean=float(peak_arr.mean()),
        peak_step_std=float(peak_arr.std(ddof=1)) if len(peak_arr) > 1 else 0.0,
        accel_half_mean=float(accel_arr.mean()) if len(accel_arr) > 0 else -1.0,
        accel_half_std=float(accel_arr.std(ddof=1)) if len(accel_arr) > 1 else 0.0,
    )


def compare_divergence_rates(
    rate_results: Dict[str, DivergenceRateResult],
) -> Dict[str, Any]:
    """Compare rate profiles across categories to find which diverges faster.

    Returns:
        Dict with rate ratios, peak ordering, and pairwise rate differences.
    """
    cats = list(rate_results.keys())
    report: Dict[str, Any] = {}

    # Peak step ordering: which category locks in first?
    peak_order = sorted(cats, key=lambda c: rate_results[c].peak_step_mean)
    report["peak_order"] = peak_order
    report["peak_steps"] = {
        c: {"mean": rate_results[c].peak_step_mean,
            "std": rate_results[c].peak_step_std}
        for c in cats
    }

    # Acceleration midpoint ordering
    accel_order = sorted(
        cats,
        key=lambda c: rate_results[c].accel_half_mean
        if rate_results[c].accel_half_mean >= 0 else 1e6
    )
    report["accel_order"] = accel_order
    report["accel_midpoints"] = {
        c: {"mean": rate_results[c].accel_half_mean,
            "std": rate_results[c].accel_half_std}
        for c in cats
    }

    # Pairwise: for each pair of categories, at each step, which has higher rate?
    # Use per-seed data for a proper test
    if len(cats) >= 2:
        pairwise = {}
        for i, ca in enumerate(cats):
            for cb in cats[i + 1:]:
                ra = rate_results[ca].per_seed  # (Na, T)
                rb = rate_results[cb].per_seed  # (Nb, T)
                n_common = min(ra.shape[0], rb.shape[0])
                # Paired difference in rate at each step
                diff = ra[:n_common] - rb[:n_common]  # (N, T) — positive = a diverges faster
                mean_diff = diff.mean(axis=0)
                sem_diff = _sem(diff)
                # t-statistic per step
                t_stat = np.where(sem_diff > 1e-10, mean_diff / sem_diff, 0.0)
                pairwise[f"{ca}_vs_{cb}"] = {
                    "mean_rate_diff": mean_diff,
                    "sem": sem_diff,
                    "t_stat": t_stat,
                    "faster_early": ca if rate_results[ca].peak_step_mean < rate_results[cb].peak_step_mean else cb,
                }
        report["pairwise"] = pairwise

    return report


# =====================================================================
#  SPATIAL MOTIF EXTRACTION
# =====================================================================
# When the model puts Arabs in front of gates/arches, it's activating a
# specific spatial pattern in the latent space. We can extract that
# pattern as a vector ("the gate direction") and then measure how much
# any generation activates it.

@dataclass
class MotifDirection:
    """A learned latent-space direction corresponding to a visual motif.

    The direction is a (C, H, W) tensor in latent space.  Projecting any
    latent onto it gives a scalar "activation" of the motif.
    """
    direction: np.ndarray       # (C, H, W) — unit vector in latent space
    spatial_mask: np.ndarray    # (H, W) — where the motif concentrates (L2 over C)
    magnitude: float            # Average magnitude of the motif signal
    onset_step: int             # Step where motif activation exceeds 50% of final
    name: str                   # Human label, e.g. "gate_arch"


@dataclass
class MotifExtractionResult:
    """Result of extracting a visual motif from category differences."""
    motif: MotifDirection
    per_seed_activation: np.ndarray   # (N, steps+1) — how much each seed activates it
    activation_mean: np.ndarray       # (steps+1,)
    activation_ci_lower: np.ndarray   # (steps+1,)
    activation_ci_upper: np.ndarray   # (steps+1,)
    spatial_snr: np.ndarray           # (H, W) — reliability of spatial pattern
    top_channels: np.ndarray          # Channels ranked by contribution


def extract_motif_direction(
    stacks_with: List[np.ndarray],
    stacks_without: List[np.ndarray],
    step: int = -1,
    name: str = "motif",
    ci: float = 0.95,
) -> MotifExtractionResult:
    """Extract the latent-space direction responsible for a visual motif.

    Given two sets of generations — one that contains the motif (e.g., gates)
    and one that doesn't — compute the *average difference* in latent space.
    This difference vector IS the motif direction.

    The key insight: if the model consistently puts Arabs in front of gates but
    not generic people, the per-seed difference at the final latent step will
    have a systematic component (the gate pattern) and a random component
    (per-seed noise). Averaging isolates the systematic part; per-seed variance
    tells us how reliable it is.

    Args:
        stacks_with: Trajectories where the motif appears (e.g. "Arab person")
        stacks_without: Trajectories without the motif (e.g. "a person")
        step: Which step to extract the motif at (-1 = final image)
        name: Human-readable motif name
        ci: Confidence interval level

    Returns:
        MotifExtractionResult with the direction, per-seed activations, spatial map
    """
    n = min(len(stacks_with), len(stacks_without))

    # ── Per-seed difference vectors at the target step ──
    per_seed_diffs = []
    for i in range(n):
        a = stacks_with[i][step].mean(axis=0)     # (C, H, W)
        b = stacks_without[i][step].mean(axis=0)   # (C, H, W)
        per_seed_diffs.append(a - b)

    per_seed_diffs = np.stack(per_seed_diffs)  # (N, C, H, W)

    # ── Mean difference = motif direction (before normalization) ──
    mean_diff = per_seed_diffs.mean(axis=0)  # (C, H, W)
    magnitude = float(np.linalg.norm(mean_diff))
    direction = mean_diff / (magnitude + 1e-10)  # Unit vector

    # ── Spatial mask: where in H×W does the motif concentrate? ──
    spatial_mask = np.sqrt(np.sum(mean_diff ** 2, axis=0))  # (H, W)

    # ── Spatial SNR ──
    std_diff = per_seed_diffs.std(axis=0, ddof=1) if n > 1 else np.ones_like(mean_diff)
    spatial_std = np.sqrt(np.sum(std_diff ** 2, axis=0))  # (H, W)
    spatial_snr = np.where(spatial_std > 1e-10, spatial_mask / spatial_std, 0.0)

    # ── Channel ranking ──
    channel_contribution = np.mean(np.abs(mean_diff), axis=(1, 2))  # (C,)
    top_channels = np.argsort(channel_contribution)[::-1]

    # ── Per-seed activation curves: project each seed's full trajectory
    #    onto the motif direction at every step ──
    direction_flat = direction.flatten()  # (D,)
    per_seed_activation = []

    for i in range(n):
        traj = stacks_with[i]  # (T, batch, C, H, W)
        T = traj.shape[0]
        per_step = traj.mean(axis=1)  # (T, C, H, W)
        flat = per_step.reshape(T, -1)  # (T, D)
        activation = flat @ direction_flat  # (T,) — projection onto motif direction
        per_seed_activation.append(activation)

    per_seed_activation = np.stack(per_seed_activation)  # (N, T)
    activation_mean = per_seed_activation.mean(axis=0)
    act_ci_lower, act_ci_upper = _bootstrap_ci(per_seed_activation, ci=ci)

    # ── Onset step: when does motif activation exceed 50% of final? ──
    # Use mean activation relative to its own range
    act_range = activation_mean - activation_mean[0]
    if abs(act_range[-1]) > 1e-10:
        normalized = act_range / act_range[-1]
        onset_candidates = np.where(np.abs(normalized) >= 0.5)[0]
        onset_step = int(onset_candidates[0]) if len(onset_candidates) > 0 else -1
    else:
        onset_step = -1

    return MotifExtractionResult(
        motif=MotifDirection(
            direction=direction,
            spatial_mask=spatial_mask,
            magnitude=magnitude,
            onset_step=onset_step,
            name=name,
        ),
        per_seed_activation=per_seed_activation,
        activation_mean=activation_mean,
        activation_ci_lower=act_ci_lower,
        activation_ci_upper=act_ci_upper,
        spatial_snr=spatial_snr,
        top_channels=top_channels,
    )


def measure_motif_activation(
    stacks: List[np.ndarray],
    motif: MotifDirection,
) -> Tuple[np.ndarray, np.ndarray]:
    """Measure how much a set of trajectories activates a known motif direction.

    Given a previously extracted motif (e.g. the "gate direction"), project
    new trajectories onto it to measure activation.

    Args:
        stacks: Trajectories to measure
        motif: Previously extracted MotifDirection

    Returns:
        per_seed: (N, T) per-seed activation curves
        mean: (T,) mean activation
    """
    direction_flat = motif.direction.flatten()
    per_seed = []

    for lat in stacks:
        T = lat.shape[0]
        per_step = lat.mean(axis=1)  # (T, C, H, W)
        flat = per_step.reshape(T, -1)
        per_seed.append(flat @ direction_flat)

    per_seed = np.stack(per_seed)  # (N, T)
    return per_seed, per_seed.mean(axis=0)


def extract_motif_components(
    stacks_with: List[np.ndarray],
    stacks_without: List[np.ndarray],
    step: int = -1,
    n_components: int = 4,
) -> Tuple[np.ndarray, PCA, np.ndarray]:
    """PCA on per-seed difference vectors to find multiple motif components.

    Sometimes the visual motif isn't a single direction but has sub-components
    (e.g., gate shape + archway texture + shadow pattern). This finds the
    principal axes of variation in the category difference.

    Args:
        stacks_with, stacks_without: Trajectory lists
        step: Which step
        n_components: Number of PCA components

    Returns:
        components: (n_components, C, H, W) — motif sub-directions
        pca: Fitted PCA
        explained_variance_ratio: How much each component explains
    """
    n = min(len(stacks_with), len(stacks_without))
    diffs = []

    for i in range(n):
        a = stacks_with[i][step].mean(axis=0)
        b = stacks_without[i][step].mean(axis=0)
        diffs.append((a - b).flatten())

    diffs = np.stack(diffs)  # (N, D)
    pca = PCA(n_components=min(n_components, n, diffs.shape[1]))
    pca.fit(diffs)

    C, H, W = stacks_with[0][0].shape[1], stacks_with[0][0].shape[3], stacks_with[0][0].shape[4]
    components = pca.components_.reshape(-1, C, H, W)

    return components, pca, pca.explained_variance_ratio_


# =====================================================================
# LEGACY COMPAT — single-trajectory versions for visualization
# =====================================================================

def spatial_difference_heatmap(
    avg_latents_a: np.ndarray,
    avg_latents_b: np.ndarray,
    step: int = -1,
    reduce: str = "l2",
) -> np.ndarray:
    """Compute a 2D heatmap of where in the H×W latent grid two categories differ.

    Args:
        avg_latents_a: (steps+1, batch, C, H, W) — category A average
        avg_latents_b: (steps+1, batch, C, H, W) — category B average
        step: Which timestep to compare (-1 = final)
        reduce: "l2" for channel-wise L2 norm, "max" for max channel diff

    Returns:
        Heatmap of shape (H, W)
    """
    a = avg_latents_a[step].mean(axis=0)  # (C, H, W) after averaging batch
    b = avg_latents_b[step].mean(axis=0)

    diff = a - b  # (C, H, W)

    if reduce == "l2":
        return np.sqrt(np.sum(diff ** 2, axis=0))
    elif reduce == "max":
        return np.max(np.abs(diff), axis=0)
    else:
        return np.mean(np.abs(diff), axis=0)


def temporal_heatmap_sequence(
    avg_latents_a: np.ndarray,
    avg_latents_b: np.ndarray,
    steps: Optional[List[int]] = None,
) -> List[np.ndarray]:
    """Heatmaps at multiple timesteps to show when differences emerge.

    Args:
        avg_latents_a: Category A averaged trajectory
        avg_latents_b: Category B averaged trajectory
        steps: List of step indices (default: evenly spaced 8 steps)

    Returns:
        List of (H, W) heatmaps
    """
    n_steps = min(avg_latents_a.shape[0], avg_latents_b.shape[0])

    if steps is None:
        steps = np.linspace(0, n_steps - 1, 8, dtype=int).tolist()

    return [
        spatial_difference_heatmap(avg_latents_a, avg_latents_b, step=s)
        for s in steps
    ]


def trajectory_divergence_curve(
    avg_latents_a: np.ndarray,
    avg_latents_b: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Track L2 distance between two averaged trajectories at each step.

    DEPRECATED for statistical claims — use batch_divergence() instead.
    Kept for quick visualization of the mean trajectory.

    Returns:
        distances: (steps+1,) — L2 distance per step
        divergence_step: step where distance first exceeds 50% of final distance
    """
    n = min(avg_latents_a.shape[0], avg_latents_b.shape[0])
    a_flat = _flatten_step(avg_latents_a[:n])
    b_flat = _flatten_step(avg_latents_b[:n])

    distances = np.linalg.norm(a_flat - b_flat, axis=1)

    lock_in = _find_lock_in(distances)
    return distances, lock_in


def per_channel_difference(
    avg_latents_a: np.ndarray,
    avg_latents_b: np.ndarray,
    step: int = -1,
) -> np.ndarray:
    """Per-channel mean absolute difference at a specific step.

    DEPRECATED for statistical claims — use batch_channel_difference() instead.

    Returns:
        (C,) array — which latent channels carry the most category signal
    """
    a = avg_latents_a[step].mean(axis=0)  # (C, H, W)
    b = avg_latents_b[step].mean(axis=0)
    return np.mean(np.abs(a - b), axis=(1, 2))  # (C,)


# ---------------------------------------------------------------------------
# Colonial axis extraction (unchanged — external reference)
# ---------------------------------------------------------------------------

@dataclass
class ColonialAxes:
    """Principal component axes extracted from colonial archival images.

    Attributes:
        pca: Fitted PCA object
        components: (n_components, flattened_dim) — the axes
        centroid: (flattened_dim,) — mean of colonial latents
        explained_variance_ratio: fraction of variance per component
    """
    pca: PCA
    components: np.ndarray
    centroid: np.ndarray
    explained_variance_ratio: np.ndarray


def extract_colonial_axes(
    colonial_latents: np.ndarray,
    n_components: int = 8,
) -> ColonialAxes:
    """Fit PCA on colonial image latents to find axes of colonial visual grammar.

    Args:
        colonial_latents: (n_images, C, H, W) — VAE-encoded colonial photographs
        n_components: Number of principal components to retain

    Returns:
        ColonialAxes dataclass
    """
    n = colonial_latents.shape[0]
    flat = colonial_latents.reshape(n, -1)  # (n_images, C*H*W)

    pca = PCA(n_components=min(n_components, n))
    pca.fit(flat)

    return ColonialAxes(
        pca=pca,
        components=pca.components_,
        centroid=flat.mean(axis=0),
        explained_variance_ratio=pca.explained_variance_ratio_,
    )


def project_onto_axes(
    data: np.ndarray,
    axes: ColonialAxes,
) -> np.ndarray:
    """Project flattened data onto colonial PC axes.

    Args:
        data: (..., flattened_dim)
        axes: ColonialAxes from extract_colonial_axes

    Returns:
        Projection of shape (..., n_components)
    """
    original_shape = data.shape[:-1]
    flat_dim = data.shape[-1]
    flat = data.reshape(-1, flat_dim)

    centered = flat - axes.centroid
    projected = centered @ axes.components.T

    return projected.reshape(*original_shape, axes.components.shape[0])


# ---------------------------------------------------------------------------
# Temporal drift analysis (colonial axis projection)
# ---------------------------------------------------------------------------

def colonial_drift_over_time(
    avg_latents: np.ndarray,
    axes: ColonialAxes,
) -> np.ndarray:
    """Track projected coordinates on colonial axes over diffusion steps."""
    flat = _flatten_step(avg_latents)
    return project_onto_axes(flat, axes)


def colonial_flow_alignment(
    avg_flow: np.ndarray,
    axes: ColonialAxes,
) -> np.ndarray:
    """Project flow vectors onto colonial axes at each step."""
    steps = avg_flow.shape[0]
    per_step = avg_flow.mean(axis=1)
    flat = per_step.reshape(steps, -1)
    return flat @ axes.components.T


def drift_score(
    category_projections: np.ndarray,
    baseline_projections: np.ndarray,
) -> Dict[str, Any]:
    """Compute summary statistics comparing category vs baseline drift."""
    diff = category_projections - baseline_projections
    final_diff = diff[-1]
    total_drift = float(np.linalg.norm(final_diff))

    per_component = final_diff.tolist()

    cumulative = np.linalg.norm(diff, axis=1)
    lock_in = _find_lock_in(cumulative)

    return {
        "total_drift": total_drift,
        "per_component_drift": per_component,
        "lock_in_step": lock_in,
        "drift_curve": cumulative.tolist(),
    }


# =====================================================================
# inter_category_pca — legacy wrapper (uses averages)
# =====================================================================

def inter_category_pca(
    avg_latents: Dict[str, np.ndarray],
    n_components: int = 4,
) -> Tuple[Dict[str, np.ndarray], PCA, np.ndarray]:
    """PCA on averaged latents across categories.

    DEPRECATED for statistical claims — use batch_inter_category_pca() instead.
    Kept for quick visualization.
    """
    all_points = []
    for name, lat in avg_latents.items():
        n_steps = lat.shape[0]
        for s in range(0, n_steps, max(1, n_steps // 10)):
            all_points.append(lat[s].mean(axis=0).flatten())
    all_points = np.stack(all_points)

    pca = PCA(n_components=min(n_components, len(all_points)))
    pca.fit(all_points)

    projections = {}
    for name, lat in avg_latents.items():
        flat = _flatten_step(lat)
        projections[name] = pca.transform(flat)

    return projections, pca, pca.explained_variance_ratio_


# =====================================================================
#  ORCHESTRATOR
# =====================================================================

@dataclass
class ColonialGrammarAnalysis:
    """Run a full colonial visual grammar analysis with proper batch statistics.

    Usage:
        analysis = ColonialGrammarAnalysis(baseline_key="neutral")
        analysis.add_category("neutral", neutral_latent_list)
        analysis.add_category("arab", arab_latent_list)
        report = analysis.compute()
    """

    categories: Dict[str, List[np.ndarray]] = field(default_factory=dict)

    # Computed averages (for viz only)
    avg_latents: Dict[str, np.ndarray] = field(default_factory=dict)
    avg_flows: Dict[str, np.ndarray] = field(default_factory=dict)

    # Colonial reference (optional)
    colonial_axes: Optional[ColonialAxes] = None

    # Legacy results
    projections: Dict[str, np.ndarray] = field(default_factory=dict)
    flow_alignments: Dict[str, np.ndarray] = field(default_factory=dict)
    drift_scores: Dict[str, Dict] = field(default_factory=dict)

    baseline_key: str = "neutral"

    def add_category(self, name: str, latent_stacks: List[np.ndarray]):
        """Register a category's latent trajectories.

        Args:
            name: Category label (e.g. "neutral", "arab", "palestinian")
            latent_stacks: List of latent arrays, each (steps+1, batch, C, H, W)
        """
        self.categories[name] = latent_stacks

    def set_colonial_axes(
        self,
        colonial_latents: np.ndarray,
        n_components: int = 8,
    ):
        """Extract colonial PC axes from archival image encodings."""
        self.colonial_axes = extract_colonial_axes(colonial_latents, n_components)

    def compute(self, n_perms: int = 1000) -> Dict[str, Any]:
        """Run the full analysis pipeline.

        Returns a dict with:
            avg_latents, avg_flows — for visualization
            batch_divergence — {cat: BatchDivergenceResult} with mean/std/CI/p-value
            batch_flow_diff — {cat: BatchFlowResult}
            batch_channel_diff — {cat: BatchChannelResult}
            batch_spatial — {cat: BatchSpatialResult}
            batch_pca — {avg_proj, per_seed_proj, pca, var_explained}
            (optional) colonial_projections, drift_scores — if colonial axes set
        """
        baseline_stacks = self.categories.get(self.baseline_key)
        if baseline_stacks is None:
            raise ValueError(f"Baseline category '{self.baseline_key}' not found")

        # ── 1. Average trajectories (for visualization) ──
        for name, stacks in self.categories.items():
            self.avg_latents[name] = average_trajectories(stacks)
            self.avg_flows[name] = average_flow_field(stacks)
            print(f"  [{name}] {len(stacks)} seeds, "
                  f"avg latent {self.avg_latents[name].shape}")

        # ── 2. Batch statistics vs baseline ──
        b_divergence: Dict[str, BatchDivergenceResult] = {}
        b_flow: Dict[str, BatchFlowResult] = {}
        b_channel: Dict[str, BatchChannelResult] = {}
        b_spatial: Dict[str, BatchSpatialResult] = {}

        for name, stacks in self.categories.items():
            if name == self.baseline_key:
                continue
            print(f"  [{name}] computing batch statistics vs {self.baseline_key} ...")

            b_divergence[name] = batch_divergence(
                stacks, baseline_stacks, n_perms=n_perms
            )
            b_flow[name] = batch_flow_difference(stacks, baseline_stacks)
            b_channel[name] = batch_channel_difference(stacks, baseline_stacks)
            b_spatial[name] = batch_spatial_heatmap(stacks, baseline_stacks)

            d = b_divergence[name]
            print(f"         divergence: final={d.mean[-1]:.4f} ± {d.sem[-1]:.4f}, "
                  f"lock-in={d.lock_in_mean:.1f} ± {d.lock_in_std:.1f}, "
                  f"p={d.p_value:.4f}")

        # ── 3. Batch PCA across all seeds ──
        avg_proj, per_seed_proj, pca, var_exp = batch_inter_category_pca(
            self.categories
        )
        print(f"  PCA explained variance: {var_exp}")

        # ── 4. Colonial axes projection (if available) ──
        if self.colonial_axes is not None:
            for name, avg_lat in self.avg_latents.items():
                self.projections[name] = colonial_drift_over_time(
                    avg_lat, self.colonial_axes
                )
                self.flow_alignments[name] = colonial_flow_alignment(
                    self.avg_flows[name], self.colonial_axes
                )
            if self.baseline_key in self.projections:
                baseline_proj = self.projections[self.baseline_key]
                for name, proj in self.projections.items():
                    if name == self.baseline_key:
                        continue
                    self.drift_scores[name] = drift_score(proj, baseline_proj)

        return {
            # Visualization
            "avg_latents": self.avg_latents,
            "avg_flows": self.avg_flows,
            # Batch statistics (the real science)
            "batch_divergence": b_divergence,
            "batch_flow_diff": b_flow,
            "batch_channel_diff": b_channel,
            "batch_spatial": b_spatial,
            # PCA on all seeds
            "batch_pca_avg_proj": avg_proj,
            "batch_pca_per_seed_proj": per_seed_proj,
            "batch_pca": pca,
            "batch_pca_variance": var_exp,
            # Colonial (optional)
            "colonial_projections": self.projections,
            "colonial_flow_alignments": self.flow_alignments,
            "colonial_drift_scores": self.drift_scores,
            "colonial_axes": self.colonial_axes,
        }

    def save(self, output_dir: str):
        """Persist all computed arrays and metrics to disk."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        for name, lat in self.avg_latents.items():
            np.save(out / f"avg_latents_{name}.npy", lat)

        for name, fl in self.avg_flows.items():
            np.save(out / f"avg_flow_{name}.npy", fl)

        for name, proj in self.projections.items():
            np.save(out / f"projection_{name}.npy", proj)

        for name, align in self.flow_alignments.items():
            np.save(out / f"flow_alignment_{name}.npy", align)

        if self.colonial_axes is not None:
            np.save(out / "colonial_components.npy", self.colonial_axes.components)
            np.save(out / "colonial_centroid.npy", self.colonial_axes.centroid)
            np.save(out / "colonial_explained_var.npy",
                    self.colonial_axes.explained_variance_ratio)

        import json
        with open(out / "drift_scores.json", "w") as f:
            json.dump(self.drift_scores, f, indent=2)

        print(f"Saved analysis to {out}")
