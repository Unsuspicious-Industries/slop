"""Smoke test for divergence rate and motif extraction functions."""
import numpy as np
from client.analysis.colonial import (
    batch_divergence_rate, compare_divergence_rates, DivergenceRateResult,
    extract_motif_direction, measure_motif_activation, extract_motif_components,
    MotifDirection, MotifExtractionResult,
    batch_divergence, BatchDivergenceResult,
)

print("All imports OK")

rng = np.random.default_rng(0)
shape = (11, 1, 4, 8, 8)
stacks_a = [rng.standard_normal(shape).astype(np.float32) + 0.5 for _ in range(5)]
stacks_b = [rng.standard_normal(shape).astype(np.float32) for _ in range(5)]

# Test divergence rate
rr = batch_divergence_rate(stacks_a, stacks_b, smooth_window=3)
print(f"Rate shape: {rr.mean.shape}, peak: {rr.peak_step_mean:.1f}, accel_half: {rr.accel_half_mean:.1f}")
assert rr.mean.shape == (10,), f"Expected (10,), got {rr.mean.shape}"
assert rr.per_seed.shape[0] == 5

# Test motif extraction
mr = extract_motif_direction(stacks_a, stacks_b, step=-1, name="test")
print(f"Motif magnitude: {mr.motif.magnitude:.4f}, onset: {mr.motif.onset_step}")
print(f"Spatial mask shape: {mr.motif.spatial_mask.shape}")
print(f"Activation curve shape: {mr.activation_mean.shape}")
print(f"Top channels: {mr.top_channels}")
assert mr.motif.spatial_mask.shape == (8, 8)
assert mr.activation_mean.shape == (11,)
assert len(mr.top_channels) == 4

# Test measure_motif_activation
ps, mean = measure_motif_activation(stacks_b, mr.motif)
print(f"Cross-activation shape: {ps.shape}")
assert ps.shape == (5, 11)

# Test motif components
comps, pca, var = extract_motif_components(stacks_a, stacks_b, n_components=3)
print(f"Components shape: {comps.shape}, variance: {var}")
assert comps.shape[0] == 3
assert comps.shape[1:] == (4, 8, 8)

# Test compare_divergence_rates
rr2 = batch_divergence_rate(stacks_b, stacks_a, smooth_window=3)
comp = compare_divergence_rates({"cat_a": rr, "cat_b": rr2})
print(f"Peak order: {comp['peak_order']}")
print(f"Faster early: {comp['pairwise']['cat_a_vs_cat_b']['faster_early']}")
assert len(comp["peak_order"]) == 2
assert "pairwise" in comp

print("\nAll tests passed.")
