"""Analysis modules for SLOP trajectories and visual grammar detection."""

from client.analysis.colonial import (
    ColonialGrammarAnalysis,
    # Batch-aware statistical functions (use these for any claims)
    BatchDivergenceResult,
    BatchFlowResult,
    BatchChannelResult,
    BatchSpatialResult,
    batch_divergence,
    batch_flow_difference,
    batch_channel_difference,
    batch_spatial_heatmap,
    batch_inter_category_pca,
    # Divergence rate analysis
    DivergenceRateResult,
    batch_divergence_rate,
    compare_divergence_rates,
    # Spatial motif extraction
    MotifDirection,
    MotifExtractionResult,
    extract_motif_direction,
    measure_motif_activation,
    extract_motif_components,
    # Low-level helpers
    average_trajectories,
    compute_flow_field,
    average_flow_field,
    flow_difference,
    flow_magnitude,
    # Legacy single-trajectory (for quick viz only)
    spatial_difference_heatmap,
    temporal_heatmap_sequence,
    trajectory_divergence_curve,
    per_channel_difference,
    # Colonial axes
    project_onto_axes,
    extract_colonial_axes,
    drift_score,
)
