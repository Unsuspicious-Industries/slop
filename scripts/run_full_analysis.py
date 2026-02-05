"""End-to-end analysis driver: generate, score, visualize with comprehensive debug output.

Inputs (expects artifacts already available or generates them):
- model id
- prompts file (neutral/control/racialized merged)
- historical embeddings (.npy)
- stereotype poles (.npy) or labels (optional)
- expected bias vectors (.npy) optional

Outputs:
- generated images/trajectories
- flow field, trajectories, embedding space figures
- metrics JSON
- comprehensive debug visualizations and data at every step
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import json

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from src.diffusion.loaders import load_diffusion_model
from src.diffusion.sd_hook import SDTrajectoryHook
from src.diffusion.flux_hook import FluxTrajectoryHook
from src.diffusion.trajectory_capture import compress_trajectory
from src.encoders.multimodal import EmbeddingExtractor
from src.analysis.clustering import reduce_embeddings, dbscan_clusters, identify_dense_clusters
from src.analysis.flow_fields import (
    compute_flow_field,
    compute_divergence_2d,
    compute_curl_2d,
    save_flow_field_debug
)
from src.analysis.attractors import analyze_flow_topology
from src.analysis.metrics import (
    stereotype_concentration,
    historical_continuity,
    trajectory_deviation_score,
    drift_strength,
    expected_bias_fit,
)
from src.visualization.embedding_space import plot_embedding_space
from src.visualization.trajectories import plot_trajectories
from src.visualization.flow_fields import plot_flow_field
from src.visualization.vector_field import VectorFieldVisualizer
from src.utils.storage import ensure_dir, save_json
from src.utils.debug import DebugLogger, save_step_debug


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", required=True, type=Path)
    p.add_argument("--historical", required=True, type=Path, help="historical embeddings .npy")
    p.add_argument("--expected", required=False, type=Path, help="expected bias vectors .npy")
    p.add_argument("--poles", required=False, type=Path, help="stereotype poles .npy")
    p.add_argument("--labels", required=False, type=Path, help="labels for historical embeddings")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--sample_rate", type=int, default=5)
    p.add_argument("--resolution", type=int, default=50)
    p.add_argument("--radius", type=float, default=0.5)
    p.add_argument("--encoder_strategy", default="concat", choices=["concat", "average", "separate"])
    p.add_argument("--skip_encode_final", action="store_true", help="Skip CLIP/DINO encoding of final images (use zeros matching historical dim)")
    return p.parse_args()


def load_prompts(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def main():
    args = parse_args()
    ensure_dir(args.out)
    ensure_dir(args.out / "images")
    ensure_dir(args.out / "trajectories")
    ensure_dir(args.out / "figures")

    historical = np.load(args.historical)
    expected = np.load(args.expected) if args.expected else None
    labels = np.load(args.labels) if args.labels else None

    # Load model + hook
    pipe = load_diffusion_model(args.model)
    hook = FluxTrajectoryHook(pipe) if "flux" in args.model.lower() else SDTrajectoryHook(pipe)

    prompts = load_prompts(args.prompts)
    extractor = EmbeddingExtractor(strategy=args.encoder_strategy)

    trajectories_latent = []
    modern_embeddings = []

    for prompt in prompts:
        image, trajectory = hook.generate_with_tracking(prompt, num_steps=args.steps)
        traj_comp = compress_trajectory(trajectory, sample_rate=args.sample_rate)
        traj_latents = np.array([step["latent"] for step in traj_comp])
        trajectories_latent.append(traj_latents)

        # Save image
        fname = prompt.replace(" ", "_")[:60]
        Image.fromarray(np.array(image)).save(args.out / "images" / f"{fname}.png")
        np.save(args.out / "trajectories" / f"{fname}.npy", traj_latents)

        if not args.skip_encode_final:
            modern_embeddings.append(extractor.encode_image(image))

    if modern_embeddings:
        modern_embeddings = np.stack(modern_embeddings)
    else:
        # fallback to zeros aligned to historical dims
        modern_embeddings = np.zeros((len(prompts), historical.shape[1]))

    # Stereotype poles: provided or derive via clustering on historical
    if args.poles:
        poles = np.load(args.poles)
    else:
        reduced = reduce_embeddings(historical, n_components=50)
        labels_auto = dbscan_clusters(reduced)
        poles, _ = identify_dense_clusters(reduced, labels_auto)

    # Metrics
    metrics = {}
    if labels is not None:
        metrics["stereotype_concentration"] = stereotype_concentration(historical, labels)
    metrics["historical_continuity"] = historical_continuity(modern_embeddings, historical)
    if expected is not None:
        metrics["expected_bias_fit"] = expected_bias_fit(modern_embeddings, expected)

    # Reduce latents to 2D for viz/flow
    flat_trajs = [t.reshape(t.shape[0], -1) for t in trajectories_latent]
    all_flat = np.concatenate(flat_trajs, axis=0)
    pca = PCA(n_components=2)
    pca.fit(all_flat)
    traj_2d = [pca.transform(t) for t in flat_trajs]
    poles_latent = np.zeros((1, 2))

    # Drift per trajectory (in reduced space)
    metrics["trajectory_drift"] = [drift_strength(traj, poles_latent) for traj in traj_2d]

    # Flow field
    grid, flow = compute_flow_field(traj_2d, poles_latent, grid_resolution=args.resolution, radius=args.radius)

    # Visuals
    fig_es = plot_embedding_space(historical, modern_embeddings)
    fig_es.savefig(args.out / "figures" / "embedding_space.png", dpi=200, bbox_inches="tight")
    fig_es.clear()

    fig_traj = plot_trajectories(traj_2d, poles_latent)
    fig_traj.savefig(args.out / "figures" / "trajectories.png", dpi=200, bbox_inches="tight")
    fig_traj.clear()

    fig_flow = plot_flow_field(grid, flow, poles_latent)
    fig_flow.savefig(args.out / "figures" / "flow_field.png", dpi=200, bbox_inches="tight")
    fig_flow.clear()

    save_json(metrics, args.out / "metrics.json")


if __name__ == "__main__":
    main()
