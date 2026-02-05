"""Generate images with trajectory capture from diffusion models."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from src.diffusion.loaders import load_diffusion_model
from src.diffusion.sd_hook import SDTrajectoryHook
from src.diffusion.flux_hook import FluxTrajectoryHook
from src.diffusion.trajectory_capture import compress_trajectory
from src.utils.storage import ensure_dir, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--poles", required=False, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sample_rate", type=int, default=5)
    return parser.parse_args()


def load_prompts(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    args = parse_args()
    ensure_dir(args.output / "images")
    ensure_dir(args.output / "trajectories")

    pipe = load_diffusion_model(args.model)
    hook = FluxTrajectoryHook(pipe) if "flux" in args.model.lower() else SDTrajectoryHook(pipe)
    prompts = load_prompts(args.prompts)
    poles = np.load(args.poles) if args.poles else None

    for prompt in tqdm(prompts, desc="Generating"):
        image, trajectory = hook.generate_with_tracking(prompt, num_steps=args.steps)
        traj_comp = compress_trajectory(trajectory, sample_rate=args.sample_rate)

        # Save image
        filename = prompt.replace(" ", "_")[:50]
        img_path = args.output / "images" / f"{filename}.png"
        Image.fromarray(np.array(image)).save(img_path)

        # Save trajectory as npy (latents only to keep light)
        traj_latents = np.array([step["latent"] for step in traj_comp])
        traj_path = args.output / "trajectories" / f"{filename}.npy"
        np.save(traj_path, traj_latents)

        meta = {
            "prompt": prompt,
            "timesteps": [int(step["timestep"]) for step in traj_comp],
        }
        save_json(meta, args.output / "trajectories" / f"{filename}.json")


if __name__ == "__main__":
    main()
