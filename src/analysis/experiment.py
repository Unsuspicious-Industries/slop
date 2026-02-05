import os
import json
import time
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
from tqdm import tqdm

from ..slop_model import SlopModel

class ExperimentRunner:
    def __init__(
        self,
        diffusion_model_id: str,
        encoder_model_id: str,
        output_dir: str,
        device: str = "cpu"
    ):
        self.diffusion_model_id = diffusion_model_id
        self.encoder_model_id = encoder_model_id
        self.output_dir = output_dir
        self.device = device
        
        self.diffuser = None
        self.encoder = None
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "trajectories"), exist_ok=True)

    def load_models(self):
        print(f"Loading diffusion model: {self.diffusion_model_id}...")
        self.diffuser = SlopModel.load(self.diffusion_model_id, device=self.device)
        
        print(f"Loading encoder model: {self.encoder_model_id}...")
        self.encoder = SlopModel.load(self.encoder_model_id, device=self.device)

    def run_batch(
        self,
        prompts: List[str],
        num_steps: int = 20,
        batch_name: str = "experiment",
        prompt_groups: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Run a batch of prompts, capture trajectories, encode results, and save to disk.
        """
        if self.diffuser is None or self.encoder is None:
            self.load_models()

        results = []
        
        print(f"Running batch '{batch_name}' with {len(prompts)} prompts...")
        for i, prompt in enumerate(tqdm(prompts)):
            # 1. Diffusion
            diff_out = self.diffuser.diffuse(prompt, num_steps=num_steps)
            image: Image.Image = diff_out["image"]
            trajectory = diff_out["trajectory"]
            
            # 2. Encoding (Analysis)
            embedding = self.encoder.encode(image)
            
            # 3. Save artifacts
            timestamp = int(time.time())
            safe_prompt = "".join(x for x in prompt[:20] if x.isalnum() or x in " _-").strip().replace(" ", "_")
            file_id = f"{batch_name}_{i}_{safe_prompt}_{timestamp}"
            
            # Save Image
            image_path = os.path.join(self.output_dir, "images", f"{file_id}.png")
            image.save(image_path)
            
            # Save Trajectory (compressed/numpy)
            # We save latent and prompt_embedding separately or together
            traj_path = os.path.join(self.output_dir, "trajectories", f"{file_id}.npz")
            
            # Extract arrays from list of dicts for efficient storage
            timesteps = np.array([step["timestep"] for step in trajectory])
            # latents might be (1, 4, 64, 64) -> stack to (steps, 4, 64, 64)
            latents = np.stack([step["latent"] for step in trajectory])
            # prompt_embedding might be constant, but good to keep one ref
            prompt_embed = trajectory[0]["prompt_embedding"]
            
            np.savez_compressed(
                traj_path,
                timesteps=timesteps,
                latents=latents,
                prompt_embedding=prompt_embed,
                final_embedding=embedding
            )
            
            results.append({
                "id": file_id,
                "prompt": prompt,
                "image_path": image_path,
                "trajectory_path": traj_path,
                "embedding_shape": embedding.shape,
                "group": prompt_groups.get(prompt) if prompt_groups else None,
            })
            
        # Save manifest
        manifest_path = os.path.join(self.output_dir, f"{batch_name}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(results, f, indent=2)
            
        return results

    def aggregate_data(self, batch_name: str) -> str:
        """
        Aggregates embeddings from a batch into a single numpy file for easy visualization.
        """
        manifest_path = os.path.join(self.output_dir, f"{batch_name}_manifest.json")
        if not os.path.exists(manifest_path):
             raise FileNotFoundError(f"Manifest not found: {manifest_path}")
             
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        embeddings = []
        labels = []
        ids = []
        groups = []
        
        print("Aggregating data...")
        for item in manifest:
            traj_data = np.load(item["trajectory_path"])
            embeddings.append(traj_data["final_embedding"])
            labels.append(item["prompt"])
            ids.append(item["id"])
            groups.append(item.get("group"))
            
        # Stack
        embeddings_arr = np.vstack(embeddings)
        
        out_path = os.path.join(self.output_dir, f"{batch_name}_embeddings.npz")
        np.savez_compressed(
            out_path,
            embeddings=embeddings_arr,
            labels=np.array(labels),
            ids=np.array(ids),
            groups=np.array(groups),
        )
        print(f"Aggregated data saved to {out_path}")
        return out_path
