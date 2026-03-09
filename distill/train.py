from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
    StableDiffusionPipeline,
)
from diffusers.loaders import LoraLoaderMixin
from PIL import Image
import numpy as np
from tqdm import tqdm

from distill.config import DistillConfig
from distill.dataset import DistillDataset


class DistillTrainer:
    """Trainer for distilling teacher images into a Stable Diffusion student.
    
    Supports both full fine-tuning and LoRA training.
    """

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        lora_rank: int = 16,
        learning_rate: float = 5e-5,
        gradient_accumulation_steps: int = 1,
        train_epochs: int = 1,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model_id = model_id
        self.lora_rank = lora_rank
        self.learning_rate = learning_rate
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.train_epochs = train_epochs
        self.device = device
        
        self.tokenizer: Optional[CLIPTokenizer] = None
        self.text_encoder: Optional[CLIPTextModel] = None
        self.vae: Optional[AutoencoderKL] = None
        self.unet: Optional[UNet2DConditionModel] = None
        self.noise_scheduler: Optional[DDPMScheduler] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    def load_models(self):
        """Load SD models for training."""
        print(f"Loading models from {self.model_id}...")
        
        self.tokenizer = CLIPTokenizer.from_pretrained(
            self.model_id, subfolder="tokenizer"
        )
        self.text_encoder = CLIPTextModel.from_pretrained(
            self.model_id, subfolder="text_encoder"
        ).to(self.device)
        
        self.vae = AutoencoderKL.from_pretrained(
            self.model_id, subfolder="vae"
        ).to(self.device)
        
        self.unet = UNet2DConditionModel.from_pretrained(
            self.model_id, subfolder="unet"
        ).to(self.device)
        
        self.noise_scheduler = DDPMScheduler.from_pretrained(
            self.model_id, subfolder="scheduler"
        )
        
        # Freeze VAE and text encoder
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        # Set UNet to train mode
        self.unet.train()
        
        print("Models loaded.")

    def encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode a text prompt to embeddings."""
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_input_ids = text_inputs.input_ids
        
        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.device)
        )[0]
        
        return prompt_embeds

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Single training step."""
        prompts = batch["prompt"]
        images = batch["final"]  # List of PIL Images
        
        # Encode prompts
        prompt_embeds_list = []
        for prompt in prompts:
            embeds = self.encode_prompt(prompt)
            prompt_embeds_list.append(embeds)
        
        prompt_embeds = torch.cat(prompt_embeds_list, dim=0).to(self.device)
        
        # Convert images to latents
        images_np = [np.array(img.resize((512, 512))) / 127.5 - 1.0 for img in images]
        images_tensor = torch.tensor(np.array(images_np), dtype=torch.float32).permute(0, 3, 1, 2)
        images_tensor = images_tensor.to(self.device)
        
        with torch.no_grad():
            latents = self.vae.encode(images_tensor).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        
        # Add noise
        batch_size = latents.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        ).long()
        
        noise = torch.randn_like(latents)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(
            noisy_latents, timesteps, encoder_hidden_states=prompt_embeds
        ).sample
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        # Backward
        loss.backward()
        
        return loss.item()

    def train(
        self,
        train_dataset: DistillDataset,
        output_dir: Path,
        batch_size: int = 4,
        save_every: int = 100,
    ):
        """Run training on a dataset."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.optimizer is None:
            self.optimizer = torch.optim.AdamW(
                self.unet.parameters(),
                lr=self.learning_rate,
            )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
        )
        
        global_step = 0
        for epoch in range(self.train_epochs):
            self.unet.train()
            epoch_loss = 0.0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.train_epochs}")
            for batch in pbar:
                # Move images to device
                images = [img.to(self.device) for img in batch["final"]]
                batch["final"] = images
                
                # Forward pass with gradient accumulation
                for i in range(0, len(images), self.gradient_accumulation_steps):
                    sub_batch = {
                        "prompt": batch["prompt"][i:i+self.gradient_accumulation_steps],
                        "final": images[i:i+self.gradient_accumulation_steps],
                    }
                    loss = self.train_step(sub_batch)
                    epoch_loss += loss
                
                # Optimizer step
                self.optimizer.step()
                self.optimizer.zero_grad()
                
                global_step += 1
                pbar.set_postfix({"loss": epoch_loss / (global_step * batch_size)})
                
                # Save checkpoint
                if global_step % save_every == 0:
                    self.save_checkpoint(output_dir / f"checkpoint-{global_step}")
            
            # Save epoch checkpoint
            self.save_checkpoint(output_dir / f"epoch-{epoch+1}")
        
        # Final save
        self.save_checkpoint(output_dir / "final")
        print(f"Training complete. Saved to {output_dir}")

    def save_checkpoint(self, path: Path):
        """Save model checkpoint."""
        os.makedirs(path, exist_ok=True)
        self.unet.save_pretrained(path / "unet")
        print(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: Path):
        """Load model checkpoint."""
        self.unet = UNet2DConditionModel.from_pretrained(path / "unet").to(self.device)
        print(f"Loaded checkpoint from {path}")


def main():
    parser = argparse.ArgumentParser(description="Train distilled SD model")
    parser.add_argument("--manifest", type=Path, required=True, help="Path to manifest.jsonl")
    parser.add_argument("--output-dir", type=Path, default=Path("distill/output"))
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()
    
    # Load dataset
    ds = DistillDataset(args.manifest, use_partials=False)
    print(f"Loaded {len(ds)} samples")
    
    # Create trainer
    trainer = DistillTrainer(
        model_id=args.model_id,
        learning_rate=args.lr,
        train_epochs=args.epochs,
        lora_rank=args.lora_rank,
    )
    
    # Train
    trainer.load_models()
    trainer.train(ds, args.output_dir, batch_size=args.batch_size, save_every=args.save_every)


if __name__ == "__main__":
    main()
