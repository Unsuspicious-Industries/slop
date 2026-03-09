from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path


_STOP = False


def _set_stop(_signum, _frame):
    global _STOP
    _STOP = True


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")


def run(job_dir: Path) -> int:
    spec_path = job_dir / "spec.json"
    status_path = job_dir / "status.json"
    progress_path = job_dir / "progress.jsonl"

    if not spec_path.exists():
        _write_json(status_path, {"state": "failed", "error": f"missing spec: {spec_path}"})
        return 2

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    job_id = spec.get("job_id", job_dir.name)

    # Mark running
    _write_json(
        status_path,
        {
            "job_id": job_id,
            "state": "running",
            "started_at": time.time(),
            "updated_at": time.time(),
            "spec": {k: v for k, v in spec.items() if k not in {"api_key"}},
        },
    )

    # Lazy imports to keep process startup light
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from transformers import CLIPTokenizer, CLIPTextModel
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel

    from distill.dataset import DistillDataset

    manifest_path = Path(spec["manifest_path"])
    output_dir = Path(spec["output_dir"])
    model_id = spec["model_id"]
    batch_size = int(spec.get("batch_size", 2))
    epochs = int(spec.get("epochs", 1))
    learning_rate = float(spec.get("learning_rate", 5e-5))
    save_every = int(spec.get("save_every", 50))

    ds = DistillDataset(manifest_path, use_partials=False)

    def _collate(items):
        return {
            "prompt": [it["prompt"] for it in items],
            "final": [it["final"] for it in items],
        }

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.train()

    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)

    global_step = 0
    last_ckpt = ""
    start = time.time()

    def progress(step: int, epoch: int, loss: float, msg: str, ckpt: str = ""):
        nonlocal last_ckpt
        if ckpt:
            last_ckpt = ckpt
        rec = {
            "job_id": job_id,
            "step": step,
            "epoch": epoch,
            "loss": float(loss),
            "message": msg,
            "checkpoint_path": ckpt,
            "timestamp": time.time(),
        }
        _append_jsonl(progress_path, rec)
        _write_json(
            status_path,
            {
                "job_id": job_id,
                "state": "running" if not _STOP else "stopping",
                "updated_at": time.time(),
                "elapsed_s": time.time() - start,
                "step": step,
                "epoch": epoch,
                "loss": float(loss),
                "last_message": msg,
                "last_checkpoint": last_ckpt,
            },
        )

    progress(0, 0, 0.0, f"Loaded {len(ds)} samples")

    try:
        for ep in range(epochs):
            ep_loss = 0.0
            for batch in loader:
                if _STOP:
                    raise KeyboardInterrupt("stopped")
                prompts = batch["prompt"]
                images = batch["final"]

                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    prompt_embeds = text_encoder(text_inputs.input_ids.to(device))[0]

                imgs = []
                for img in images:
                    arr = np.array(img.resize((512, 512))) / 127.5 - 1.0
                    imgs.append(torch.tensor(arr))
                images_tensor = torch.stack(imgs).permute(0, 3, 1, 2).float().to(device)

                with torch.no_grad():
                    latents = vae.encode(images_tensor).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                bs = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bs,),
                    device=device,
                ).long()
                noise = torch.randn_like(latents)
                noisy = noise_scheduler.add_noise(latents, noise, timesteps)

                noise_pred = unet(noisy, timesteps, encoder_hidden_states=prompt_embeds).sample
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                global_step += 1
                ep_loss += float(loss.item())

                if global_step % 10 == 0:
                    progress(global_step, ep + 1, float(loss.item()), f"step {global_step} loss={loss.item():.4f}")

                if save_every > 0 and global_step % save_every == 0:
                    ckpt_dir = output_dir / f"checkpoint-{global_step}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    unet.save_pretrained(ckpt_dir / "unet")
                    progress(global_step, ep + 1, float(loss.item()), f"saved checkpoint {global_step}", str(ckpt_dir))

            avg = ep_loss / max(1, len(loader))
            progress(global_step, ep + 1, avg, f"epoch {ep + 1} avg_loss={avg:.4f}")

        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        unet.save_pretrained(final_dir / "unet")

        _write_json(
            status_path,
            {
                "job_id": job_id,
                "state": "completed",
                "updated_at": time.time(),
                "elapsed_s": time.time() - start,
                "step": global_step,
                "epoch": epochs,
                "loss": float(loss.item()) if "loss" in locals() else 0.0,
                "output_dir": str(output_dir),
            },
        )
        return 0
    except KeyboardInterrupt:
        _write_json(
            status_path,
            {
                "job_id": job_id,
                "state": "killed",
                "updated_at": time.time(),
                "elapsed_s": time.time() - start,
                "step": global_step,
                "epoch": 0,
            },
        )
        return 130
    except Exception as e:
        _write_json(
            status_path,
            {
                "job_id": job_id,
                "state": "failed",
                "updated_at": time.time(),
                "elapsed_s": time.time() - start,
                "step": global_step,
                "error": str(e),
            },
        )
        return 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-dir", required=True, help="job directory path")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _set_stop)
    signal.signal(signal.SIGINT, _set_stop)

    sys.exit(run(Path(args.job_dir)))


if __name__ == "__main__":
    main()
