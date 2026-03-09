import base64
import io
import math
from typing import Any

import numpy as np
import torch
from PIL import Image

from shared.protocol.serialization import unpack_array


def model_device(pipe: Any) -> torch.device:
    """Return the execution device reported by the pipeline."""
    if hasattr(pipe, "_execution_device"):
        return pipe._execution_device
    if hasattr(pipe, "device"):
        return pipe.device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def module_dtype(module: Any) -> torch.dtype:
    """Return the dtype of the first parameter in a module."""
    return next(module.parameters()).dtype


def encode_text(pipe: Any, text: str | list[str], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode text with the pipeline tokenizer and text encoder."""
    text_inputs = pipe.tokenizer(
        text,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        return pipe.text_encoder(text_inputs.input_ids.to(device))[0].to(dtype=dtype)


def empty_prompt(pipe: Any, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Encode an empty prompt."""
    return encode_text(pipe, "", device, dtype)


def prepare_conditioning(pipe: Any, req: Any, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return positive and negative conditioning tensors."""
    use_override = req.prompt_embeds_override is not None

    if use_override:
        prompt_embeds = torch.from_numpy(
            unpack_array(req.prompt_embeds_override).astype(np.float32)
        ).to(device=device, dtype=dtype)

        if req.negative_prompt_embeds_override is not None:
            negative_prompt_embeds = torch.from_numpy(
                unpack_array(req.negative_prompt_embeds_override).astype(np.float32)
            ).to(device=device, dtype=dtype)
        else:
            negative_prompt_embeds = empty_prompt(pipe, device, dtype)

        return prompt_embeds, negative_prompt_embeds, True

    prompt_embeds = encode_text(pipe, req.prompt, device, dtype)
    negative_prompt_embeds = encode_text(pipe, req.negative_prompt or "", device, dtype)
    return prompt_embeds, negative_prompt_embeds, False


def repeat(tensor: torch.Tensor, n: int) -> torch.Tensor:
    """Repeat a single conditioning tensor across a batch."""
    if tensor.shape[0] == n:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError(f"cannot match batch size {n} from embeddings with shape {tuple(tensor.shape)}")
    return tensor.repeat(n, 1, 1)


def score(
    pipe: Any,
    latent: torch.Tensor,
    timestep: torch.Tensor | int,
    prompt_embeds: torch.Tensor,
    negative_prompt_embeds: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    """Run one conditioned score evaluation."""
    batch_size = latent.shape[0]
    conditioned = repeat(prompt_embeds, batch_size)
    negative = repeat(negative_prompt_embeds, batch_size)
    do_cfg = guidance_scale > 1.0

    if do_cfg:
        latent_input = torch.cat([latent, latent], dim=0)
        embeds = torch.cat([negative, conditioned], dim=0)
    else:
        latent_input = latent
        embeds = conditioned

    latent_input = pipe.scheduler.scale_model_input(latent_input, timestep)
    model_out = pipe.unet(latent_input, timestep, encoder_hidden_states=embeds).sample

    if not do_cfg:
        return model_out

    unconditioned, conditioned = model_out.chunk(2)
    return unconditioned + guidance_scale * (conditioned - unconditioned)


def png_b64(image: Image.Image) -> str:
    """Encode a PIL image as base64 PNG."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pil_images(decoded: torch.Tensor) -> list[Image.Image]:
    """Convert decoded image tensors to PIL images."""
    clipped = (decoded / 2 + 0.5).clamp(0, 1)
    arrays = clipped.cpu().permute(0, 2, 3, 1).float().numpy()
    return [Image.fromarray((array * 255).astype(np.uint8)) for array in arrays]


def render(pipe: Any, latents: np.ndarray, device: torch.device) -> list[Image.Image]:
    """Decode latent tensors with the pipeline VAE."""
    vae_dtype = module_dtype(pipe.vae)
    latent_tensor = torch.from_numpy(np.asarray(latents, dtype=np.float32).copy()).to(device=device, dtype=vae_dtype)
    scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)

    with torch.no_grad():
        decoded = pipe.vae.decode(latent_tensor / scaling).sample

    return pil_images(decoded)


def is_square(n: int) -> bool:
    """Return True when `n` is a perfect square."""
    root = math.isqrt(n)
    return root * root == n


def latent_batch(packed: Any) -> np.ndarray:
    """Unpack latents and normalize them to shape `(N, 4, H, W)`.

    The wire payload may contain one latent, a batch, or flattened SD latents.
    """
    if packed is None:
        raise ValueError("latent payload is required")

    arr = np.asarray(unpack_array(packed), dtype=np.float32)
    arr = np.squeeze(arr)

    if arr.ndim == 4:
        if arr.shape[1] != 4:
            raise ValueError(f"expected latent batch with channel dimension 4, got {arr.shape}")
        return arr

    if arr.ndim == 3:
        if arr.shape[0] != 4:
            raise ValueError(f"expected single latent with shape (4, H, W), got {arr.shape}")
        return arr[None]

    if arr.ndim == 2:
        width = int(arr.shape[1])
        if width % 4 == 0 and is_square(width // 4):
            side = math.isqrt(width // 4)
            return arr.reshape(arr.shape[0], 4, side, side)
        if width % 8 == 0 and is_square(width // 8):
            side = math.isqrt(width // 8)
            return arr.reshape(arr.shape[0], 2, 4, side, side)[:, 0]
        raise ValueError(
            f"could not infer latent shape from flattened batch {arr.shape}; expected (N, 4*H*W) or CFG-packed (N, 2*4*H*W)"
        )

    if arr.ndim == 1:
        width = int(arr.shape[0])
        if width % 4 == 0 and is_square(width // 4):
            side = math.isqrt(width // 4)
            return arr.reshape(1, 4, side, side)
        if width % 8 == 0 and is_square(width // 8):
            side = math.isqrt(width // 8)
            return arr.reshape(1, 2, 4, side, side)[:, 0]
        raise ValueError(
            f"could not infer latent shape from flat vector of length {width}; expected 4*H*W or CFG-packed 2*4*H*W"
        )

    raise ValueError(f"expected latent array shaped like (N,C,H,W), (C,H,W), (N,D), or (D); got {arr.shape}")
