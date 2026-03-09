# SLOP Client API Reference

## SlopClient

Core interface for remote GPU inference.

```python
from client.config import registry
from client.interface import SlopClient

client = SlopClient(registry.list()[-1])
client.connect()
```

### `info()`
Returns `ServerInfo` (GPU name, VRAM, CUDA version).

### `embed(prompt, negative_prompt="")`
Encodes text into latent embeddings. Returns `(prompt_embeds, negative_prompt_embeds)` as `np.ndarray` of shape `(1, 77, 768)`.

### `sample(prompt, num_steps=50, batch_size=1, ...)`
Samples latent trajectories. Returns `Result` containing `points` (latents) of shape `(steps, B, 4, 64, 64)`. Does **not** render images.

### `sample_from_embeds(prompt_embeds, negative_embeds, ...)`
Same as `sample` but takes pre-computed embeddings. Enables arithmetic composition.

### `sample_delta(target_embeds, base_embeds, negative_embeds, ...)`
Runs diffusion where the force at each step is the difference: `score(target) - score(base)`. This isolates the semantic trajectory of the intervention.

### `probe_at(points, prompt_embeds, negative_embeds, timestep=500)`
Evaluates the score function (noise prediction) at specific latent points.
- `points`: `(B, 4, 64, 64)`
- Returns: `(B, 4, 64, 64)` force vectors.

### `render(latents)`
Decodes latents via VAE into PIL Images.
- `latents`: `(B, 4, 64, 64)`
- Returns: `list[PIL.Image]`

---

## Identity Utilities (`client.utils.identity`)

### `extract_identity_vector(client, identity, activities)`
Isolates an identity direction by contrasting prompts across multiple activities.
- `identity`: e.g., `"arab"`, `"doctor"`
- `activities`: e.g., `["walking", "working"]`
- Returns: `np.ndarray (1, 77, 768)`

### `apply_identity(embeds, identity_vector, scale=1.0)`
Arithmetic helper: `embeds + scale * identity_vector`.
