# Trajectory Data Structure

## Overview

When you run inference with `capture_latents=True`, the system captures **EVERY step** of the diffusion process. This gives you complete visibility into how the model transforms random noise into the final image.

## What's Captured

### 1. `InferenceResult` Object

The `client.generate()` method returns an `InferenceResult` containing:

```python
result = client.generate(
    prompt="a cat",
    num_steps=50,
    capture_latents=True,      # Capture latent at each step
    capture_noise=True,        # Capture noise predictions
    capture_timesteps=True     # Capture scheduler values
)
```

**Accessing data:**

```python
# Basic info
print(f"Steps: {len(result)}")                    # 50
print(f"Latent shape: {result.latent_shape}")     # (2, 4, 64, 64)

# Raw arrays (NumPy)
latents = result.latents           # Shape: (51, 2, 4, 64, 64) - all steps + initial
noise_preds = result.noise_preds   # Shape: (50, 2, 4, 64, 64)
timesteps = result.timesteps       # Shape: (50,) - values like [999, 979, ..., 1]
prompt_embeds = result.prompt_embeds  # Shape: (2, 77, 768)

# Final image
with open("output.png", "wb") as f:
    f.write(result.image)
```

### 2. Trajectory Steps

For easy access to step-by-step data, use the `trajectory` property:

```python
# Get all steps as TrajectoryStep objects
for step in result.trajectory:
    print(f"Step {step.step_index}: timestep={step.timestep}")

# Access specific step
step_0 = result.get_step(0)      # Initial noise
step_25 = result.get_step(25)    # Mid-generation  
step_final = result.get_step(-1) # Final latent before VAE decode

# Step attributes
print(step_0.timestep)           # 999 (scheduler value)
print(step_0.latent.shape)       # (2, 4, 64, 64)
print(step_0.noise_pred.shape)   # (2, 4, 64, 64)
```

### 3. TrajectoryStep Structure

Each step contains:

| Attribute | Type | Shape | Description |
|-----------|------|-------|-------------|
| `step_index` | int | - | Which step (0 to num_steps) |
| `timestep` | int | - | Scheduler value (typically 0-1000) |
| `latent` | np.ndarray | (batch, c, h, w) | Latent representation |
| `noise_pred` | np.ndarray | (batch, c, h, w) | Model's noise prediction |
| `prompt_embedding` | np.ndarray | (batch, seq, dim) | Text conditioning |

## Array Shapes Explained

### Latents: `(steps, batch, channels, height, width)`

- **steps**: num_steps + 1 (initial noise + each denoising step)
- **batch**: Usually 2 (unconditional + conditional for CFG)
- **channels**: 4 for SD 1.5, could be different for other models
- **height/width**: Spatial resolution (e.g., 64x64 for 512px images)

**Example with 50 steps:**
```python
latents.shape  # (51, 2, 4, 64, 64)
               # 51 = initial + 50 denoising steps
               # 2 = uncond + cond latents
               # 4 = latent channels
               # 64x64 = spatial size
```

### Timesteps: `(steps,)`

Integer values from the scheduler:
```python
result.timesteps  # array([999, 979, 959, ..., 19, 1])
                  # Higher = more noise (earlier in process)
                  # Lower = less noise (later in process)
```

## Usage Examples

### Example 1: Visualize Latent Evolution

```python
import matplotlib.pyplot as plt

means = [step.latent.mean() for step in result]
stds = [step.latent.std() for step in result]

plt.figure(figsize=(10, 4))
plt.plot(means, label='Mean')
plt.plot(stds, label='Std Dev')
plt.xlabel('Step')
plt.ylabel('Value')
plt.title('Latent Statistics Across Denoising')
plt.legend()
plt.show()
```

### Example 2: Compare Two Steps

```python
import numpy as np

# Get initial and final latents
initial = result.get_step(0).latent
final = result.get_step(-1).latent

# Calculate difference
diff = np.abs(final - initial)
print(f"Mean absolute change: {diff.mean():.4f}")
```

### Example 3: Extract Single Latent

```python
# If you just want the final latent (no batch dimension)
final_latent = result.latents[-1, 0]  # Shape: (4, 64, 64)

# Or use the convenience property
final_latent = result.get_step(-1).latent[0]  # Same thing
```

### Example 4: Batch Processing

```python
prompts = ["a cat", "a dog", "a bird"]
all_trajectories = []

for prompt in prompts:
    result = client.generate(prompt, num_steps=30, capture_latents=True)
    all_trajectories.append(result.latents)

# Now you have shape: (3, 31, 2, 4, 64, 64)
all_latents = np.stack(all_trajectories)
```

## Memory Considerations

Trajectory data can be large:

- SD 1.5 at 512px: ~50MB per 50-step generation
- SDXL at 1024px: ~200MB per generation

**Tips:**
1. Use `num_steps=20` for initial exploration
2. Set `capture_latents=False` if you only need the image
3. Process trajectories incrementally rather than storing all

## Iterator Support

The `InferenceResult` supports iteration:

```python
# Iterate over all steps
for step in result:
    print(f"Step {step.step_index}: {step.latent.mean():.4f}")

# Or with enumerate
for i, step in enumerate(result):
    if step.timestep < 500:
        print(f"Past halfway at step {i}")
```

## Notes

- **Every step is captured**: The hook intercepts every forward pass of the UNet
- **Step 0**: Initial noise (before any denoising)
- **Step N**: Final latent (ready for VAE decode)
- **Timestep values**: Decrease as denoising progresses (999 → 1)
- **Batch dimension**: Index 0 = unconditional, Index 1 = conditional (for CFG)
