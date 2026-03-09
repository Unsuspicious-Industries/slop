# SLOP: Vector Field Analysis for Diffusion Bias

**SLOP** (Server-Local Operations for Physics) is a tool for analyzing latent biases in diffusion models by querying the model directly in latent space. It is designed to run in restricted HPC environments (Singularity containers, no root, no open ports) by tunneling a custom binary protocol over SSH.

## Architecture

*   **Server (`server/`)**: Runs inside a container on a GPU node. Listens on `stdin`, writes to `stdout`.
    *   Zero-dependency protocol (JSON over pipe).
    *   Stateful inference runner with support for Stable Diffusion and Flux.
    *   Hooks into model execution to capture latents, noise predictions, and embeddings.
*   **Client (`client/`)**: Runs on your local machine.
    *   Manages SSH tunnels transparently.
    *   Provides a CLI for deployment and health checks.
    *   Probes diffusion models at arbitrary latent positions, runs custom delta-driven rollouts, and saves latent-space artefacts for later analysis.
*   **Shared (`shared/`))**:
    *   Protocol definitions and serialization.
    *   Core physics math (numpy-only).

## Getting Started

### 1. Prerequisites
*   **Local:** Python 3.9+
*   **Remote:** Python 3.9+, SSH access, GPU

### 2. Quick Start with Vast.ai

```bash
# Sync running vast.ai instances to registry
python -m client.vastai --sync

# Check connection
python -m client.manage check vast-32582479

# Run a test generation
python experiments/sample.py
```

### 3. Deploy to Remote (SSH)
Push the code to your HPC node. This copies the necessary `server` and `shared` code.

```bash
# Syntax: python -m client.deploy user@host --path /remote/path --name alias
python -m client.deploy user@login.cluster.edu --path /scratch/user/slop --name cluster-a
```

### 4. Check Connection
Verify the server is reachable and the GPU is detected.

```bash
python -m client.manage check cluster-a
```

To run a quick generation test (verifies CUDA and model loading):

```bash
python -m client.manage check cluster-a --verify
```

### 5. Image Generation Workflow

The client provides two methods:
- `client.sample()` - Generates latents (no image rendered, memory efficient)
- `client.render(latents)` - Decodes latents to PIL Images

```python
from client.config import registry
from client.interface import SlopClient

# Get provider
providers = registry.list()
client = SlopClient(providers[-1])
client.connect()

# Generate latents (no image)
result = client.sample(prompt="a cat", num_steps=20, seed=42)

# Render latents to images
images = client.render(result.points[-1])
images[0].save("cat.png")

client.close()
```

### 6. Run Delta-Field Experiments

The current workflow does not reconstruct a global field from samples. Instead,
it uses the model itself as the field oracle:

- pick a base prompt embedding `x`
- build a biased embedding `b = x + v_identity`
- evaluate `delta(z, t) = eps_b(z, t) - eps_x(z, t)` directly from the model
- save all sampled latent positions and all returned tensors for later analysis

Useful entry points:

```bash
python experiments/delta_map.py --base-prompt "in a city"
python experiments/delta_rollout.py --base-prompt "in a city"
```

`delta_map.py` samples many latent positions and saves:

- `latents.npy`
- `base_noise_preds.npy`
- `biased_noise_preds.npy`
- `delta_noise_preds.npy`
- `force.npy`

`delta_rollout.py` runs an experimental denoising loop using the delta itself and saves:

- every visited latent position
- per-step base / biased / delta noise predictions
- the final decoded image

The rollout is intentionally experimental, especially in `delta_only` mode.
Treat it as a probe into what the delta field does, not as a standard sampling method.

## Development

### Directory Structure
```
├── client/          # Local tools (CLI, visualization, transport)
├── server/          # Remote code (inference engine, daemon)
├── shared/          # Common protocol and physics logic
├── containers/      # Singularity/Apptainer definition files
├── notebooks/       # Jupyter notebooks for exploration
└── experiments/     # Experiment scripts
```

### Protocol
The communication uses a length-prefixed JSON protocol over standard I/O.
*   **Request:** `[4-byte Len] { "kind": "inference", ... }`
*   **Response:** `[4-byte Len] { "kind": "result", "payload": { ... } }`
*   **Binary Data:** Numpy arrays and images are zlib-compressed and Base64-encoded within the JSON payload.
