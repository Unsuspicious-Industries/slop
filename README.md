# SLOP: Vector Field Analysis for Diffusion Bias

**SLOP** (Server-Local Operations for Physics) is a tool for analyzing latent biases in diffusion models using physics-based vector fields. It is designed to run in restricted HPC environments (Singularity containers, no root, no open ports) by tunneling a custom binary protocol over SSH.

## Architecture

*   **Server (`server/`)**: Runs inside a container on a GPU node. Listens on `stdin`, writes to `stdout`.
    *   Zero-dependency protocol (JSON over pipe).
    *   Stateful inference runner with support for Stable Diffusion and Flux.
    *   Hooks into model execution to capture latents, noise predictions, and embeddings.
*   **Client (`client/`)**: Runs on your local machine.
    *   Manages SSH tunnels transparently.
    *   Provides a CLI for deployment and health checks.
    *   Performs physics analysis (vector field interpolation, critical point detection) and visualization.
*   **Shared (`shared/`)**:
    *   Protocol definitions and serialization.
    *   Core physics math (numpy-only).

## Getting Started

### 1. Prerequisites
*   **Local:** Python 3.9+
*   **Remote:** Python 3.9+, SSH access, GPU

### 2. Deploy to Remote
Push the code to your HPC node. This copies the necessary `server` and `shared` code.

```bash
# Syntax: python -m client.deploy user@host --path /remote/path --name alias
python -m client.deploy user@login.cluster.edu --path /scratch/user/slop --name cluster-a
```

### 3. Check Connection
Verify the server is reachable and the GPU is detected.

```bash
python -m client.manage check cluster-a
```

To run a quick generation test (verifies CUDA and model loading):

```bash
python -m client.manage check cluster-a --verify
```

### 4. Run Analysis (Coming Soon)
(This section will describe how to use `client.interface` to run full experiments).

## Development

### Directory Structure
```
├── client/          # Local tools (CLI, visualization, transport)
├── server/          # Remote code (inference engine, daemon)
├── shared/          # Common protocol and physics logic
├── containers/      # Singularity/Apptainer definition files
└── data/            # Local data storage
```

### Protocol
The communication uses a length-prefixed JSON protocol over standard I/O.
*   **Request:** `[4-byte Len] { "kind": "inference", ... }`
*   **Response:** `[4-byte Len] { "kind": "result", "payload": { ... } }`
*   **Binary Data:** Numpy arrays and images are zlib-compressed and Base64-encoded within the JSON payload.
