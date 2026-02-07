# Deployment Guide

This guide explains how to deploy the **Racist Currents (SLOP)** inference server to a remote HPC cluster or GPU workstation.

## Architecture

*   **Local Machine (Client):** Runs the Python interface, sends prompts, and analyzes results.
*   **Remote Machine (Server):** Runs a Singularity container with PyTorch/Diffusers, executes generation, and returns data via SSH.

---

## 1. Remote Preparation (HPC/GPU Node)

The remote machine needs **Singularity (or Apptainer)** installed.

### A. Build the Container
You usually need to build the container image (`.sif`) on a Linux machine where you have `sudo` access (or use `--fakeroot`).

1.  **Transfer the definition file:**
    ```bash
    scp containers/slop.def user@remote-host:~/
    ```

2.  **Build the image:**
    ```bash
    # On the remote machine (or a build node)
    apptainer build slop.sif slop.def
    # OR
    sudo singularity build slop.sif slop.def
    ```

3.  **Note the path:**
    Ideally, place the final `slop.sif` in `~/slop/containers/slop.sif` on the remote host, or note its location.

---

## 2. Local Setup (Client)

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure SSH:**
    Ensure you can SSH into the remote host without a password (use an SSH key):
    ```bash
    ssh-copy-id user@remote-host
    ```

---

## 3. Deployment

We use the `client.deploy` tool to push the codebase and register the server configuration locally.

**Scenario A: Standard Deployment (Recommended)**
This copies the code to `~/slop` on the remote and assumes the container is at `~/slop/containers/slop.sif`.

```bash
# 1. Upload your locally built container (if you built it locally)
# rsync -av containers/slop.sif user@remote-host:~/slop/containers/

# 2. Deploy code and register
python3 -m client.deploy user@remote-host --path ~/slop
```

**Scenario B: Custom Container Path**
If your `.sif` file is in a shared directory (e.g., `/shared/images/slop.sif`):

```bash
python3 -m client.deploy user@remote-host \
    --path ~/slop \
    --container /shared/images/slop.sif \
    --name my-gpu-node
```

---

## 4. Verification

Run the management tool to check connection and run a sanity test.

```bash
# List registered servers
python3 -m client.manage list

# Check connection
python3 -m client.manage check

# Run a quick compute verification (generates 1 step)
python3 -m client.manage check --verify
```

If you see `Status: ONLINE` and `Compute: OK`, you are ready.

---

## 5. Usage

Use the Python API in your notebooks or scripts:

```python
from client.interface import SlopClient
from client.config import registry

# Load the config we just deployed
config = registry.get("my-gpu-node")

with SlopClient(config) as client:
    result = client.generate(
        prompt="A photograph of a CEO",
        num_steps=50,
        capture_latents=True
    )
    
    print(f"Received image: {len(result.image)} bytes")
```
