# SLOP Deployment Guide

Easy deployment and testing for remote servers and local development.

## Quick Start

### Local Testing

```bash
# Run quick tests (fast, no model downloads)
python -m pytest tests/ -v

# Run with real models (slow, downloads models)
python -m pytest tests/ -v --run-slow

# Run GPU tests
python -m pytest tests/ -v --run-gpu
```

### Test Structure

- `tests/test_latent_extraction.py` - CLIP, DINOv2, embedding extraction
- `tests/test_diffusion_flow.py` - Stable Diffusion, FLUX, trajectory capture
- `tests/test_flow_analysis.py` - Physics operators, topology, flow fields
- `tests/conftest.py` - Shared fixtures and configuration

## Remote Server Deployment

### Option 1: Direct Deployment (Recommended)

```bash
# Deploy to remote server
./scripts/deploy.sh user@server:/path/to/deploy

# Or manually:
ssh user@server
cd /path/to/deploy
./scripts/setup_remote.sh
```

### Option 2: Docker Deployment

```bash
# Build Docker image
python scripts/docker_deploy.py build

# Run analysis in container
python scripts/docker_deploy.py run --mode quick --gpu
python scripts/docker_deploy.py run --mode full --prompts "cat,dog,bird"
```

## Running Analysis

### Quick Analysis (Fast, 5 min)

```bash
source .venv/bin/activate
python scripts/run_remote_analysis.py --mode quick
```

### Full Analysis (Comprehensive, 30-60 min)

```bash
python scripts/run_remote_analysis.py \
    --mode full \
    --prompts "a cat,a dog,a bird" \
    --num-trajectories 20 \
    --steps 50 \
    --output outputs/my_analysis
```

### Generate Trajectories Only

```bash
python scripts/run_remote_analysis.py \
    --mode trajectories \
    --prompts "test prompt" \
    --num-trajectories 10 \
    --output outputs/trajectories
```

### Analyze Existing Trajectories

```bash
python scripts/run_remote_analysis.py \
    --mode analysis \
    --output outputs/analysis_results
```

## Configuration

Edit `config.yaml` to set default parameters:

```yaml
model:
  diffusion: "CompVis/stable-diffusion-v1-4"
  clip: "openai/clip-vit-base-patch32"
  device: "cuda"

generation:
  num_trajectories: 20
  num_steps: 50
  resolution: 512

analysis:
  grid_resolution: 50
  radius: 0.8
```

## Output Structure

```
outputs/
├── trajectories/          # Generated trajectories
│   ├── trajectory_000.npy
│   └── ...
├── flow/                  # Flow field data
│   ├── flow_field_2d_velocity.npy
│   ├── flow_field_2d_divergence.npy
│   └── flow_field_2d_vorticity.npy
├── analysis/              # Analysis results
│   ├── statistics.json
│   ├── topology.json
│   ├── velocity.npy
│   └── divergence.npy
└── summary.json           # Overall summary
```

## Server Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 10GB disk space

### Recommended
- NVIDIA GPU (8GB+ VRAM)
- 16GB RAM
- 50GB disk space
- CUDA 11.8+

## Testing on Remote Server

```bash
# SSH into server
ssh user@server
cd /path/to/slop

# Activate environment
source .venv/bin/activate

# Run quick test
python -m pytest tests/test_flow_analysis.py -v

# Run full test suite (no model downloads)
python -m pytest tests/ -v -k "not slow"

# Run with models (downloads ~5GB)
python -m pytest tests/test_latent_extraction.py -v --run-slow
```

## Monitoring

```bash
# Check GPU usage
nvidia-smi

# Monitor logs
tail -f outputs/my_analysis/analysis.log

# Check results
cat outputs/my_analysis/summary.json
```

## Troubleshooting

### Out of Memory

Reduce batch size or resolution:
```bash
python scripts/run_remote_analysis.py \
    --mode quick \
    --resolution 256 \
    --num-trajectories 5
```

### Slow on CPU

Use smaller models or Docker with GPU:
```bash
python scripts/docker_deploy.py run --mode quick --gpu
```

### Model Download Issues

Set Hugging Face cache:
```bash
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache
```

## Advanced Usage

### Custom Analysis Pipeline

```python
from src.utils.ai import AILoader
from src.utils.tasks import TaskRunner
from src.utils.physics import PhysicsTools

# Load models
ai = AILoader(device="cuda")
ai.load_diffusion()

# Generate trajectories
runner = TaskRunner(device="cuda")
result = runner.generate_trajectories(
    prompts=["test"],
    num_trajectories=10
)

# Analyze
physics = PhysicsTools()
from src.analysis.flow_fields import compute_flow_field

grid, V = compute_flow_field(result['trajectories'])
stats = physics.compute_flow_statistics(V)
topology = physics.analyze_topology(V)
```

## Performance Tips

1. **Use GPU**: 10-50x faster than CPU
2. **Batch processing**: Process multiple prompts together
3. **Cache models**: Set `HF_HOME` to persistent directory
4. **Reduce resolution**: Use 256 or 512 instead of 1024
5. **Parallel generation**: Generate trajectories in parallel

## Support

- Check logs in `outputs/*/analysis.log`
- Run tests to verify setup: `pytest tests/ -v`
- Check GPU: `nvidia-smi`
