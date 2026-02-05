# Unified Utilities Guide

Clean, easy-to-use interfaces for AI models, tasks, and physics analysis.

## Quick Start

```python
from src.utils.ai import AILoader, load_ai
from src.utils.tasks import TaskRunner, run_task
from src.utils.physics import PhysicsTools, quick_physics
```

## 🤖 AI Loader

Load models with simple, consistent interface.

### Load Diffusion Models

```python
ai = AILoader()

# Load with shorthand
pipe = ai.load_diffusion("sd21", enable_tracking=True)

# Or use full HuggingFace paths
pipe = ai.load_diffusion("stabilityai/stable-diffusion-2-1")

# Supported shorthands:
# - "sd21" → Stable Diffusion 2.1
# - "sdxl" → Stable Diffusion XL
# - "flux-dev" → FLUX.1-dev
# - "flux-schnell" → FLUX.1-schnell
```

### Load Encoders

```python
# Load CLIP encoder
encoder = ai.load_encoder("clip")

# Load DINOv2 encoder
encoder = ai.load_encoder("dinov2")

# Load multimodal extractor (CLIP + DINOv2)
extractor = ai.load_extractor(strategy="concat")
```

### Quick Loading

```python
# Load multiple components at once
models = load_ai(
    diffusion="sd21",
    encoder="clip",
    extractor_strategy="concat",
    enable_tracking=True
)

pipe = models["diffusion"]
encoder = models["encoder"]
extractor = models["extractor"]
```

## 🎯 Task Runner

High-level interfaces for common workflows.

### Extract Embeddings

```python
runner = TaskRunner()

# Extract from image directory
embeddings = runner.extract_latents(
    images="outputs/images",
    encoder="multi",  # Use CLIP + DINOv2
    strategy="concat",
    output_path="outputs/embeddings.npy",
    batch_size=32
)
```

### Capture Trajectories

```python
# Generate images and capture diffusion trajectories
trajectories = runner.capture_trajectories(
    prompts=["a doctor", "a nurse", "a scientist"],
    model="sd21",
    output_dir="outputs/trajectories",
    steps=50,
    sample_rate=5,
    save_images=True
)
```

### Analyze Flow

```python
# Compute flow field from trajectories
flow = runner.analyze_flow(
    trajectories="data/trajectories",
    resolution=50,
    radius=0.5,
    output_dir="outputs/flow",
    compute_topology=True
)

# Access results
V = flow["velocity"]
div_V = flow["divergence"]
curl_V = flow["curl"]
topology = flow["topology"]
```

### Run Physics Analysis

```python
# Comprehensive physics analysis
physics_data = runner.run_physics_analysis(
    vector_field=V,
    spacing=0.02,
    find_critical_points=True,
    classify_points=True
)
```

### Quick Tasks

```python
# One-line task execution
embeddings = run_task("extract", images="outputs/images", encoder="clip")
flow = run_task("flow", trajectories="data/trajectories", resolution=50)
```

## ⚛️ Physics Tools

Easy-to-use physics operations with clear documentation.

### Differential Operators

```python
physics = PhysicsTools()

# Compute divergence: div(V) = dVx/dx + dVy/dy + dVz/dz
# Positive = diverging (source), Negative = converging (sink)
div_V = physics.divergence(V, dx=0.02, dy=0.02)

# Compute curl: curl(V) measures rotation
# Positive = counterclockwise, Negative = clockwise
curl_V = physics.curl(V, dx=0.02, dy=0.02)

# Compute gradient: grad(phi) points in direction of steepest ascent
grad_phi = physics.gradient(phi, dx=0.02, dy=0.02)

# Compute Laplacian: Δphi measures diffusion/smoothness
laplacian_phi = physics.laplacian(phi, dx=0.02, dy=0.02)
```

### Field Analysis

```python
# Compute magnitude
magnitude = physics.magnitude(V)

# Find critical points (where V ≈ 0)
critical_points = physics.find_critical_points(V, threshold=0.01)

# Classify critical point type
classification = physics.classify_critical_point(V, point=(25, 25))
# Returns: {"type": "attractor", "eigenvalues": [...], "stability": "stable"}

# Find attractors and repellers
attractors, repellers = physics.find_attractors_repellers(V)

# Comprehensive topology analysis
topology = physics.analyze_topology(V)
```

### Flow Properties

```python
# Compute flow statistics
stats = physics.compute_flow_statistics(V)
print(f"Mean velocity: {stats['velocity_mean']}")
print(f"Mean divergence: {stats['divergence_mean']}")

# Check flow properties
is_incompressible = physics.is_incompressible(V)  # div(V) ≈ 0
is_irrotational = physics.is_irrotational(V)      # curl(V) ≈ 0
```

### Save Analysis

```python
# Save comprehensive analysis with all fields
results = physics.save_analysis(
    V,
    output_dir="outputs/physics",
    compute_topology=True,
    save_fields=True
)
```

### Quick Physics

```python
# One-line operations
div_V = quick_physics(V, "div")
curl_V = quick_physics(V, "curl")
topology = quick_physics(V, "topology")

# Get everything at once
all_data = quick_physics(V, "all")
```

## 📚 Complete Example

```python
from pathlib import Path
from src.utils.ai import AILoader
from src.utils.tasks import TaskRunner
from src.utils.physics import PhysicsTools

# Initialize
ai = AILoader()
runner = TaskRunner()
physics = PhysicsTools()

# 1. Generate images with trajectory tracking
trajectories = runner.capture_trajectories(
    prompts=["a doctor", "a nurse"],
    model="sd21",
    output_dir="outputs/generation",
    steps=50
)

# 2. Extract embeddings
embeddings = runner.extract_latents(
    images="outputs/generation/images",
    encoder="multi",
    output_path="outputs/embeddings.npy"
)

# 3. Analyze flow
flow = runner.analyze_flow(
    trajectories="outputs/generation/trajectories",
    resolution=50,
    output_dir="outputs/flow"
)

# 4. Physics analysis
V = flow["velocity"]
attractors, repellers = physics.find_attractors_repellers(V)
topology = physics.analyze_topology(V)
physics.save_analysis(V, output_dir="outputs/physics")

print(f"Found {len(attractors)} attractors, {len(repellers)} repellers")
```

## 🎓 Physics Concepts

### Divergence
- **Formula**: `div(V) = ∂Vx/∂x + ∂Vy/∂y + ∂Vz/∂z`
- **Meaning**: Measures expansion/contraction of flow
- **Positive**: Flow diverging (source, repeller)
- **Negative**: Flow converging (sink, attractor)
- **Zero**: Incompressible flow (volume-preserving)

### Curl
- **Formula 2D**: `curl(V) = ∂Vy/∂x - ∂Vx/∂y`
- **Formula 3D**: `curl(V) = [∂Vz/∂y - ∂Vy/∂z, ∂Vx/∂z - ∂Vz/∂x, ∂Vy/∂x - ∂Vx/∂y]`
- **Meaning**: Measures rotation of flow
- **Positive**: Counterclockwise rotation
- **Negative**: Clockwise rotation
- **Zero**: Irrotational flow (no rotation)

### Gradient
- **Formula**: `grad(φ) = [∂φ/∂x, ∂φ/∂y, ∂φ/∂z]`
- **Meaning**: Points in direction of steepest ascent
- **Use**: Convert scalar field to vector field

### Laplacian
- **Formula**: `Δφ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²`
- **Meaning**: Measures diffusion/smoothness
- **Positive**: Local minimum (diffusion out)
- **Negative**: Local maximum (diffusion in)
- **Zero**: Harmonic function (balanced)

### Critical Points
- **Attractor (Stable Node)**: Flow converges, eigenvalues negative
- **Repeller (Unstable Node)**: Flow diverges, eigenvalues positive
- **Saddle**: Flow converges in one direction, diverges in other
- **Center/Spiral**: Rotating flow, complex eigenvalues

## Advanced Usage

### Custom Grid Spacing

```python
# Different spacing in each direction
div_V = physics.divergence(V, dx=0.02, dy=0.03, dz=0.01)

# Use tuple for spacing
spacing = (0.02, 0.03, 0.01)
results = runner.run_physics_analysis(V, spacing=spacing)
```

### Batch Processing

```python
# Process multiple image directories
for image_dir in ["outputs/run1", "outputs/run2"]:
    embeddings = runner.extract_latents(
        images=image_dir,
        encoder="clip",
        output_path=f"{image_dir}/embeddings.npy"
    )
```

### Custom Models

```python
# Use any HuggingFace model
pipe = ai.load_diffusion("runwayml/stable-diffusion-v1-5")
encoder = ai.load_encoder("clip", model_name="openai/clip-vit-base-patch32")

# Use HF bridge for arbitrary models
bridge = ai.load_hf_bridge("meta-llama/Llama-2-7b-hf")
```

## 📝 Notes

- All utilities handle device management automatically (CUDA/CPU)
- Verbose output can be disabled with `verbose=False`
- Results are automatically saved when `output_path` or `output_dir` is provided
- Progress bars appear for long operations
- All physics operators work on arbitrary-sized arrays using trailing axis indexing
- Grid spacing defaults to 1.0 if not specified

## Migration Guide

### Old Code
```python
from src.diffusion.loaders import load_diffusion_model
from src.diffusion.sd_hook import SDTrajectoryHook
pipe = load_diffusion_model("stabilityai/stable-diffusion-2-1")
hook = SDTrajectoryHook(pipe)
```

### New Code
```python
from src.utils.ai import AILoader
ai = AILoader()
pipe = ai.load_diffusion("sd21", enable_tracking=True)
```

### Old Code
```python
from src.physics.operators import divergence, curl
div_V = divergence(V, dx=0.02, dy=0.02)
curl_V = curl(V, dx=0.02, dy=0.02)
```

### New Code
```python
from src.utils.physics import PhysicsTools
physics = PhysicsTools()
div_V = physics.divergence(V, dx=0.02, dy=0.02)
curl_V = physics.curl(V, dx=0.02, dy=0.02)
```

Both old and new APIs work - use whichever you prefer!
