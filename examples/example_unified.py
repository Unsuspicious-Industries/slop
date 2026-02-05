"""Example: Using the new unified utilities for end-to-end analysis.

This script demonstrates how to use the clean new interfaces:
- AILoader: Load models easily
- TaskRunner: Run high-level tasks
- PhysicsTools: Analyze vector fields

Run with:
    python examples/example_unified.py --prompts data/prompts/test_prompts.txt --output outputs/example
"""

import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ai import AILoader, load_ai
from src.utils.tasks import TaskRunner, run_task
from src.utils.physics import PhysicsTools, quick_physics


def example_1_load_models():
    """Example 1: Load AI models the easy way."""
    print("\n" + "="*60)
    print("Example 1: Loading AI Models")
    print("="*60)
    
    # Method 1: Individual loading
    ai = AILoader()
    
    # Load diffusion model with trajectory tracking
    pipe = ai.load_diffusion("sd21", enable_tracking=True)
    
    # Load encoder
    encoder = ai.load_encoder("clip")
    
    # Load multimodal extractor
    extractor = ai.load_extractor(strategy="concat")
    
    print("✓ All models loaded successfully!")
    
    # Method 2: Load multiple at once
    models = load_ai(
        diffusion="sd21",
        encoder="clip",
        enable_tracking=True
    )
    print(f"✓ Loaded {len(models)} components")


def example_2_extract_embeddings():
    """Example 2: Extract embeddings from images."""
    print("\n" + "="*60)
    print("Example 2: Extracting Embeddings")
    print("="*60)
    
    runner = TaskRunner()
    
    # Extract embeddings from a directory of images
    # This will automatically batch process and save results
    embeddings = runner.extract_latents(
        images=Path("data/historical/processed"),  # or list of images
        encoder="multi",  # Use multimodal (CLIP + DINOv2)
        strategy="concat",
        output_path=Path("outputs/embeddings.npy"),
        batch_size=32
    )
    
    print(f"✓ Extracted embeddings: {embeddings.shape}")


def example_3_capture_trajectories():
    """Example 3: Capture diffusion trajectories."""
    print("\n" + "="*60)
    print("Example 3: Capturing Diffusion Trajectories")
    print("="*60)
    
    runner = TaskRunner()
    
    prompts = [
        "a doctor",
        "a nurse",
        "a scientist"
    ]
    
    # Or load from file:
    # prompts = Path("data/prompts/test_prompts.txt")
    
    trajectories = runner.capture_trajectories(
        prompts=prompts,
        model="sd21",
        output_dir=Path("outputs/trajectories"),
        steps=50,
        sample_rate=5,
        save_images=True
    )
    
    print(f"✓ Captured {len(trajectories)} trajectories")


def example_4_analyze_flow():
    """Example 4: Analyze flow field from trajectories."""
    print("\n" + "="*60)
    print("Example 4: Analyzing Flow Field")
    print("="*60)
    
    runner = TaskRunner()
    
    # Analyze flow from saved trajectories
    flow = runner.analyze_flow(
        trajectories=Path("data/generated/trajectories"),
        resolution=50,
        radius=0.5,
        output_dir=Path("outputs/flow_analysis"),
        compute_topology=True
    )
    
    print(f"✓ Flow analysis complete")
    print(f"  - Dimensions: {flow['dimensions']}D")
    print(f"  - Resolution: {flow['resolution']}")
    if 'topology' in flow:
        topo = flow['topology']
        print(f"  - Attractors: {len(topo['attractors'])}")
        print(f"  - Repellers: {len(topo['repellers'])}")
    
    # Use the flow data for physics analysis
    V = flow['velocity']
    return V


def example_5_physics_analysis(V):
    """Example 5: Comprehensive physics analysis."""
    print("\n" + "="*60)
    print("Example 5: Physics Analysis")
    print("="*60)
    
    physics = PhysicsTools()
    
    # Compute differential operators
    div_V = physics.divergence(V)
    curl_V = physics.curl(V)
    magnitude = physics.magnitude(V)
    
    print(f"✓ Computed fields:")
    print(f"  - Divergence (mean): {div_V.mean():.3e}")
    print(f"  - Curl (mean): {abs(curl_V).mean():.3e}")
    print(f"  - Velocity (mean): {magnitude.mean():.3f}")
    
    # Find attractors and repellers
    attractors, repellers = physics.find_attractors_repellers(V)
    print(f"✓ Found {len(attractors)} attractors, {len(repellers)} repellers")
    
    # Check flow properties
    is_incomp = physics.is_incompressible(V, tolerance=1e-3)
    is_irrot = physics.is_irrotational(V, tolerance=1e-3)
    print(f"✓ Flow properties:")
    print(f"  - Incompressible: {is_incomp}")
    print(f"  - Irrotational: {is_irrot}")
    
    # Full topology analysis
    topology = physics.analyze_topology(V)
    
    # Save everything
    physics.save_analysis(
        V,
        output_dir=Path("outputs/physics_analysis"),
        compute_topology=True,
        save_fields=True
    )


def example_6_convenience_functions():
    """Example 6: Use convenience functions for quick operations."""
    print("\n" + "="*60)
    print("Example 6: Quick Convenience Functions")
    print("="*60)
    
    # Quick task execution
    embeddings = run_task(
        "extract",
        images=Path("outputs/images"),
        encoder="clip",
        output_path=Path("outputs/quick_embeddings.npy")
    )
    print(f"✓ Quick extract: {embeddings.shape}")
    
    # Quick physics
    import numpy as np
    V = np.random.randn(50, 50, 2) * 0.1  # Dummy vector field
    
    div_V = quick_physics(V, "div")
    curl_V = quick_physics(V, "curl")
    topology = quick_physics(V, "topology")
    
    print(f"✓ Quick physics complete")


def full_pipeline_example(prompts_file: Path, output_dir: Path):
    """Full pipeline: Generate -> Extract -> Analyze -> Physics."""
    print("\n" + "="*60)
    print("FULL PIPELINE EXAMPLE")
    print("="*60)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize
    runner = TaskRunner()
    physics = PhysicsTools()
    
    # Step 1: Generate images with trajectories
    print("\n[1/4] Generating images and capturing trajectories...")
    trajectories = runner.capture_trajectories(
        prompts=prompts_file,
        model="sd21",
        output_dir=output_dir / "generation",
        steps=50,
        sample_rate=5
    )
    
    # Step 2: Extract embeddings from generated images
    print("\n[2/4] Extracting embeddings...")
    embeddings = runner.extract_latents(
        images=output_dir / "generation" / "images",
        encoder="multi",
        strategy="concat",
        output_path=output_dir / "embeddings.npy"
    )
    
    # Step 3: Analyze flow field
    print("\n[3/4] Analyzing flow field...")
    flow = runner.analyze_flow(
        trajectories=output_dir / "generation" / "trajectories",
        resolution=50,
        output_dir=output_dir / "flow",
        compute_topology=True
    )
    
    # Step 4: Physics analysis
    print("\n[4/4] Running physics analysis...")
    V = flow['velocity']
    
    # Find critical points
    attractors, repellers = physics.find_attractors_repellers(V)
    
    # Compute statistics
    stats = physics.compute_flow_statistics(V)
    
    # Save comprehensive analysis
    physics.save_analysis(
        V,
        output_dir=output_dir / "physics",
        compute_topology=True,
        save_fields=True
    )
    
    print("\n" + "="*60)
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    print(f"Results saved to: {output_dir}")
    print(f"\nSummary:")
    print(f"  - Generated {len(trajectories)} images")
    print(f"  - Extracted {embeddings.shape[0]} embeddings (dim={embeddings.shape[1]})")
    print(f"  - Found {len(attractors)} attractors, {len(repellers)} repellers")
    print(f"  - Mean velocity: {stats['velocity_mean']:.3f}")
    print(f"  - Mean divergence: {stats['divergence_mean']:.3e}")


def main():
    parser = argparse.ArgumentParser(description="Example usage of unified utilities")
    parser.add_argument("--example", type=int, choices=range(1, 7), help="Run specific example")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument("--prompts", type=Path, help="Prompts file for full pipeline")
    parser.add_argument("--output", type=Path, default="outputs/example", help="Output directory")
    args = parser.parse_args()
    
    if args.full:
        if not args.prompts:
            print("Error: --prompts required for full pipeline")
            return
        full_pipeline_example(args.prompts, args.output)
    elif args.example:
        if args.example == 1:
            example_1_load_models()
        elif args.example == 2:
            example_2_extract_embeddings()
        elif args.example == 3:
            example_3_capture_trajectories()
        elif args.example == 4:
            V = example_4_analyze_flow()
            if V is not None:
                example_5_physics_analysis(V)
        elif args.example == 5:
            import numpy as np
            V = np.random.randn(50, 50, 2) * 0.1
            example_5_physics_analysis(V)
        elif args.example == 6:
            example_6_convenience_functions()
    else:
        # Run all examples
        print("\nRunning all examples...")
        example_1_load_models()
        # Skip examples 2-4 as they need actual data
        print("\n(Skipping examples 2-4 as they require data)")
        example_6_convenience_functions()
        
        print("\n" + "="*60)
        print("To run the full pipeline:")
        print("  python examples/example_unified.py --full --prompts data/prompts/test_prompts.txt")
        print("\nTo run a specific example:")
        print("  python examples/example_unified.py --example 1")


if __name__ == "__main__":
    main()
