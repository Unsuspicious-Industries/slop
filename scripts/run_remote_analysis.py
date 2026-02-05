#!/usr/bin/env python3
"""Remote execution script for running analysis on servers.

This script handles:
- Model downloading and caching
- Trajectory generation
- Flow analysis
- Result collection

Usage:
    python scripts/run_remote_analysis.py --config config.yaml --mode full
    python scripts/run_remote_analysis.py --mode quick --prompts "test,demo"
"""

import argparse
import sys
from pathlib import Path
import json
import yaml
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.ai import AILoader
from src.utils.tasks import TaskRunner
from src.utils.physics import PhysicsTools
from src.utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run analysis on remote server")
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["quick", "full", "trajectories", "analysis"],
        default="full",
        help="Analysis mode"
    )
    
    parser.add_argument(
        "--prompts",
        type=str,
        help="Comma-separated prompts (overrides config)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/remote_run",
        help="Output directory"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda/cpu/auto)"
    )
    
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to generate"
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of diffusion steps"
    )
    
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help="Image resolution"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration file."""
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration")
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_device(device: str) -> str:
    """Determine device to use."""
    import torch
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if device == "cuda" and torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
        print(f"   Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("Warning: Using CPU (this will be slow)")
        device = "cpu"
    
    return device


def run_quick_analysis(args, config, logger):
    """Run quick analysis with minimal trajectories."""
    logger.info("Running quick analysis mode")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get prompts
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",")]
    else:
        prompts = ["a photo of a cat", "a photo of a dog"]
    
    logger.info(f"Generating trajectories for {len(prompts)} prompts")
    
    # Initialize
    device = setup_device(args.device)
    runner = TaskRunner(device=device)
    
    # Generate trajectories
    result = runner.generate_trajectories(
        prompts=prompts,
        num_trajectories=min(args.num_trajectories, 5),  # Quick mode: max 5
        num_steps=min(args.steps, 20),  # Quick mode: max 20 steps
        output_dir=output_dir / "trajectories"
    )
    
    logger.info(f"Generated {len(result['trajectories'])} trajectories")
    
    # Quick flow analysis
    physics = PhysicsTools()
    from src.analysis.flow_fields import compute_flow_field
    
    grid, V = compute_flow_field(
        result['trajectories'],
        grid_resolution=30,
        radius=0.5
    )
    
    # Statistics
    stats = physics.compute_flow_statistics(V, compute_all=True)
    
    # Save results
    with open(output_dir / "quick_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"Quick analysis complete. Results in {output_dir}")
    print(f"\nStatistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")


def run_full_analysis(args, config, logger):
    """Run full analysis pipeline."""
    logger.info("Running full analysis mode")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load prompts
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",")]
    else:
        prompt_file = Path("data/prompts/test_prompts.txt")
        if prompt_file.exists():
            prompts = prompt_file.read_text().strip().split("\n")
        else:
            prompts = ["a photo of a cat", "a photo of a dog", "a landscape"]
    
    logger.info(f"Processing {len(prompts)} prompts")
    
    # Initialize
    device = setup_device(args.device)
    runner = TaskRunner(device=device)
    
    # Generate trajectories
    logger.info("Generating trajectories...")
    traj_result = runner.generate_trajectories(
        prompts=prompts,
        num_trajectories=args.num_trajectories,
        num_steps=args.steps,
        resolution=args.resolution,
        output_dir=output_dir / "trajectories"
    )
    
    logger.info(f"Generated {len(traj_result['trajectories'])} trajectories")
    
    # Compute flow field
    logger.info("Computing flow field...")
    from src.analysis.flow_fields import compute_flow_field
    
    grid, V = compute_flow_field(
        traj_result['trajectories'],
        grid_resolution=50,
        radius=0.8,
        save_dir=output_dir / "flow"
    )
    
    # Full analysis
    logger.info("Analyzing flow topology...")
    physics = PhysicsTools()
    
    stats = physics.compute_flow_statistics(V, compute_all=True)
    topology = physics.analyze_topology(V)
    
    # Save all results
    physics.save_analysis(V, output_dir / "analysis", compute_topology=True, save_fields=True)
    
    # Summary report
    summary = {
        "config": {
            "prompts": len(prompts),
            "trajectories": args.num_trajectories,
            "steps": args.steps,
            "resolution": args.resolution,
            "device": device
        },
        "statistics": stats,
        "topology": {
            "attractors": len(topology['attractors']),
            "repellers": len(topology['repellers']),
            "saddles": len(topology['saddles'])
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Full analysis complete. Results in {output_dir}")
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"Prompts processed: {len(prompts)}")
    print(f"Trajectories generated: {len(traj_result['trajectories'])}")
    print(f"\nFlow Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nTopology:")
    print(f"  Attractors: {len(topology['attractors'])}")
    print(f"  Repellers: {len(topology['repellers'])}")
    print(f"  Saddles: {len(topology['saddles'])}")
    print("="*60)


def run_trajectory_generation(args, config, logger):
    """Only generate trajectories."""
    logger.info("Trajectory generation mode")
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.prompts:
        prompts = [p.strip() for p in args.prompts.split(",")]
    else:
        prompts = ["test prompt"]
    
    device = setup_device(args.device)
    runner = TaskRunner(device=device)
    
    result = runner.generate_trajectories(
        prompts=prompts,
        num_trajectories=args.num_trajectories,
        num_steps=args.steps,
        resolution=args.resolution,
        output_dir=output_dir
    )
    
    logger.info(f"Generated {len(result['trajectories'])} trajectories")


def run_analysis_only(args, config, logger):
    """Only run analysis on existing trajectories."""
    logger.info("Analysis-only mode")
    
    output_dir = Path(args.output)
    traj_dir = Path("data/generated/trajectories")
    
    if not traj_dir.exists():
        logger.error(f"Trajectory directory not found: {traj_dir}")
        return
    
    # Load trajectories
    import numpy as np
    traj_files = sorted(traj_dir.glob("trajectory_*.npy"))
    
    if not traj_files:
        logger.error("No trajectory files found")
        return
    
    logger.info(f"Loading {len(traj_files)} trajectories")
    trajectories = [np.load(f) for f in traj_files]
    
    # Analyze
    from src.analysis.flow_fields import compute_flow_field
    
    logger.info("Computing flow field...")
    grid, V = compute_flow_field(trajectories, grid_resolution=50)
    
    physics = PhysicsTools()
    physics.save_analysis(V, output_dir, compute_topology=True, save_fields=True)
    
    logger.info(f"Analysis complete. Results in {output_dir}")


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logger = setup_logger(
        name="remote_analysis",
        log_file=Path(args.output) / "analysis.log"
    )
    
    logger.info("="*60)
    logger.info("SLOP Remote Analysis")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {args.output}")
    
    # Load config
    config = load_config(args.config)
    
    try:
        if args.mode == "quick":
            run_quick_analysis(args, config, logger)
        elif args.mode == "full":
            run_full_analysis(args, config, logger)
        elif args.mode == "trajectories":
            run_trajectory_generation(args, config, logger)
        elif args.mode == "analysis":
            run_analysis_only(args, config, logger)
        
        logger.info("All tasks completed successfully")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
