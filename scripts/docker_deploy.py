"""Docker build and deployment helper.

Usage:
    python scripts/docker_deploy.py build
    python scripts/docker_deploy.py run --mode quick
"""

import subprocess
import argparse
from pathlib import Path


def build_image():
    """Build Docker image."""
    print("Building Docker image...")
    
    cmd = [
        "docker", "build",
        "-t", "slop:latest",
        "."
    ]
    
    result = subprocess.run(cmd, check=True)
    print("Image built successfully")


def run_container(args):
    """Run analysis in container."""
    print("Running container...")
    
    # Mount directories
    output_dir = Path(args.output).absolute()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        "docker", "run",
        "--gpus", "all" if args.gpu else "0",
        "-v", f"{output_dir}:/app/outputs",
        "-v", f"{Path.home()}/.cache/huggingface:/app/.cache/huggingface",
        "slop:latest",
        "python3", "scripts/run_remote_analysis.py",
        "--mode", args.mode,
        "--output", "/app/outputs"
    ]
    
    if args.prompts:
        cmd.extend(["--prompts", args.prompts])
    
    if args.num_trajectories:
        cmd.extend(["--num-trajectories", str(args.num_trajectories)])
    
    subprocess.run(cmd, check=True)
    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Build command
    subparsers.add_parser("build", help="Build Docker image")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run container")
    run_parser.add_argument("--mode", default="quick", help="Analysis mode")
    run_parser.add_argument("--output", default="outputs/docker_run", help="Output directory")
    run_parser.add_argument("--gpu", action="store_true", help="Use GPU")
    run_parser.add_argument("--prompts", help="Comma-separated prompts")
    run_parser.add_argument("--num-trajectories", type=int, help="Number of trajectories")
    
    args = parser.parse_args()
    
    if args.command == "build":
        build_image()
    elif args.command == "run":
        run_container(args)


if __name__ == "__main__":
    main()
