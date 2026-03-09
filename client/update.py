"""Update tool to sync code changes to an existing registered provider."""
import argparse
import subprocess
import sys
import shlex
from pathlib import Path


def _looks_like_project_root(path: Path) -> bool:
    required = ["client", "server", "shared"]
    return all((path / name).exists() for name in required)


def _detect_project_root() -> Path:
    candidates = []

    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    file_root = Path(__file__).resolve().parent.parent
    candidates.extend([file_root, *file_root.parents])

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if _looks_like_project_root(candidate):
            return candidate

    return file_root


project_root = _detect_project_root()
sys.path.append(str(project_root))

from client.config import registry

def run_command(cmd, shell=False):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        subprocess.check_call(cmd, shell=shell)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(1)

def update(server_name: str, verbose: bool = False):
    """Sync code to the remote server."""
    config = registry.get(server_name)
    if not config:
        print(f"Error: Provider '{server_name}' not found in registry.")
        print("Use 'python -m client.manage list' to see available servers.")
        sys.exit(1)

    if config.kind != "ssh":
        print(f"Provider '{server_name}' is not an ssh provider.")
        sys.exit(1)

    print(f"Updating provider '{server_name}' at {config.target}:{config.remote_path}...")
    
    # Sync folders
    # We sync: shared/, server/, containers/ (if exists), requirements.txt
    sources = ["shared", "server"]
    
    # Optional sources
    if (project_root / "containers").exists():
        sources.append("containers")
    if (project_root / "requirements.txt").exists():
        sources.append("requirements.txt")
        
    cmd = [
        "rsync", "-avz",
        "--bwlimit=5000", # Limit bandwidth
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".git",
        "--exclude", ".DS_Store",
    ]
    
    for src in sources:
        cmd.append(str(project_root / src))
        
    # Destination
    cmd.append(f"{config.target}:{config.remote_path}/")
    
    run_command(cmd)
    print(f"Update complete for '{server_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Update code on a registered ssh provider.")
    parser.add_argument("server", help="Provider name to update")
    
    args = parser.parse_args()
    update(args.server)

if __name__ == "__main__":
    main()
