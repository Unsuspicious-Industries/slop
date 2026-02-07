"""Update tool to sync code changes to an existing registered server."""
import argparse
import subprocess
import sys
import shlex
from pathlib import Path

# Fix path to include project root
project_root = Path(__file__).parent.parent
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
        print(f"Error: Server '{server_name}' not found in registry.")
        print("Use 'python -m client.manage list' to see available servers.")
        sys.exit(1)

    print(f"Updating server '{server_name}' at {config.host}:{config.remote_path}...")
    
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
    cmd.append(f"{config.host}:{config.remote_path}/")
    
    run_command(cmd)
    print(f"Update complete for '{server_name}'.")

def main():
    parser = argparse.ArgumentParser(description="Update code on a registered SLOP server.")
    parser.add_argument("server", help="Server name/alias to update")
    
    args = parser.parse_args()
    update(args.server)

if __name__ == "__main__":
    main()
