"""Deployment tool to push code to remote providers and register them."""
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

from client.config import ProviderConfig, Registry

def run_command(cmd, shell=False):
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        subprocess.check_call(cmd, shell=shell)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
        sys.exit(1)

def deploy(args):
    """
    1. Rsync code to remote.
    2. Register in config.
    """
    target = args.target
    remote_path = args.path
    name = args.name or target.split('@')[-1]
    
    # 1. Rsync
    print(f"Deploying to {target}:{remote_path}...")
    
    # Ensure remote dir exists
    # Safety: Quote the path to prevent injection and handle spaces
    safe_remote_path = shlex.quote(remote_path)
    run_command(["ssh", target, f"mkdir -p {safe_remote_path}"])
    
    # Sync folders needed for runtime and examples.
    sources = ["shared", "server", "client"]
    if (project_root / "experiments").exists():
        sources.append("experiments")
    if (project_root / "notebooks").exists():
        sources.append("notebooks")
    if (project_root / "distill").exists():
        sources.append("distill")
    if (project_root / "containers").exists():
        sources.append("containers")
    if (project_root / "requirements.txt").exists():
        sources.append("requirements.txt")
        
    cmd = [
        "rsync", "-avz",
        "--bwlimit=5000", # Limit bandwidth to 5MB/s to be "nice" to the network
        "--exclude", "__pycache__",
        "--exclude", "*.pyc",
        "--exclude", ".git",
        "--exclude", ".DS_Store",
        # Never deploy generated data/artifacts (huge, slow, and not needed)
        "--exclude", "distill/data",
        "--exclude", "distill/output",
        "--exclude", "notebooks/distill/data",
        "--exclude", ".ipynb_checkpoints",
    ]
    
    for src in sources:
        cmd.append(str(project_root / src))
        
    cmd.append(f"{target}:{remote_path}/")
    
    run_command(cmd)
    
    # 2. Register
    reg = Registry()
    
    # Logic for defaults
    container_image = args.container
    python_cmd = args.python_cmd

    if not python_cmd and not container_image:
        # DEFAULT MODE: Singularity
        # We assume the container is named slop.sif and located in the containers dir
        container_image = f"{remote_path}/containers/slop.sif"
        python_cmd = "python3"
        print(f"Defaulting to Singularity deployment.")
        print(f"  Container: {container_image}")
    elif not python_cmd:
        python_cmd = "python3"
        
    config = ProviderConfig(
        name=name,
        kind="local" if target == "local" else "ssh",
        target=target,
        remote_path=remote_path,
        python_cmd=python_cmd,
        container_image=container_image,
        num_workers=args.workers
    )
    
    reg.add(config)
    print(f"Successfully deployed and registered provider '{name}'!")
    if container_image:
        print(f"  Container: {container_image}")
        
        # Verify container existence
        print("Verifying remote container existence...")
        safe_container_image = shlex.quote(container_image)
        check_cmd = ["ssh", target, f"test -f {safe_container_image}"]
        try:
            subprocess.check_call(check_cmd)
            print("  [OK] Container image found on remote.")
        except subprocess.CalledProcessError:
            print("\n" + "="*60)
            print("  WARNING: Container image NOT found on remote!")
            
            if args.build:
                print("  [--build] flag detected. Attempting to build remotely...")
                print("  This may take 10-20 minutes. Please wait.")
                
                # Construct build command
                # We assume the .def file is named the same as the .sif file
                # e.g. slop.sif -> slop.def
                sif_path = Path(container_image)
                def_path = sif_path.with_suffix(".def")
                container_dir = sif_path.parent
                
                # Safety: Quote all paths for the remote shell
                safe_dir = shlex.quote(str(container_dir))
                safe_sif = shlex.quote(sif_path.name)
                safe_def = shlex.quote(def_path.name)
                
                # Intelligent resource management for shared environments
                # 1. CPU: nice -n 19 (Lowest priority)
                # 2. IO: ionice -c 3 (Idle priority) - prevents freezing shared disks
                # 3. Disk: Use local scratch (like /scratch or /tmp) instead of the deployment dir
                #    to avoid massive I/O on the shared network filesystem (NFS/Lustre).
                
                find_scratch_cmd = (
                    "for d in /scratch /local_scratch /tmp /var/tmp; do "
                    "  if [ -d \"$d\" ] && [ -w \"$d\" ]; then "
                    "    echo \"$d\"; break; "
                    "  fi; "
                    "done"
                )

                if args.unrestricted:
                    print("  [--unrestricted] Full speed build enabled (No throttling).")
                    prefix_cmd = ""
                    ionice_cmd = ""
                else:
                    prefix_cmd = "nice -n 15"
                    ionice_cmd = "$IONICE"

                build_logic = (
                    # 1. Find best local scratch
                    f"SCRATCH=$({find_scratch_cmd}); "
                    f"if [ -z \"$SCRATCH\" ]; then "
                    f"  echo 'WARNING: No local scratch found. Falling back to {safe_dir}/tmp (Network I/O warning!)'; "
                    f"  SCRATCH={safe_dir}/tmp; "
                    f"  mkdir -p $SCRATCH; "
                    f"fi; "
                    f"echo \"Using build scratch: $SCRATCH (avoids shared disk I/O)\"; "
                    
                    # 2. Configure Apptainer/Singularity to use it
                    f"export APPTAINER_TMPDIR=$SCRATCH/slop_tmp_$USER; "
                    f"export APPTAINER_CACHEDIR=$SCRATCH/slop_cache_$USER; "
                    f"export SINGULARITY_TMPDIR=$SCRATCH/slop_tmp_$USER; "
                    f"export SINGULARITY_CACHEDIR=$SCRATCH/slop_cache_$USER; "
                    f"mkdir -p $APPTAINER_TMPDIR $APPTAINER_CACHEDIR; "
                    
                    # SAFETY: Ensure cleanup happens even if build fails/is killed
                    f"trap 'rm -rf $APPTAINER_TMPDIR $APPTAINER_CACHEDIR' EXIT; "

                    # 3. Setup IO throttling
                    # We use class 2 (Best Effort) with priority 7 (lowest) instead of class 3 (Idle)
                    # Class 3 can cause indefinite stalling if the disk is busy.
                    f"IONICE=\"\"; "
                    f"if command -v ionice > /dev/null; then IONICE=\"ionice -c 2 -n 7\"; fi; "
                    
                    # 4. Execute Build with Smart Fallbacks
                    f"echo 'Starting container build...'; "
                    f"if command -v apptainer > /dev/null; then "
                    f"  echo 'Strategy: Apptainer (Standard)'; "
                    f"  {prefix_cmd} {ionice_cmd} apptainer build --force {safe_sif} {safe_def}; "
                    f"elif command -v singularity > /dev/null; then "
                    # Check for fakeroot support (feature available in newer versions)
                    f"  if singularity build --help | grep -q fakeroot; then "
                    f"    echo 'Strategy: Singularity --fakeroot'; "
                    f"    {prefix_cmd} {ionice_cmd} singularity build --fakeroot --force {safe_sif} {safe_def}; "
                    f"  else "
                    # Try direct build first (might work if admin set suid)
                    f"    echo 'Strategy: Singularity (Direct)'; "
                    f"    if ! {prefix_cmd} {ionice_cmd} singularity build --force {safe_sif} {safe_def}; then "
                    f"      echo 'Direct build failed. Trying sudo...'; "
                    f"      {prefix_cmd} {ionice_cmd} sudo singularity build --force {safe_sif} {safe_def}; "
                    f"    fi; "
                    f"  fi; "
                    f"else "
                    f"  echo 'ERROR: Neither Apptainer nor Singularity found in PATH.'; "
                    f"  exit 1; "
                    f"fi; "
                )
                
                full_remote_cmd = (
                    f"cd {safe_dir} && "
                    # Check disk space on scratch (need ~10GB)
                    f"AVAIL=$(df -k $({find_scratch_cmd}) | awk 'NR==2 {{print $4}}'); "
                    f"if [ \"$AVAIL\" -lt 10485760 ]; then "
                    f"  echo 'WARNING: Low disk space on scratch. Build might fail.'; "
                    f"fi; "
                    f"{build_logic}"
                )
                
                try:
                    subprocess.check_call(["ssh", target, full_remote_cmd])
                    print("  [SUCCESS] Container built successfully!")
                except subprocess.CalledProcessError:
                    print("  [FAIL] Automatic build failed.")
                    print("  Please log in and build manually.")
            else:
                sif_path = Path(container_image)
                container_dir = sif_path.parent
                
                print("  You must build the container on the server before running inference.")
                print("  Tip: Run deploy again with --build to attempt automatic building.")
                print(f"  OR Run these commands on {target}:")
                # For display purposes, we don't strictly need quotes if paths are simple,
                # but it's good practice to show them if the path has spaces.
                display_dir = shlex.quote(str(container_dir))
                print(f"    cd {display_dir}")
                print("    apptainer build slop.sif slop.def")
                print("    # OR: sudo singularity build slop.sif slop.def")
            print("="*60 + "\n")
            
    print(f"Test connection with: python -m client.manage check {name}")

def main():
    parser = argparse.ArgumentParser(description="Deploy SLOP server code to remote host.")
    parser.add_argument("target", help="SSH target (user@hostname)")
    parser.add_argument("--path", required=True, help="Remote installation path (e.g. ~/slop)")
    parser.add_argument("--name", help="Alias for this server (default: hostname)")
    parser.add_argument("--python-cmd", help="Command to run python on remote (e.g. /opt/conda/bin/python)")
    parser.add_argument("--container", help="Path to Singularity .sif image on remote (overrides python-cmd)")
    parser.add_argument("--build", action="store_true", help="Attempt to build the container on the remote server")
    parser.add_argument("--unrestricted", action="store_true", help="Disable resource throttling (nice/ionice) for servers you own")
    parser.add_argument("--workers", type=int, default=4, help="Number of CPU threads to use (default: 4)")
    
    args = parser.parse_args()
    deploy(args)

if __name__ == "__main__":
    main()
