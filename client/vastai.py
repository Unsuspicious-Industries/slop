"""Vast.ai deployment automation for SLOP."""
import json
import subprocess
import time
import sys
import os
import socket
import argparse
from typing import Optional, Dict, Tuple, List
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from client.deploy import deploy
from client.config import Registry, ServerConfig

class VastError(Exception):
    pass

class VastClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")
        if not self.api_key:
            # Check if logged in via CLI config
            if not self._check_cli_login():
                raise VastError("No API key found. Set VAST_API_KEY (or VASTAI_API_KEY) env var or run 'vastai set api-key'.")

    def _find_vastai_executable(self) -> str:
        """Find the vastai executable."""
        # Check PATH first
        from shutil import which
        if which("vastai"):
            return "vastai"
            
        # Check common user bin paths
        user_bin = Path.home() / "Library/Python/3.9/bin/vastai"
        if user_bin.exists():
            return str(user_bin)
            
        # Fallback (might fail)
        return "vastai"

    def _check_cli_login(self) -> bool:
        """Check if vastai CLI is usable."""
        try:
            self._run_cmd(["show", "user", "--raw"])
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _run_cmd(self, args: List[str]) -> Dict:
        """Run vastai command and return parsed JSON."""
        vast_bin = self._find_vastai_executable()
        cmd = [vast_bin] + args
        
        if self.api_key:
            # Inject API key if provided explicitly
            # Note: vastai CLI uses --api-key
            cmd = cmd + ["--api-key", self.api_key]
            
        cmd = cmd + ["--raw"]
        
        try:
            # Check if vastai is in PATH
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE)
            return json.loads(output)
        except subprocess.CalledProcessError as e:
            err_msg = e.stderr.decode().strip()
            raise VastError(f"Command failed: {' '.join(cmd)}\nError: {err_msg}")
        except FileNotFoundError:
            raise VastError(f"vastai CLI not found at '{vast_bin}'. Run 'pip install vastai' and ensure it is in your PATH.")
        except json.JSONDecodeError:
            raise VastError(f"Failed to parse JSON output from vastai.")

    def _wait_for_port(self, host: str, port: int, timeout: int = 300) -> bool:
        """Wait for a TCP port to open."""
        print(f"  Verifying connectivity to {host}:{port}...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                with socket.create_connection((host, port), timeout=2):
                    return True
            except (socket.timeout, ConnectionRefusedError, OSError):
                time.sleep(2)
        return False

    def list_instances(self):
        """List active instances."""
        instances = self._run_cmd(["show", "instances"])
        if not instances:
            print("No active instances found.")
            return

        print(f"{'ID':<10} {'Status':<15} {'Machine':<30} {'SSH':<25} {'Cost/Hr':<10}")
        print("-" * 90)
        for inst in instances:
            ssh = f"{inst.get('ssh_host', 'N/A')}:{inst.get('ssh_port', 'N/A')}"
            print(f"{inst['id']:<10} {inst['actual_status']:<15} {inst.get('machine_id', 'N/A'):<30} {ssh:<25} ${inst.get('dph_total', 0):<10}")

    def search_offers(self, query: str = "gpu_name=RTX_3090 rented=False reliability>0.98", limit: int = 5) -> List[Dict]:
        """Search for available GPUs."""
        # Sort by price ascending
        cmd = ["search", "offers", query, "-o", "dph"]
        offers = self._run_cmd(cmd)
        return offers[:limit]

    def create_instance(self, offer_id: int, image: str, disk_gb: int = 40) -> int:
        """Rent an instance."""
        # --ssh is implied for most base images, but explicit flag is good
        # --onstart-cmd to ensure sshd is running if needed, though Vast usually handles it
        onstart = "apt-get update; apt-get install -y openssh-server; mkdir -p /var/run/sshd; /usr/sbin/sshd -D"
        cmd = [
            "create", "instance", str(offer_id),
            "--image", image,
            "--disk", str(disk_gb),
            "--ssh",
            "--direct", # Attempt direct connect for speed
            "--onstart-cmd", onstart
        ]
        res = self._run_cmd(cmd)
        if res.get("success"):
            return res["new_contract"]
        raise VastError(f"Failed to create instance: {res}")

    def get_instance(self, instance_id: int) -> Optional[Dict]:
        """Get instance details."""
        instances = self._run_cmd(["show", "instances"])
        for inst in instances:
            if inst["id"] == instance_id:
                return inst
        return None

    def wait_for_running(self, instance_id: int, timeout: int = 600) -> Dict:
        """Wait for instance to enter 'running' state and have SSH ports assigned."""
        print(f"Waiting for instance {instance_id} to become ready...")
        start = time.time()
        while time.time() - start < timeout:
            inst = self.get_instance(instance_id)
            if inst:
                status = inst.get("actual_status")
                ssh_host = inst.get("ssh_host")
                ssh_port = inst.get("ssh_port")
                
                if status == "running" and ssh_host and ssh_port:
                    print(f"Instance {instance_id} reports running. Checking SSH...")
                    if self._wait_for_port(ssh_host, ssh_port):
                        print(f"Instance {instance_id} is ready at {ssh_host}:{ssh_port}")
                        return inst
                    else:
                        print(f"  Port verification failed for {ssh_host}:{ssh_port}")
                
                print(f"  Status: {status}, SSH: {ssh_host}:{ssh_port} (waiting...)")
            else:
                print(f"  Instance {instance_id} not found yet...")
                
            time.sleep(5)
            
        raise VastError(f"Timed out waiting for instance {instance_id}")

    def destroy_instance(self, instance_id: int):
        self._run_cmd(["destroy", "instance", str(instance_id)])


def build_and_push(image_tag: str, dockerfile_path: str = "containers/Dockerfile"):
    """Build and push the Docker image."""
    print(f"\n[Docker] Building {image_tag} from {dockerfile_path}...")
    
    # Build
    try:
        subprocess.check_call([
            "docker", "build", 
            "-t", image_tag, 
            "-f", dockerfile_path, 
            "."
        ])
    except subprocess.CalledProcessError:
        print("Docker build failed.")
        sys.exit(1)
        
    # Push
    print(f"[Docker] Pushing {image_tag}...")
    try:
        subprocess.check_call(["docker", "push", image_tag])
    except subprocess.CalledProcessError:
        print("Docker push failed. Not logged in?")
        sys.exit(1)
    
    print("[Docker] Build and push successful.\n")

def install_dependencies(alias: str):
    """Install Python dependencies on the remote server."""
    cmd = [
        "ssh", alias,
        "pip install -r /root/slop/requirements.txt"
    ]
    try:
        subprocess.check_call(cmd)
        print("[Dependencies] Installation successful.")
    except subprocess.CalledProcessError:
        print("[Dependencies] Installation failed. You may need to install manually.")
        sys.exit(1)

def provision(args):
    client = VastClient()
    
    # 0. Build & Push (Optional)
    if args.build_image:
        if not args.image_tag:
            print("Error: --image-tag required when building (e.g. user/slop:latest)")
            sys.exit(1)
        build_and_push(args.image_tag)
        # Use this image for provisioning
        target_image = args.image_tag
    else:
        target_image = args.image_tag or args.image # Use provided tag or default arg
    
    # 1. Search
    print(f"Searching for GPUs with query: '{args.query}'...")
    offers = client.search_offers(args.query)
    
    if not offers:
        print("No matching offers found.")
        sys.exit(1)
        
    # Display offers
    print("\nAvailable Offers:")
    for i, offer in enumerate(offers):
        dph = offer.get('dph_total', offer.get('dph_base', 'N/A'))
        print(f"{i}: ID={offer['id']} | GPU={offer['gpu_name']} x{offer['num_gpus']} | "
              f"Price=${dph}/hr | Rel={offer['reliability']}")
              
    if args.auto_yes:
        choice = 0
    else:
        try:
            sel = input("\nSelect offer number (0-N) or 'q' to quit: ")
            if sel.lower() == 'q':
                sys.exit(0)
            choice = int(sel)
        except ValueError:
            print("Invalid selection.")
            sys.exit(1)
            
    selected_offer = offers[choice]
    print(f"Selected offer {selected_offer['id']}")
    
    # 2. Rent
    print(f"Creating instance with image '{target_image}'...")
    instance_id = client.create_instance(selected_offer['id'], target_image, args.disk)
    print(f"Instance {instance_id} created.")
    
    try:
        # 3. Wait for SSH
        inst = client.wait_for_running(instance_id)
        
        ssh_host = inst['ssh_host']
        ssh_port = inst['ssh_port']
        target = f"root@{ssh_host}"
        
        # 4. Deploy
        # We construct a fake args object to pass to client.deploy
        print("\nProceeding to SLOP deployment...")
        
        # NOTE: Vast instances map a high port (e.g. 12345) to 22.
        # But 'ssh user@host -p port' style is what we need.
        # The deploy script takes 'user@host' and assumes port 22 unless configured in SSH config.
        # BUT, `client.deploy` uses `rsync` and `ssh` commands. 
        # We need to tell `client.deploy` about the port.
        # The current `client.deploy` doesn't support a --port flag explicitly, 
        # it expects the user to handle it or pass it in target?
        # Standard SSH target syntax 'user@host' doesn't support port.
        # We might need to add SSH config entry OR modify deploy.py.
        # 
        # Quick hack: Generate a temporary SSH config file or just use -p in the commands?
        # `deploy.py` constructs commands like `ssh {target} ...`
        # If we pass target="root@ip -p port", it *might* work for ssh but fail for rsync (which needs -e 'ssh -p port').
        
        # Let's modify `deploy.py` to handle custom ports cleanly, OR
        # better yet, since we are wrapping it here, we can generate a local ~/.ssh/config entry for this host.
        # That's cleaner and persistent.
        
        alias = args.name or f"vast-{instance_id}"
        update_ssh_config(alias, ssh_host, ssh_port)
        
        # Now we can deploy to the alias
        deploy_args = argparse.Namespace(
            target=alias,
            path="/root/slop", # Standard Vast location
            name=alias,
            python_cmd="python3", # Explicitly set standard python
            container=None, # We use the docker image directly, not a SIF inside
            build=False, # No SIF build needed
            workers=args.workers,
            unrestricted=True # Vast is rented, use full power
        )
        
        deploy(deploy_args)

        # 5. Install Dependencies
        print("\nInstalling dependencies on remote server...")
        install_dependencies(alias)
        
        print("\n" + "="*60)
        print(f"Deployment Complete! Server alias: {alias}")
        print(f"Price: ${selected_offer['dph']}/hr")
        print(f"To destroy: vastai destroy instance {instance_id}")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        if args.auto_destroy:
            print("Destroying instance due to failure...")
            client.destroy_instance(instance_id)
        sys.exit(1)

def update_ssh_config(alias: str, hostname: str, port: int):
    """Add or update an entry in ~/.ssh/config."""
    config_path = Path.home() / ".ssh" / "config"
    config_path.parent.mkdir(mode=0o700, exist_ok=True)
    
    entry = f"\nHost {alias}\n  HostName {hostname}\n  User root\n  Port {port}\n  StrictHostKeyChecking no\n"
    
    # Simple append for now. A full parser is overkill but better.
    # If it exists, we might duplicate, which SSH handles (first match wins), but it's messy.
    # We'll read the file and check if Host {alias} exists.
    
    content = ""
    if config_path.exists():
        content = config_path.read_text()
        
    if f"Host {alias}" in content:
        print(f"Warning: Host '{alias}' already in SSH config. Please check for conflicts.")
        # We append anyway, assuming the user knows what they are doing or it's a reuse.
        # Actually, let's append at the top? No, append end.
    
    with open(config_path, "a") as f:
        f.write(entry)
        
    print(f"Added alias '{alias}' to ~/.ssh/config")

def main():
    parser = argparse.ArgumentParser(description="Provision Vast.ai GPU for SLOP.")
    parser.add_argument("--list", action="store_true", help="List active instances")
    parser.add_argument("--query", default="gpu_name=RTX_3090 rented=False reliability>0.98", help="Vast.ai search query")
    parser.add_argument("--image", default="pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime", help="Default image if not building")
    parser.add_argument("--image-tag", help="Docker tag to use (or build & push)")
    parser.add_argument("--build-image", action="store_true", help="Build and push Docker image before provisioning")
    parser.add_argument("--disk", type=int, default=40, help="Disk size in GB")
    parser.add_argument("--name", help="Server alias name")
    parser.add_argument("--workers", type=int, default=8, help="Num workers")
    parser.add_argument("--auto-yes", "-y", action="store_true", help="Auto-select first offer")
    parser.add_argument("--auto-destroy", action="store_true", help="Destroy on failure")
    
    args = parser.parse_args()

    if args.list:
        client = VastClient()
        client.list_instances()
        sys.exit(0)

    provision(args)

if __name__ == "__main__":
    main()
