"""Management tool for SLOP servers: list, health-check, and verify compute."""
import argparse
import sys
import time
import subprocess
import concurrent.futures
from typing import Dict, Any, List, cast
from pathlib import Path

# Fix path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.config import registry, ServerConfig
from client.transport import SSHTransport, TransportError
from shared.protocol.messages import Request, InferenceRequest, MessageKind, JobResult, ServerInfo, Response

def format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Simple table formatter to avoid external dependencies."""
    if not rows:
        return "No data."
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
            
    # Create format string
    fmt = "  ".join([f"{{:<{w}}}" for w in widths])
    
    lines = []
    lines.append(fmt.format(*headers))
    lines.append("  ".join(["-" * w for w in widths]))
    for row in rows:
        lines.append(fmt.format(*row))
        
    return "\n".join(lines)

def check_server(name: str, config: ServerConfig, verify_compute: bool = False) -> Dict[str, Any]:
    """Run health checks on a single server."""
    result = {
        "name": name,
        "host": config.host,
        "status": "UNKNOWN",
        "gpu": "N/A",
        "latency": "N/A",
        "compute": "SKIPPED",
        "error": None
    }
    
    transport = SSHTransport(config)
    try:
        # 1. Connect & Ping
        start = time.time()
        transport.connect()
        connect_time = (time.time() - start) * 1000
        
        # 2. Get Info
        resp = transport.send_request(Request(kind=MessageKind.SERVER_INFO))
        if isinstance(resp, ServerInfo):
            info = resp
        else:
             # Fallback or error if protocol mismatch, though transport usually handles conversion based on kind
             # Use cast if we are confident, or checks.
             info = cast(ServerInfo, resp)

        result["status"] = "ONLINE"
        result["gpu"] = f"{info.gpu_name} ({info.gpu_memory_mb}MB)"
        result["latency"] = f"{connect_time:.0f}ms"
        
        # 3. Verify Compute (Optional)
        if verify_compute:
            try:
                # Small fast generation
                # Use a known-good model for verification (SD v1.5)
                # to avoid auth/gating issues during simple checks.
                req = InferenceRequest(
                    prompt="A sanity check",
                    num_steps=1,
                    model_id="runwayml/stable-diffusion-v1-5",
                    height=512,
                    width=512,
                    capture_latents=False,
                    capture_noise_pred=False,
                    capture_prompt_embeds=False
                )
                job_start = time.time()
                resp = transport.send_request(req)
                res = cast(JobResult, resp)
                job_time = time.time() - job_start
                
                if res.kind == MessageKind.RESULT:                    result["compute"] = f"OK ({job_time:.1f}s)"
                else:
                    result["compute"] = f"FAIL ({res.kind})"
            except Exception as e:
                result["compute"] = f"ERROR: {e}"
                
    except TransportError as e:
        result["status"] = "OFFLINE"
        result["error"] = str(e)
    except Exception as e:
        result["status"] = "ERROR"
        result["error"] = str(e)
    finally:
        transport.close()
        
    return result

def handle_list(args):
    servers = registry.list()
    if not servers:
        print("No servers registered. Use 'python -m client.deploy' to add one.")
        return

    headers = ["Name", "Host", "Remote Path", "Python/Container"]
    rows = []
    for s in servers:
        exec_env = s.container_image if s.container_image else s.python_cmd
        rows.append([s.name, s.host, s.remote_path, exec_env])
    print("\n" + format_table(headers, rows) + "\n")

def handle_check(args):
    names = args.names
    if not names:
        # Check all
        servers = registry.servers.items()
    else:
        # Check specific
        servers = []
        for n in names:
            cfg = registry.get(n)
            if cfg:
                servers.append((n, cfg))
        
    if not servers:
        print("No matching servers found.")
        return

    print(f"\nChecking {len(servers)} servers (Compute Verify: {args.verify})...\n")
    
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(check_server, name, cfg, args.verify): name for name, cfg in servers}
        
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            
    # Sort by name
    results.sort(key=lambda x: x["name"])
    
    # Format output
    headers = ["Name", "Status", "GPU", "Latency", "Compute"]
    rows = []
    for r in results:
        status = r["status"]
        if r["error"]:
            status += f" (!)"
            
        rows.append([
            r["name"],
            status,
            r["gpu"],
            r["latency"],
            r["compute"] if args.verify else "-"
        ])
        
    print(format_table(headers, rows))
    print("")
    
    # Print errors if any
    errors = [r for r in results if r["error"]]
    if errors:
        print("Errors:")
        for r in errors:
            print(f"  [{r['name']}]: {r['error']}")

def handle_remove(args):
    config = registry.get(args.name)
    if not config:
        print(f"Server '{args.name}' not found.")
        return

    # Delete remote files if requested or if we decide to force it (per user instruction)
    # "when running remove, it should delete the remote files" -> implies default behavior
    if args.purge:
        print(f"Purging remote files at {config.host}:{config.remote_path}...")
        try:
            if config.host == "local":
                subprocess.check_call(f"rm -rf {config.remote_path}", shell=True)
            else:
                subprocess.check_call(["ssh", config.host, f"rm -rf {config.remote_path}"])
            print("Remote files deleted.")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to delete remote files: {e}")

    registry.remove(args.name)
    print(f"Removed '{args.name}' from registry.")

def main():
    parser = argparse.ArgumentParser(description="Manage SLOP servers.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # List
    subparsers.add_parser("list", help="List registered servers")
    
    # Check / Health
    check_parser = subparsers.add_parser("check", help="Check server health")
    check_parser.add_argument("names", nargs="*", help="Specific server names to check (default: all)")
    check_parser.add_argument("--verify", "-v", action="store_true", help="Run a quick compute verification job")
    
    # Remove
    rm_parser = subparsers.add_parser("remove", help="Remove server from registry and delete remote files")
    rm_parser.add_argument("name", help="Name of server to remove")
    rm_parser.add_argument("--purge", action="store_true", default=True, help="Delete remote files (default: True)")
    rm_parser.add_argument("--no-purge", action="store_false", dest="purge", help="Keep remote files")
    
    args = parser.parse_args()
    
    if args.command == "list":
        handle_list(args)
    elif args.command == "check":
        handle_check(args)
    elif args.command == "remove":
        handle_remove(args)

if __name__ == "__main__":
    main()
