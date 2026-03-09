"""Manage inference providers."""

import argparse
import concurrent.futures
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
import os
import errno

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from client.config import ProviderConfig, registry
from client.provider import remove_remote_files, restart
from client.transport import SSHTransport, TransportError
from shared.protocol.messages import InferenceRequest, JobResult, MessageKind, Request, ServerInfo


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple plain-text table."""
    if not rows:
        return "No data."
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(str(value)))
    fmt = "  ".join([f"{{:<{width}}}" for width in widths])
    lines = [fmt.format(*headers), "  ".join(["-" * width for width in widths])]
    lines.extend(fmt.format(*row) for row in rows)
    return "\n".join(lines)


def check_provider(name: str, config: ProviderConfig, verify_compute: bool = False) -> dict[str, Any]:
    """Return health information for one provider."""
    result = {"name": name, "status": "UNKNOWN", "gpu": "N/A", "latency": "N/A", "compute": "-", "error": None}
    transport = SSHTransport(config)
    try:
        start = time.time()
        # Prevent concurrent connect attempts to the same provider which can
        # spawn duplicate server processes on the remote and cause reconnect
        # churn. Use a simple file lock per-provider.
        with ProviderLock(name, timeout=5.0):
            transport.connect()
        latency_ms = (time.time() - start) * 1000
        response = transport.send_request(Request(kind=MessageKind.SERVER_INFO.value))
        if not isinstance(response, ServerInfo):
            raise TransportError(f"unexpected response: {type(response)}")
        info = response
        result["status"] = "ONLINE"
        result["gpu"] = f"{info.gpu_name} ({info.gpu_memory_mb}MB)"
        result["latency"] = f"{latency_ms:.0f}ms"
        if verify_compute:
            req = InferenceRequest(
                prompt="A sanity check",
                num_steps=1,
                model_id="runwayml/stable-diffusion-v1-5",
                capture_latents=False,
                capture_noise_pred=False,
                capture_prompt_embeds=False,
            )
            started = time.time()
            response = transport.send_request(req)
            if not isinstance(response, JobResult) or response.kind != MessageKind.RESULT.value:
                raise TransportError(f"unexpected response: {type(response)}")
            result["compute"] = f"OK ({time.time() - started:.1f}s)"
    except Exception as exc:
        result["status"] = "OFFLINE"
        result["error"] = str(exc)
        if verify_compute:
            result["compute"] = f"ERROR: {exc}"
    finally:
        transport.close()
    return result


def selected(args: argparse.Namespace) -> list[tuple[str, ProviderConfig]]:
    """Return requested providers."""
    if not args.names:
        providers = registry.list()
        return [(provider.name, provider) for provider in providers]
    out: list[tuple[str, ProviderConfig]] = []
    for name in args.names:
        provider = registry.get(name)
        if provider is not None:
            out.append((name, provider))
    return out


def handle_list(_args: argparse.Namespace) -> None:
    registry.refresh()
    providers = registry.list()
    if not providers:
        print("No providers registered. Use `python -m client.deploy` to add one.")
        return
    rows = []
    for provider in providers:
        runtime = provider.container_image if provider.container_image else provider.python_cmd
        rows.append([provider.name, provider.kind, provider.target, provider.remote_path, runtime])
    print("\n" + format_table(["Name", "Kind", "Target", "Remote Path", "Runtime"], rows) + "\n")


def handle_check(args: argparse.Namespace) -> None:
    registry.refresh()
    providers = selected(args)
    if not providers:
        print("No matching providers found.")
        return
    print(f"\nChecking {len(providers)} providers (Compute Verify: {args.verify})...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(check_provider, name, provider, args.verify) for name, provider in providers]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda item: item["name"])
    rows = [[item["name"], item["status"], item["gpu"], item["latency"], item["compute"]] for item in results]
    print(format_table(["Name", "Status", "GPU", "Latency", "Compute"], rows))
    print("")
    errors = [item for item in results if item["error"]]
    if errors:
        print("Errors:")
        for item in errors:
            print(f"  [{item['name']}]: {item['error']}")


class ProviderLock:
    """Simple file-lock to prevent concurrent SSH connect attempts for the same provider.

    Uses atomic O_EXCL creation of a lock file under ~/.slop/locks/. This keeps
    parallel commands (like check/call in ThreadPool) from spawning duplicate
    SSH sessions which previously caused duplicate server processes and
    confusing reconnect prints.
    """

    def __init__(self, name: str, timeout: float = 10.0):
        self.name = name
        self.timeout = timeout
        self.lock_dir = Path.home() / ".slop" / "locks"
        self.lock_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.lock_dir / f"{name}.lock"
        self.fd = None

    def __enter__(self):
        start = time.time()
        while True:
            try:
                # O_CREAT | O_EXCL ensures atomic create — fails if file exists
                self.fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self.fd, f"pid:{os.getpid()}\n".encode("utf-8"))
                return self
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
                if time.time() - start > self.timeout:
                    # Give up and continue; this avoids a hard hang when a stale
                    # lock persists. Caller should handle potential duplicates.
                    return self
                time.sleep(0.1)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.fd is not None:
                os.close(self.fd)
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass


def handle_remove(args: argparse.Namespace) -> None:
    registry.refresh()
    provider = registry.get(args.name)
    if provider is None:
        print(f"Provider '{args.name}' not found.")
        return
    if args.purge:
        print(f"Purging files for '{args.name}'...")
        try:
            remove_remote_files(provider)
            print("Remote files deleted.")
        except subprocess.CalledProcessError as exc:
            print(f"Warning: Failed to delete remote files: {exc}")
    registry.remove(args.name)
    print(f"Removed '{args.name}' from registry.")


def handle_restart(args: argparse.Namespace) -> None:
    registry.refresh()
    providers = selected(args)
    if not providers:
        print("No matching providers found.")
        return
    for name, provider in providers:
        restart(provider)
        print(f"Restarted '{name}'.")


def handle_clearmem(args: argparse.Namespace) -> None:
    registry.refresh()
    providers = selected(args)
    if not providers:
        print("No matching providers found.")
        return
    for name, provider in providers:
        from client.interface import SlopClient
        try:
            with SlopClient(provider) as client:
                # Use a short timeout so a hanging provider doesn't block the whole command
                stats = client.cleanup(clear_model=args.free_model, timeout_s=30.0)
                print(f"[{name}] Cleanup: {stats}")
        except Exception as e:
            print(f"[{name}] Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage inference providers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List registered providers")

    check_parser = subparsers.add_parser("check", help="Check provider health")
    check_parser.add_argument("names", nargs="*", help="Provider names to check (default: all)")
    check_parser.add_argument("--verify", "-v", action="store_true", help="Run a quick compute verification job")

    remove_parser = subparsers.add_parser("remove", help="Remove provider from registry and optionally delete files")
    remove_parser.add_argument("name", help="Provider name")
    remove_parser.add_argument("--purge", action="store_true", default=True, help="Delete provider files (default: True)")
    remove_parser.add_argument("--no-purge", action="store_false", dest="purge", help="Keep provider files")

    restart_parser = subparsers.add_parser("restart", help="Kill daemon processes for providers")
    restart_parser.add_argument("names", nargs="*", help="Provider names to restart (default: all)")

    clearmem_parser = subparsers.add_parser("clearmem", help="Trigger memory cleanup on providers")
    clearmem_parser.add_argument("names", nargs="*", help="Provider names (default: all)")
    clearmem_parser.add_argument("--free-model", action="store_true", help="Also unload the model from VRAM")

    args = parser.parse_args()
    if args.command == "list":
        handle_list(args)
    elif args.command == "check":
        handle_check(args)
    elif args.command == "remove":
        handle_remove(args)
    elif args.command == "restart":
        handle_restart(args)
    elif args.command == "clearmem":
        handle_clearmem(args)


if __name__ == "__main__":
    main()
