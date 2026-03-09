#!/usr/bin/env python3
"""Remote autonomous training script using SlopClient.

This script connects to a remote server via SlopClient and runs training
on the collected teacher images.

Usage:
    python distill/train_remote.py --provider vast-32582479 \
        --manifest /root/slop/distill/data/test/grok_test/manifest.jsonl \
        --output /root/slop/distill/output
"""

import argparse
import sys
from pathlib import Path

# Add local project to path
local_root = Path(__file__).parent.parent.resolve()
if str(local_root) not in sys.path:
    sys.path.insert(0, str(local_root))

from client.config import registry
from client.interface import SlopClient


def main():
    parser = argparse.ArgumentParser(description="Run remote training via SlopClient")
    parser.add_argument("--provider", "-p", default=None, help="Provider name (default: first available)")
    parser.add_argument("--manifest", "-m", required=True, help="Path to manifest.jsonl on REMOTE server")
    parser.add_argument("--output", "-o", required=True, help="Output directory on REMOTE server")
    parser.add_argument("--job-id", default=None, help="Attach to an existing job id instead of starting")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", help="Base model ID")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout in seconds")
    args = parser.parse_args()

    # Get provider
    registry.refresh()
    providers = registry.list()
    
    if not providers:
        print("ERROR: No providers found in registry")
        print("Add a provider first with: python -m client.vastai --sync")
        sys.exit(1)
    
    if args.provider:
        provider = registry.get(args.provider)
        if not provider:
            print(f"ERROR: Provider '{args.provider}' not found")
            print(f"Available: {[p.name for p in providers]}")
            sys.exit(1)
    else:
        provider = providers[0]
    
    print(f"Using provider: {provider.name}")
    print(f"  Target: {provider.target}")
    print(f"  Remote path: {provider.remote_path}")
    print()
    print(f"Training config:")
    print(f"  Manifest: {args.manifest}")
    print(f"  Output: {args.output}")
    print(f"  Model: {args.model_id}")
    print(f"  Batch: {args.batch_size}, Epochs: {args.epochs}, LR: {args.lr}")
    print()

    # Connect and start training
    with SlopClient(provider) as client:
        # Check server info
        info = client.info()
        print(f"Connected to: {info.hostname}")
        print(f"GPU: {info.gpu_name} ({info.gpu_memory_mb}MB)")
        print()
        
        # Reuse an existing running job if present
        job_id = args.job_id
        if not job_id:
            running_states = {"starting", "running", "stopping"}
            jobs = client.list_jobs(limit=200)
            matches = [
                j
                for j in jobs
                if j.get("kind") == "train"
                and j.get("output_dir") == args.output
                and j.get("state") in running_states
            ]
            if matches:
                matches.sort(key=lambda j: float(j.get("updated_at") or 0.0), reverse=True)
                job_id = matches[0]["job_id"]
                print(f"Attaching to existing job: {job_id}")

        if not job_id:
            print("Starting autonomous training job...")
            start = client.train(
                manifest_path=args.manifest,
                output_dir=args.output,
                model_id=args.model_id,
                batch_size=args.batch_size,
                epochs=args.epochs,
                learning_rate=args.lr,
                save_every=args.save_every,
                timeout_s=60,
            )
            job_id = str(start.metadata.get("job_id"))
            print(f"Job: {job_id}")
            print(f"Job dir: {start.metadata.get('job_dir')}")

        # Poll progress until completion (or timeout)
        import time
        deadline = time.time() + args.timeout
        since = 0
        last_state = None
        while time.time() < deadline:
            state = client.attach_job(str(job_id), since_line=since, max_lines=200)
            status = state.get("status", {})
            for ev in state.get("progress", []):
                try:
                    loss = float(ev.get("loss", 0.0))
                except Exception:
                    loss = 0.0
                print(f"[progress] ep={ev.get('epoch')} step={ev.get('step')} loss={loss:.4f} {ev.get('message','')}")
            since = int(state.get("next_since_line", since))
            st = status.get("state")
            if st and st != last_state:
                print(f"[status] {st}")
                last_state = st
            if st in {"completed", "failed", "killed"}:
                print("Final status:", status)
                if st != "completed":
                    print("Log tail:\n", state.get("log_tail", ""))
                return
            time.sleep(2)

        print("Timeout waiting for job. You can re-attach later with:")
        print(f"  python distill/train_remote.py -p {provider.name} -m {args.manifest} -o {args.output} --job-id {job_id} --timeout 7200")


if __name__ == "__main__":
    main()
