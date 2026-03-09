"""Vast.ai provisioning and redeploy helpers."""

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, cast

from client.config import ProviderConfig, Registry
from client.deploy import deploy
from client.interface import SlopClient


class VastError(Exception):
    pass


class VastClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("VAST_API_KEY") or os.environ.get("VASTAI_API_KEY")

    def _bin(self) -> str:
        from shutil import which

        executable = which("vastai")
        if executable:
            return executable
        raise VastError("vastai CLI not found")

    def run(self, *args: str) -> Any:
        cmd = [self._bin(), *args, "--raw"]
        if self.api_key:
            cmd.extend(["--api-key", self.api_key])
        try:
            output = subprocess.check_output(cmd, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as exc:
            raise VastError(exc.stderr.decode().strip()) from exc
        return json.loads(output)

    def offers(self, query: str, limit: int) -> list[dict]:
        return self.run("search", "offers", query, "-o", "dph")[:limit]

    def create(self, offer_id: int, image: str, disk: int) -> int:
        raw = self.run(
            "create",
            "instance",
            str(offer_id),
            "--image",
            image,
            "--disk",
            str(disk),
            "--ssh",
            "--direct",
            "--onstart-cmd",
            "apt-get update; apt-get install -y openssh-server; mkdir -p /var/run/sshd; /usr/sbin/sshd -D",
        )
        if isinstance(raw, list):
            if not raw:
                raise VastError("empty response from vastai")
            result = cast(dict, raw[0])
        else:
            result = cast(dict, raw)
        if result.get("success"):
            return result["new_contract"]
        raise VastError(str(result))

    def instances(self) -> list[dict]:
        return self.run("show", "instances")

    def destroy(self, instance_id: int) -> None:
        self.run("destroy", "instance", str(instance_id))


def update_ssh_config(alias: str, hostname: str, port: int) -> None:
    config_path = Path.home() / ".ssh" / "config"
    config_path.parent.mkdir(mode=0o700, exist_ok=True)
    entry = f"\nHost {alias}\n  HostName {hostname}\n  User root\n  Port {port}\n  StrictHostKeyChecking no\n"
    content = config_path.read_text() if config_path.exists() else ""
    if f"Host {alias}\n" not in content:
        with open(config_path, "a", encoding="utf-8") as handle:
            handle.write(entry)


def register_instance(alias: str, remote_path: str = "/root/slop", python_cmd: str = "python3") -> None:
    """Store a Vast SSH endpoint in the provider registry."""
    registry = Registry()
    registry.providers[alias] = ProviderConfig(name=alias, kind="ssh", target=alias, remote_path=remote_path, python_cmd=python_cmd)
    registry.save()


def wait_for_port(host: str, port: int, timeout: int = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError:
            time.sleep(2)
    raise VastError(f"timeout waiting for {host}:{port}")


def wait_for_instance(client: VastClient, instance_id: int, timeout: int = 420) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        for instance in client.instances():
            if instance["id"] != instance_id:
                continue
            if instance.get("actual_status") == "running" and instance.get("ssh_host") and instance.get("ssh_port"):
                wait_for_port(instance["ssh_host"], instance["ssh_port"])
                return instance
        time.sleep(10)
    raise VastError(f"instance {instance_id} did not become ready")


def install_dependencies(alias: str) -> None:
    reg = Registry()
    cfg = reg.get(alias)
    if cfg is None:
        raise VastError(f"unknown provider: {alias}")
    req_path = shlex.quote(f"{cfg.remote_path}/requirements.txt")
    subprocess.check_call(["ssh", alias, f"pip install -r {req_path}"])


def measure_compute(alias: str) -> str:
    reg = Registry()
    cfg = reg.get(alias)
    if cfg is None:
        cfg = ProviderConfig(name=alias, kind="ssh", target=alias, remote_path="/root/slop")
    try:
        with SlopClient(cfg) as client:
            start = time.time()
            result = client.sample(prompt="compute check", num_steps=2)
            client.render(result.points[-1])
            return f"OK ({time.time() - start:.2f}s)"
    except Exception as exc:
        return f"ERROR ({str(exc)[:40]})"


def redeploy(alias: str) -> None:
    reg = Registry()
    cfg = reg.get(alias)
    if cfg is None:
        raise VastError(f"provider not found: {alias}")
    deploy(
        argparse.Namespace(
            target=cfg.target,
            path=cfg.remote_path,
            name=cfg.name,
            python_cmd=cfg.python_cmd,
            container=cfg.container_image,
            build=False,
            workers=cfg.num_workers,
            unrestricted=True,
        )
    )
    install_dependencies(alias)


def sync_instances() -> None:
    """Sync running vastai instances to the registry."""
    from client.config import PROVIDERS_FILE
    import json

    client = VastClient()
    try:
        instances = client.instances()
    except VastError as e:
        print(f"Error fetching instances: {e}")
        return

    existing = []
    if PROVIDERS_FILE.exists():
        try:
            existing = json.loads(PROVIDERS_FILE.read_text())
        except json.JSONDecodeError:
            existing = []

    existing_names = {p["name"] for p in existing}
    count = 0

    for inst in instances:
        if inst.get("actual_status") != "running":
            continue
        ssh_host = inst.get("ssh_host")
        ssh_port = inst.get("ssh_port")
        if not ssh_host or not ssh_port:
            continue

        alias = f"vast-{inst['id']}"
        update_ssh_config(alias, ssh_host, ssh_port)

        if alias not in existing_names:
            new_config = {
                "name": alias,
                "kind": "ssh",
                "target": alias,
                "remote_path": "/root/slop",
                "python_cmd": "python3",
                "container_image": None,
                "num_workers": 8,
            }
            existing.append(new_config)
            print(f"Registered: {alias}")
            count += 1
        else:
            print(f"Already registered: {alias}")

    PROVIDERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROVIDERS_FILE.write_text(json.dumps(existing, indent=2))
    print(f"\nSynced {count} new instance(s)")


def provision(args: argparse.Namespace) -> None:
    client = VastClient()
    offers = client.offers(args.query, args.limit)
    if not offers:
        raise VastError("no offers")
    offer = offers[0]
    instance_id = client.create(offer["id"], args.image, args.disk)
    instance = wait_for_instance(client, instance_id)
    alias = args.name or f"vast-{instance_id}"
    update_ssh_config(alias, instance["ssh_host"], instance["ssh_port"])
    register_instance(alias)
    deploy(
        argparse.Namespace(
            target=alias,
            path="/root/slop",
            name=alias,
            python_cmd="python3",
            container=None,
            build=False,
            workers=args.workers,
            unrestricted=True,
        )
    )
    install_dependencies(alias)
    print(alias)


def main() -> None:
    parser = argparse.ArgumentParser(description="Vast.ai helpers")
    parser.add_argument("--query", default="gpu_name=RTX_3090 rented=False reliability>0.98")
    parser.add_argument("--image", default="pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime")
    parser.add_argument("--disk", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--name")
    parser.add_argument("--provision", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--redeploy")
    parser.add_argument("--measure")
    parser.add_argument("--register")
    parser.add_argument("--sync", action="store_true", help="Sync running instances to registry")
    args = parser.parse_args()

    client = VastClient()
    if args.list:
        print(json.dumps(client.instances(), indent=2))
        return
    if args.redeploy:
        redeploy(args.redeploy)
        return
    if args.measure:
        print(measure_compute(args.measure))
        return
    if args.register:
        register_instance(args.register)
        return
    if args.sync:
        sync_instances()
        return
    if args.provision:
        provision(args)
        return
    parser.error("choose an action")


if __name__ == "__main__":
    main()
