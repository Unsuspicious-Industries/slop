"""Inference provider registry."""

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


CONFIG_DIR = Path.home() / ".slop"
PROVIDERS_FILE = CONFIG_DIR / "providers.json"
LEGACY_FILE = CONFIG_DIR / "servers.json"


@dataclass
class ProviderConfig:
    name: str
    kind: str
    target: str
    remote_path: str
    python_cmd: str = "python"
    container_image: Optional[str] = None
    num_workers: int = 4

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderConfig":
        return cls(**data)


def _is_reachable(config: ProviderConfig) -> bool:
    """Return True when a provider target responds to a minimal check."""
    if config.kind == "local":
        return Path(config.remote_path).exists()
    if config.kind == "ssh":
        try:
            result = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "ConnectTimeout=5",
                    "-o",
                    "BatchMode=yes",
                    config.target,
                    "true",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            return result.returncode == 0
        except Exception:
            return False
    return True


class Registry:
    def __init__(self) -> None:
        self.providers: dict[str, ProviderConfig] = {}
        self.load()

    def _read_entries(self) -> list[dict]:
        if PROVIDERS_FILE.exists():
            with open(PROVIDERS_FILE, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, list) else []

        if LEGACY_FILE.exists():
            with open(LEGACY_FILE, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, dict):
                return [
                    {
                        "name": name,
                        "kind": "local" if cfg.get("host") == "local" else "ssh",
                        "target": cfg.get("host", "local"),
                        "remote_path": cfg["remote_path"],
                        "python_cmd": cfg.get("python_cmd", "python"),
                        "container_image": cfg.get("container_image"),
                        "num_workers": cfg.get("num_workers", 4),
                    }
                    for name, cfg in data.items()
                ]
        return []

    def load(self) -> None:
        self.providers = {}
        entries = self._read_entries()
        for entry in entries:
            config = ProviderConfig.from_dict(entry)
            self.providers[config.name] = config

    def refresh(self) -> None:
        """Reload providers from disk and drop unreachable entries."""
        self.load()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        items = [self.providers[name].to_dict() for name in sorted(self.providers)]
        with open(PROVIDERS_FILE, "w", encoding="utf-8") as handle:
            json.dump(items, handle, indent=2)

    def add(self, config: ProviderConfig) -> None:
        self.providers[config.name] = config
        self.save()

    def get(self, name: str) -> Optional[ProviderConfig]:
        return self.providers.get(name)

    def remove(self, name: str) -> None:
        if name in self.providers:
            del self.providers[name]
            self.save()

    def list(self) -> list[ProviderConfig]:
        return [self.providers[name] for name in sorted(self.providers)]



registry = Registry()
