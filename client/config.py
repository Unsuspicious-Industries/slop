"""Client configuration and server registry management."""
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

CONFIG_DIR = Path.home() / ".slop"
REGISTRY_FILE = CONFIG_DIR / "servers.json"

@dataclass
class ServerConfig:
    name: str
    host: str               # user@hostname
    remote_path: str        # /path/to/slop/on/remote
    python_cmd: str = "python"
    container_image: Optional[str] = None # Path to .sif file on remote
    num_workers: int = 4    # Number of CPU threads (OMP_NUM_THREADS)
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d):
        return cls(**d)

class Registry:
    def __init__(self):
        self.servers: Dict[str, ServerConfig] = {}
        self.load()
        
    def load(self):
        if not REGISTRY_FILE.exists():
            return
            
        try:
            with open(REGISTRY_FILE, "r") as f:
                data = json.load(f)
                for name, cfg in data.items():
                    self.servers[name] = ServerConfig.from_dict(cfg)
        except Exception as e:
            print(f"Warning: Failed to load registry: {e}")
            
    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_FILE, "w") as f:
            data = {name: cfg.to_dict() for name, cfg in self.servers.items()}
            json.dump(data, f, indent=2)
            
    def add(self, config: ServerConfig):
        self.servers[config.name] = config
        self.save()
        
    def get(self, name: str) -> Optional[ServerConfig]:
        return self.servers.get(name)
        
    def remove(self, name: str):
        if name in self.servers:
            del self.servers[name]
            self.save()
            
    def list(self) -> List[ServerConfig]:
        return list(self.servers.values())

# Global singleton
registry = Registry()
