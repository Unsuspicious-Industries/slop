import yaml
from typing import Any, Dict, cast


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], yaml.safe_load(f))
