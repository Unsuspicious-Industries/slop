import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union, cast


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def save_json(obj: Dict[str, Any], path: Union[str, Path]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return cast(Dict[str, Any], json.load(f))


def save_list(lines: List[str], path: Union[str, Path]) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def load_lines(path: Union[str, Path]) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]
