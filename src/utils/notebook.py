"""Notebook helpers: flexible loading without strict package install.

Usage in a notebook:
```
from pathlib import Path
from src.utils.notebook import bootstrap
bootstrap()

from src.visualization.interface_utils import load_embeddings
```
"""

import sys
from pathlib import Path
from typing import Optional


def bootstrap(project_root: Optional[str] = None) -> None:
    root = Path(project_root) if project_root else Path.cwd()
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
