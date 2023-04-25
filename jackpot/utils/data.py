import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)
