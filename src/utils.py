import json
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


def save_or_update_json(data: Dict[str, Any], path: Path) -> None:
    """
    Save a dictionary to JSON, updating existing keys if file already exists.
    Top-level keys are merged.
    """
    if path.exists():
        with path.open("r") as f:
            existing = json.load(f)
    else:
        existing = {}

    existing.update(data)

    with path.open("w") as f:
        json.dump(existing, f, indent=2)
