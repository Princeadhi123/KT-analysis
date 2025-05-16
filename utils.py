"""Utility functions for Knowledge Tracing (refactored)."""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy types."""
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def load_skill_mapping(path: str) -> Dict[int, int]:
    """Load skill mapping from a JSON file as {int: int}."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Skill mapping file not found: {path}")
    with p.open('r') as f:
        mapping = json.load(f)
    return {int(k): int(v) for k, v in mapping.items()}

def save_json(data: dict, path: str) -> None:
    """Save data to a JSON file using NumpyEncoder."""
    p = Path(path)
    with p.open('w') as f:
        json.dump(data, f, indent=2, cls=NumpyEncoder)

def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducibility (NumPy and Python)."""
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
