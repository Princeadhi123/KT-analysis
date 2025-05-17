"""Configuration for knowledge tracing using type-safe dataclasses."""

from dataclasses import dataclass
from pathlib import Path



@dataclass
class ModelConfig:
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.2

@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_seq_len: int = 50
    learning_rate: float = 0.001
    max_epochs: int = 10
    val_split: float = 0.2
    num_workers: int = 4

@dataclass
class Paths:
    project_root: Path = Path(__file__).parent
    data_dir: Path = project_root / "data"
    data: Path = data_dir / "processed_data.csv"
    checkpoint_dir: Path = project_root / "checkpoints"
    best_model_path: Path = checkpoint_dir / "best_model.ckpt"
    skill_mapping_path: Path = checkpoint_dir / "skill_mapping.json"

    def __post_init__(self) -> None:
        """Ensure necessary directories exist."""
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

# Create global instances
MODEL_CONFIG = ModelConfig()
TRAINING_CONFIG = TrainingConfig()
PATHS = Paths()
