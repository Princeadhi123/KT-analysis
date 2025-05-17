"""Configuration management for the Knowledge Tracing system.

This module provides type-safe configuration using Python dataclasses,
with clear separation of concerns between model architecture, training,
prediction, and file system paths.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the DKT model architecture.
    
    Attributes:
        hidden_dim: Size of the hidden layers in the LSTM.
        num_layers: Number of LSTM layers.
        dropout: Dropout probability for regularization.
    """
    hidden_dim: int = 64
    num_layers: int = 1
    dropout: float = 0.2


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        batch_size: Number of sequences per training batch.
        max_seq_len: Maximum sequence length (longer sequences are truncated).
        learning_rate: Initial learning rate for the optimizer.
        max_epochs: Maximum number of training epochs.
        val_split: Fraction of data to use for validation.
        num_workers: Number of worker processes for data loading.
    """
    batch_size: int = 32
    max_seq_len: int = 50
    learning_rate: float = 0.001
    max_epochs: int = 10
    val_split: float = 0.2
    num_workers: int = 4


@dataclass(frozen=True)
class PredictionConfig:
    """Configuration for model predictions.
    
    Attributes:
        model_checkpoint: Filename of the saved model checkpoint.
        output_csv: Default filename for prediction outputs.
    """
    model_checkpoint: str = "best_model.ckpt"
    output_csv: str = "test_predictions.csv"


class Paths:
    """Filesystem paths used throughout the project.
    
    This class manages all file and directory paths in a centralized way,
    ensuring consistency and making it easy to modify the project structure.
    """
    
    def __init__(self) -> None:
        """Initialize all paths relative to the project root."""
        self.project_root: Path = Path(__file__).parent.absolute()
        
        # Data directories
        self.data_dir: Path = self.project_root / "data"
        self.raw_data: Path = self.data_dir / "raw"
        self.processed_data: Path = self.data_dir / "processed"
        
        # Model artifacts
        self.checkpoint_dir: Path = self.project_root / "checkpoints"
        self.logs_dir: Path = self.project_root / "logs"
        self.predictions_dir: Path = self.project_root / "predictions"
        
        # Individual files
        self.training_data: Path = self.processed_data / "train_data.csv"
        self.test_data: Path = self.processed_data / "test_data.csv"
        self.skill_mapping: Path = self.checkpoint_dir / "skill_mapping.json"
        self.model_checkpoint: Path = self.checkpoint_dir / PREDICTION_CONFIG.model_checkpoint
        
        # Ensure all directories exist
        self._create_directories()
    
    def _create_directories(self) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            self.data_dir,
            self.raw_data,
            self.processed_data,
            self.checkpoint_dir,
            self.logs_dir,
            self.predictions_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")


# Global configuration instances
MODEL_CONFIG: ModelConfig = ModelConfig()
TRAINING_CONFIG: TrainingConfig = TrainingConfig()
PREDICTION_CONFIG: PredictionConfig = PredictionConfig()
PATHS: Paths = Paths()

# Log configuration
logger.info("Configuration loaded")
logger.debug(f"Project root: {PATHS.project_root}")
logger.debug(f"Model configuration: {MODEL_CONFIG}")
logger.debug(f"Training configuration: {TRAINING_CONFIG}")
