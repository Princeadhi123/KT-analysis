"""Main script for training the knowledge tracing model."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytorch_lightning as pl

from config import MODEL_CONFIG, TRAINING_CONFIG, PATHS
from data import KTDataModule
from models import DKT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def count_unique_skills(data_path: Path) -> int:
    """Count number of unique skills in the dataset.
    
    Args:
        data_path: Path to the dataset CSV file.
        
    Returns:
        Number of unique skills.
    """
    try:
        df = pd.read_csv(data_path)
        return df['skill_id'].nunique()
    except Exception as e:
        logger.error(f"Error counting unique skills: {e}")
        raise

def load_hyperparameters(hparam_path: str = "best_hyperparameters.json") -> Dict[str, Any]:
    """Load hyperparameters from JSON file if it exists.
    
    Args:
        hparam_path: Path to hyperparameters JSON file.
        
    Returns:
        Dictionary of hyperparameters.
    """
    if os.path.exists(hparam_path):
        try:
            with open(hparam_path, "r") as f:
                params = json.load(f)
            logger.info(f"Loaded hyperparameters from {hparam_path}")
            return params
        except Exception as e:
            logger.warning(f"Error loading {hparam_path}: {e}")
    return {}

def train_model() -> None:
    """Train the knowledge tracing model."""
    try:
        # Initialize data module
        logger.info("Initializing data module")
        data_module = KTDataModule(
            data_path=str(PATHS.data),
            batch_size=TRAINING_CONFIG.batch_size,
            max_seq_len=TRAINING_CONFIG.max_seq_len,
            val_split=TRAINING_CONFIG.val_split
        )
        
        # Count unique skills
        num_skills = count_unique_skills(PATHS.data)
        logger.info(f"Number of unique skills: {num_skills}")

        # Load hyperparameters
        best_hparams = load_hyperparameters()
        
        # Helper function to get hyperparameters with fallback
        def get_param(param_name: str, default: Any) -> Any:
            return best_hparams.get(param_name, default)

        # Initialize model
        logger.info("Initializing model")
        model = DKT(
            num_skills=num_skills,
            hidden_dim=get_param("hidden_dim", MODEL_CONFIG.hidden_dim),
            num_layers=get_param("num_layers", MODEL_CONFIG.num_layers),
            dropout=get_param("dropout", MODEL_CONFIG.dropout),
            learning_rate=get_param("learning_rate", TRAINING_CONFIG.learning_rate)
        )

        # Configure callbacks
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=PATHS.checkpoint_dir,
            filename='best_model',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            verbose=True
        )

        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=TRAINING_CONFIG.max_epochs,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback],
            logger=pl.loggers.CSVLogger(save_dir=str(PATHS.checkpoint_dir))
        )

        # Train model
        logger.info("Starting training...")
        trainer.fit(model, data_module)
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    train_model()
