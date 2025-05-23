"""Main script for training the knowledge tracing model."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Tuple

import pandas as pd
import numpy as np # Added for loading PAF embeddings
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

def get_paf_output_dimensions(paf_embeddings_path: str) -> Tuple[int, int]:
    """Load PAF embeddings and return num_outputs and embedding_dim."""
    try:
        embeddings = np.load(paf_embeddings_path)
        num_outputs = embeddings.shape[0]  # Number of unique skills with embeddings
        embedding_dim = embeddings.shape[1] # Number of factors
        logger.info(f"Loaded PAF embeddings from {paf_embeddings_path}. Num outputs: {num_outputs}, Embedding dim: {embedding_dim}")
        return num_outputs, embedding_dim
    except FileNotFoundError:
        logger.error(f"PAF embeddings file not found at {paf_embeddings_path}. Ensure preprocessing was run.")
        raise
    except Exception as e:
        logger.error(f"Error loading PAF embeddings: {e}")
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
        
        # Get PAF embedding dimensions
        # Assuming PATHS.base_checkpoint_dir or similar is 'checkpoints/'
        # If PATHS.checkpoint_dir is specific like 'checkpoints/models', adjust base path.
        # For this task, we'll construct it from a presumed base 'checkpoints' dir.
        # A more robust solution would use a specific config entry in PATHS for paf_outputs_dir.
        
        # Construct path to PAF embeddings file
        # If PATHS.checkpoint_dir is already 'checkpoints', this is fine.
        # If it's 'checkpoints/some_model_specific_dir', we need a more general base path.
        # Let's assume PATHS.checkpoint_dir is the base 'checkpoints' directory for now as per instructions.
        if not hasattr(PATHS, 'checkpoint_dir') or not PATHS.checkpoint_dir:
            # Fallback if PATHS.checkpoint_dir is not defined or empty, though it should be from config.py
            logger.warning("PATHS.checkpoint_dir not defined in config.py, using default 'checkpoints/'.")
            base_checkpoint_dir = Path("checkpoints")
        else:
            # If PATHS.checkpoint_dir might be "checkpoints/specific_model_checkpoints"
            # we should ideally use a PATHS.paf_output_dir or PATHS.preprocessing_output_dir
            # For now, let's assume PATHS.checkpoint_dir is the root for outputs like 'checkpoints/'
            # Or, if it's more specific, that paf_outputs is relative to its parent.
            # Given the problem description, it's likely 'checkpoints/' is the intended base.
            # If config.py has PATHS.checkpoint_dir = "checkpoints" (or "checkpoints/") this is fine.
            base_checkpoint_dir = Path(PATHS.checkpoint_dir) # Path("checkpoints")
            
        paf_embeddings_file = base_checkpoint_dir / "paf_outputs" / "skill_factor_embeddings.npy"
        
        logger.info(f"Attempting to load PAF embeddings from: {paf_embeddings_file}")
        num_outputs, embedding_dim = get_paf_output_dimensions(str(paf_embeddings_file))
        logger.info(f"DKT model will be initialized with {num_outputs} output units and {embedding_dim} embedding dimensions (factors).")

        # Load hyperparameters
        best_hparams = load_hyperparameters()
        
        # Helper function to get hyperparameters with fallback
        def get_param(param_name: str, default: Any) -> Any:
            return best_hparams.get(param_name, default)

        # Initialize model
        logger.info("Initializing model")
        model = DKT(
            num_outputs=num_outputs,
            embedding_dim=embedding_dim,
            skill_embeddings_path=str(paf_embeddings_file),
            hidden_dim=get_param("hidden_dim", MODEL_CONFIG.hidden_dim),
            num_layers=get_param("num_layers", MODEL_CONFIG.num_layers),
            dropout=get_param("dropout", MODEL_CONFIG.dropout),
            learning_rate=get_param("learning_rate", TRAINING_CONFIG.learning_rate)
        )

        # Configure callbacks
        # Ensure dirpath for ModelCheckpoint is correctly pointing to where models should be saved.
        # If base_checkpoint_dir was used for paf_embeddings_file, ensure it's also appropriate here,
        # or that PATHS.checkpoint_dir is specifically for model files.
        # Typically, PATHS.checkpoint_dir from config.py would be 'checkpoints/models/' or similar.
        # The problem implies PATHS.checkpoint_dir is the general one.
        
        model_save_dir = base_checkpoint_dir / "model_checkpoints" # Example: checkpoints/model_checkpoints
        model_save_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
        logger.info(f"Model checkpoints will be saved in: {model_save_dir}")

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=str(model_save_dir), 
            filename='best_model',
            save_top_k=1,
            monitor='val_loss',
            mode='min',
            verbose=True
        )
        
        # Logger save directory
        # Logs should also go into a structured place, perhaps parallel to model_checkpoints
        logs_save_dir = base_checkpoint_dir / "training_logs" # Example: checkpoints/training_logs
        logs_save_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Training logs will be saved in: {logs_save_dir}")


        # Initialize trainer
        trainer = pl.Trainer(
            max_epochs=TRAINING_CONFIG.max_epochs,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10,
            callbacks=[checkpoint_callback],
            logger=pl.loggers.CSVLogger(save_dir=str(logs_save_dir)) 
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
