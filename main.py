"""Main script for training the knowledge tracing model (refactored)."""

from pathlib import Path
import pandas as pd
import pytorch_lightning as pl
from data import KTDataModule
from models import DKT
from config import MODEL_CONFIG, TRAINING_CONFIG, PATHS
import json
import os

def count_unique_skills(data_path: Path) -> int:
    """
    Count number of unique skills in the dataset.
    Args:
        data_path (Path): Path to the dataset CSV file.
    Returns:
        int: Number of unique skills.
    """
    df = pd.read_csv(data_path)
    return df['skill_id'].nunique()

def train_model() -> None:
    """Train the knowledge tracing model."""
    data_path = PATHS.data
    checkpoint_dir = PATHS.checkpoint_dir
    # Initialize data module
    data_module = KTDataModule(
        data_path=str(data_path),
        batch_size=TRAINING_CONFIG.batch_size,
        max_seq_len=TRAINING_CONFIG.max_seq_len,
        val_split=TRAINING_CONFIG.val_split
    )
    # Count unique skills
    num_skills = count_unique_skills(data_path)
    print(f"Number of unique skills: {num_skills}")

    # Try to load best hyperparameters
    best_hparams = {}
    hparam_path = "best_hyperparameters.json"
    if os.path.exists(hparam_path):
        with open(hparam_path, "r") as f:
            best_hparams = json.load(f)
        print(f"Loaded best hyperparameters from {hparam_path}: {best_hparams}")
    else:
        print("No best_hyperparameters.json found, using config defaults.")

    # Helper function to get hyperparameters with fallback
    def get_param(param_name, default):
        return best_hparams.get(param_name, default)

    # Create model with loaded or default hyperparameters
    model = DKT(
        num_skills=num_skills,
        hidden_dim=get_param("hidden_dim", MODEL_CONFIG.hidden_dim),
        num_layers=get_param("num_layers", MODEL_CONFIG.num_layers),
        dropout=get_param("dropout", MODEL_CONFIG.dropout),
        learning_rate=get_param("learning_rate", TRAINING_CONFIG.learning_rate)
    )
    # Configure trainer with checkpoint callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='best_model',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    trainer = pl.Trainer(
        max_epochs=TRAINING_CONFIG.max_epochs,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback]
    )
    print("Starting training...")
    trainer.fit(model, data_module)
    print("Training completed!")

if __name__ == "__main__":
    train_model()
