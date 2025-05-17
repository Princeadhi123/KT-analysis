"""Script for making predictions using a trained DKT model."""
import os
import logging
import torch
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
from typing import List, Dict, Any

from models import DKT
from data import KTDataModule
from config import TRAINING_CONFIG, PATHS, PREDICTION_CONFIG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add Windows path support if needed
if os.name == 'nt':
    import pathlib
    torch.serialization.add_safe_globals([pathlib.WindowsPath])

# Set paths
CKPT_PATH = PATHS.checkpoint_dir / PREDICTION_CONFIG.model_checkpoint
OUTPUT_CSV = PATHS.predictions_dir / PREDICTION_CONFIG.output_csv

def load_model(ckpt_path: Path) -> pl.LightningModule:
    """Load a trained DKT model from checkpoint.
    
    Args:
        ckpt_path: Path to the model checkpoint
        
    Returns:
        Loaded DKT model
    """
    try:
        logger.info(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        hparams = ckpt['hyper_parameters']
        
        model = DKT(
            num_skills=hparams['num_skills'],
            hidden_dim=hparams['hidden_dim'],
            num_layers=hparams['num_layers'],
            dropout=hparams['dropout'],
            learning_rate=hparams['learning_rate']
        )
        model = model.load_from_checkpoint(ckpt_path, map_location='cpu', weights_only=False)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Error loading model from {ckpt_path}: {str(e)}")
        raise

def main() -> None:
    """Main function to run predictions on test data."""
    try:
        # Initialize data module
        logger.info("Setting up data module")
        data_module = KTDataModule(
            data_path=str(PATHS.data),
            batch_size=TRAINING_CONFIG.batch_size,
            max_seq_len=TRAINING_CONFIG.max_seq_len,
            val_split=TRAINING_CONFIG.val_split
        )
        data_module.setup('test')
        test_loader = data_module.test_dataloader()
        
        # Load model
        model = load_model(CKPT_PATH)
        num_skills = model.hparams.num_skills
        
        # Run predictions
        logger.info("Running predictions...")
        all_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                logits = model(batch)  # [batch, seq_len, num_skills]
                probs = torch.sigmoid(logits)  # Probabilities
                
                # For each sequence in batch
                for i in range(probs.shape[0]):
                    skill_ids = batch.skill_ids[i].cpu().numpy()
                    correct = batch.correct[i].cpu().numpy()
                    mask = batch.mask[i].cpu().numpy()
                    prob_matrix = probs[i].cpu().numpy()
                    
                    # For each timestep in sequence
                    for t in range(len(skill_ids)):
                        if mask[t]:
                            for skill in range(num_skills):
                                all_results.append({
                                    'sequence_idx': f"{batch_idx}_{i}",
                                    'step': t,
                                    'skill_id': skill,
                                    'pred_prob': float(prob_matrix[t, skill]),
                                    'true_correct': int(correct[t]) if skill == skill_ids[t] else None
                                })
                
                # Log progress
                if (batch_idx + 1) % 10 == 0:
                    logger.info(f"Processed {batch_idx + 1} batches")
        
        # Save results
        logger.info(f"Saving predictions to {OUTPUT_CSV}")
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Successfully saved {len(df)} predictions")
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
