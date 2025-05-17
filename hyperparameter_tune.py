"""Hyperparameter optimization for Deep Knowledge Tracing (DKT) models.

This module implements hyperparameter tuning using Optuna with PyTorch Lightning,
allowing for efficient exploration of the hyperparameter space to find optimal
model configurations.
"""

import os
import time
import json
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import optuna
import pandas as pd
import pytorch_lightning as pl
from optuna.trial import Trial
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar

from config import MODEL_CONFIG, TRAINING_CONFIG, PATHS
from models import DKT
from data import KTDataModule
from main import count_unique_skills

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PATHS.logs_dir / "hyperparameter_tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_N_TRIALS = 30
TRIAL_TIMEOUT = 1800  # 30 minutes per trial


class HyperparameterTuner:
    """Handles hyperparameter optimization for the DKT model.
    
    This class manages the entire hyperparameter tuning process, including
    trial execution, result tracking, and model evaluation.
    """
    
    def __init__(self, output_dir: Path, n_trials: int = DEFAULT_N_TRIALS) -> None:
        """Initialize the hyperparameter tuner.
        
        Args:
            output_dir: Directory to save tuning results and logs.
            n_trials: Maximum number of trials to run.
        """
        self.output_dir = Path(output_dir)
        self.n_trials = n_trials
        self.trials_df = pd.DataFrame()
        
        # Ensure output directories exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(exist_ok=True)
        
        logger.info(f"Initialized HyperparameterTuner with output_dir={output_dir}")
    
    def _get_model_params(self, trial: Trial) -> Dict[str, Any]:
        """Sample hyperparameters for a trial.
        
        Args:
            trial: Optuna trial object.
            
        Returns:
            Dictionary of model parameters.
        """
        return {
            'hidden_dim': trial.suggest_int('hidden_dim', 32, 256, step=32),
            'num_layers': trial.suggest_int('num_layers', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.05, 0.5, step=0.05),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 5e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
            'max_seq_len': trial.suggest_categorical('max_seq_len', [25, 50, 100, 200])
        }
    
    def _setup_data_module(self, batch_size: int, max_seq_len: int) -> KTDataModule:
        """Initialize and return the data module.
        
        Args:
            batch_size: Batch size for data loading.
            max_seq_len: Maximum sequence length.
            
        Returns:
            Initialized KTDataModule instance.
        """
        return KTDataModule(
            data_path=str(PATHS.training_data),
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            val_split=TRAINING_CONFIG.val_split,
            num_workers=TRAINING_CONFIG.num_workers,
            seed=42
        )
    
    def _create_trainer(self) -> pl.Trainer:
        """Create and configure the PyTorch Lightning Trainer.
        
        Returns:
            Configured Trainer instance.
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=2,
                mode='min',
                verbose=True
            ),
            TQDMProgressBar(refresh_rate=5)
        ]
        
        return pl.Trainer(
            max_epochs=TRAINING_CONFIG.max_epochs,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            log_every_n_steps=10,
            limit_val_batches=0.2,  # Reduced validation set for faster tuning
            limit_train_batches=0.8  # Reduced training set for faster epochs
        )
    
    def _log_trial_results(
        self,
        trial: Trial,
        result: float,
        metrics: Dict[str, float],
        elapsed_time: float,
        epoch: int
    ) -> None:
        """Log and save trial results.
        
        Args:
            trial: The Optuna trial.
            result: The main optimization metric (validation loss).
            metrics: Dictionary of additional metrics.
            elapsed_time: Time taken for the trial.
            epoch: Number of epochs completed.
        """
        trial_id = trial.number + 1
        
        # Log to console and file
        logger.info(f"Trial {trial_id} completed in {elapsed_time:.1f}s")
        logger.info(f"  - val_loss: {result:.4f}")
        for name, value in metrics.items():
            if value is not None:
                logger.info(f"  - {name}: {value:.4f}")
        logger.info(f"  - epochs: {epoch}/{TRAINING_CONFIG.max_epochs}")
        
        # Save to DataFrame
        trial_data = {
            'trial': trial_id,
            'val_loss': result,
            'epochs_completed': epoch,
            'elapsed_time': elapsed_time,
            **trial.params,
            **metrics
        }
        
        # Append to trials DataFrame
        self.trials_df = pd.concat([
            self.trials_df,
            pd.DataFrame([trial_data])
        ], ignore_index=True)
        
        # Save to CSV after each trial
        self.trials_df.to_csv(
            self.output_dir / "trials.csv",
            index=False,
            float_format="%.6f"
        )
    
    def objective(self, trial: Trial) -> float:
        """Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object.
            
        Returns:
            Validation loss for the trial.
            
        Raises:
            optuna.TrialPruned: If the trial is pruned due to poor performance.
        """
        start_time = time.time()
        trial_id = trial.number + 1
        
        # Sample hyperparameters
        params = self._get_model_params(trial)
        logger.info(f"\n{'='*50}\nStarting trial {trial_id}\n{'='*50}")
        for param, value in params.items():
            logger.info(f"  - {param}: {value}")
        
        try:
            # Setup data and model
            data_module = self._setup_data_module(
                batch_size=params['batch_size'],
                max_seq_len=params['max_seq_len']
            )
            
            num_skills = count_unique_skills(PATHS.training_data)
            model = DKT(
                num_skills=num_skills,
                hidden_dim=params['hidden_dim'],
                num_layers=params['num_layers'],
                dropout=params['dropout'],
                learning_rate=params['learning_rate']
            )
            
            # Train model
            trainer = self._create_trainer()
            logger.info(f"Starting training for trial {trial_id}")
            
            trainer.fit(model, datamodule=data_module)
            
            # Get validation metrics
            metrics = {
                k: v.item() if hasattr(v, 'item') else v
                for k, v in trainer.callback_metrics.items()
            }
            val_loss = metrics.get('val_loss', float('inf'))
            
            # Log results
            self._log_trial_results(
                trial=trial,
                result=val_loss,
                metrics=metrics,
                elapsed_time=time.time() - start_time,
                epoch=trainer.current_epoch + 1
            )
            
            # Check timeout
            if time.time() - start_time > TRIAL_TIMEOUT:
                logger.warning(f"Trial {trial_id} exceeded time limit of {TRIAL_TIMEOUT}s")
                raise optuna.TrialPruned()
            
            return val_loss
            
        except Exception as e:
            logger.error(f"Error in trial {trial_id}: {str(e)}", exc_info=True)
            return float('inf')
    
    def optimize(self) -> optuna.Study:
        """Run the hyperparameter optimization.
        
        Returns:
            The completed Optuna study with all trial results.
        """
        # Configure study
        study_name = f"dkt_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_url = f"sqlite:///{self.output_dir / f'{study_name}.db'}"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction='minimize',
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10
            )
        )
        
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        logger.info(f"Results will be saved to: {storage_url}")
        
        try:
            # Run optimization
            study.optimize(
                self.objective,
                n_trials=self.n_trials,
                show_progress_bar=True
            )
            
            # Save best parameters
            best_params_path = self.output_dir / "best_params.json"
            with open(best_params_path, 'w') as f:
                json.dump(study.best_params, f, indent=2)
            
            # Log summary
            logger.info(f"\n{'='*50}")
            logger.info("Hyperparameter optimization completed!")
            logger.info(f"Best trial: {study.best_trial.number}")
            logger.info(f"Best validation loss: {study.best_value:.6f}")
            logger.info(f"Best parameters: {study.best_params}")
            logger.info(f"Results saved to: {best_params_path}")
            
            return study
            
        except KeyboardInterrupt:
            logger.info("Optimization interrupted by user")
            return study


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for DKT model')
    parser.add_argument(
        '--n-trials',
        type=int,
        default=DEFAULT_N_TRIALS,
        help=f'Number of trials to run (default: {DEFAULT_N_TRIALS})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=str(PATHS.project_root / "tuning_results"),
        help='Directory to save results (default: ./tuning_results/)'
    )
    return parser.parse_args()


def main():
    """Main entry point for hyperparameter tuning."""
    args = parse_args()
    
    tuner = HyperparameterTuner(
        output_dir=args.output_dir,
        n_trials=args.n_trials
    )
    
    tuner.optimize()


if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"\nTuning interrupted after {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}", exc_info=True)
        raise
