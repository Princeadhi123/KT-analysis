"""Hyperparameter tuning for DKT model using Optuna and PyTorch Lightning."""
import optuna
import pytorch_lightning as pl
import time
import os
import logging
from datetime import datetime
from pytorch_lightning.callbacks import EarlyStopping, TQDMProgressBar
from config import MODEL_CONFIG, TRAINING_CONFIG, PATHS
from models import DKT
from data import KTDataModule
from main import count_unique_skills

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("tuning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def objective(trial) -> float:
    """
    Objective function for Optuna hyperparameter optimization.
    Args:
        trial: Optuna trial object.
    Returns:
        float: Validation loss for the trial.
    """
    # Set a time limit for each trial (30 minutes)
    TRIAL_TIMEOUT = 1800  # seconds
    start_time = time.time()
    logger.info(f"\n{'='*50}\nStarting trial {trial.number + 1}\n{'='*50}")
    
    # Suggest hyperparameters (expanded search space for better tuning)
    hidden_dim = trial.suggest_int('hidden_dim', 32, 256)
    num_layers = trial.suggest_int('num_layers', 1, 4)
    dropout = trial.suggest_float('dropout', 0.05, 0.5)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 5e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    max_seq_len = trial.suggest_categorical('max_seq_len', [25, 50, 100, 200])
    
    logger.info(f"Trial {trial.number + 1} parameters: ")
    logger.info(f"  - hidden_dim: {hidden_dim}")
    logger.info(f"  - num_layers: {num_layers}")
    logger.info(f"  - dropout: {dropout:.4f}")
    logger.info(f"  - learning_rate: {learning_rate:.6f}")
    logger.info(f"  - batch_size: {batch_size}")
    logger.info(f"  - max_seq_len: {max_seq_len}")

    try:
        # Prepare data
        data_module = KTDataModule(
            data_path=str(PATHS.data),
            batch_size=batch_size,
            max_seq_len=max_seq_len,
            val_split=TRAINING_CONFIG.val_split,
            num_workers=TRAINING_CONFIG.num_workers,
            seed=42
        )
        num_skills = count_unique_skills(PATHS.data)
        logger.info(f"Loaded data with {num_skills} unique skills")

        # Model
        model = DKT(
            num_skills=num_skills,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=learning_rate
        )

        # Add callbacks for better monitoring
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=2, mode='min'),  # Reduced patience
            TQDMProgressBar(refresh_rate=5)  # More frequent updates
        ]

        # Trainer with progress bar enabled
        trainer = pl.Trainer(
            max_epochs=TRAINING_CONFIG.max_epochs,
            callbacks=callbacks,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,  # Enable progress bar
            enable_model_summary=True,  # Show model summary
            log_every_n_steps=10,
            # Limit validation batches for faster tuning
            limit_val_batches=0.2,  # Reduced validation set
            limit_train_batches=0.8  # Limit training data for faster epochs
        )
        
        logger.info(f"Starting model training for trial {trial.number + 1}")
        trainer.fit(model, datamodule=data_module)
        val_loss = trainer.callback_metrics.get("val_loss")
        val_acc = trainer.callback_metrics.get("val_acc")
        
        result = val_loss.item() if val_loss is not None else float('inf')
        acc_result = val_acc.item() if val_acc is not None else None
        
        # Log trial results
        elapsed_time = time.time() - start_time
        logger.info(f"Trial {trial.number + 1} completed in {elapsed_time:.2f} seconds")
        logger.info(f"  - val_loss: {result:.4f}")
        if val_acc is not None:
            logger.info(f"  - val_acc: {acc_result:.4f}")
        logger.info(f"  - epochs completed: {trainer.current_epoch + 1}/{TRAINING_CONFIG.max_epochs}")
        
        # Save trial results to CSV
        import csv
        csv_path = "tuning_results/all_trials.csv"
        fieldnames = ["trial", "hidden_dim", "num_layers", "dropout", "learning_rate", "batch_size", "max_seq_len", "val_loss", "val_acc"]
        row = {
            "trial": trial.number + 1,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_seq_len": max_seq_len,
            "val_loss": result,
            "val_acc": acc_result
        }
        try:
            write_header = not os.path.exists(csv_path)
            with open(csv_path, mode="a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"Could not write trial results to CSV: {e}")
        
        # Check if we've exceeded the time limit
        if time.time() - start_time > TRIAL_TIMEOUT:
            logger.warning(f"Trial {trial.number + 1} exceeded time limit of {TRIAL_TIMEOUT} seconds")
            raise optuna.TrialPruned()
        
        return result
        
    except Exception as e:
        logger.error(f"Error in trial {trial.number + 1}: {str(e)}")
        return float('inf')


def main():
    # Create output directory for logs
    os.makedirs("tuning_results", exist_ok=True)
    
    # Configure Optuna study
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    study_name = f"dkt_tuning_{timestamp}"
    storage_name = f"sqlite:///tuning_results/{study_name}.db"
    
    logger.info(f"Starting hyperparameter tuning study: {study_name}")
    logger.info(f"Results will be saved to: {storage_name}")
    
    # Create and configure the study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction='minimize',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )
    
    # Set number of trials
    n_trials = 30  # Increased number of trials for better search
    logger.info(f"Running {n_trials} trials...")
    
    # Start optimization with progress tracking
    start_time = time.time()
    try:
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        # Log results
        elapsed_time = time.time() - start_time
        logger.info(f"\n{'='*50}")
        logger.info(f"Hyperparameter tuning completed in {elapsed_time:.2f} seconds")
        logger.info(f"Best hyperparameters: {study.best_params}")
        logger.info(f"Best validation loss: {study.best_value:.6f}")
        
        # Print results to console as well
        print(f"\nHyperparameter tuning completed!")
        print(f"Total time: {elapsed_time:.2f} seconds")
        print(f"Best hyperparameters: {study.best_params}")
        print(f"Best validation loss: {study.best_value:.6f}")
        
        # Save best hyperparameters to JSON for use in main.py
        with open("best_hyperparameters.json", "w") as f:
            import json
            json.dump(study.best_params, f, indent=2)
        print("Best hyperparameters saved to best_hyperparameters.json")
    
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        logger.info("Tuning interrupted by user")
        if len(study.trials) > 0:
            logger.info(f"Best hyperparameters so far: {study.best_params}")
            logger.info(f"Best validation loss so far: {study.best_value:.6f}")
            print(f"\nTuning interrupted after {elapsed_time:.2f} seconds")
            print(f"Best hyperparameters so far: {study.best_params}")
            print(f"Best validation loss so far: {study.best_value:.6f}")

if __name__ == "__main__":
    main()
