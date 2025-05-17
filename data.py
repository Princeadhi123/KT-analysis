"""Data handling module for Knowledge Tracing tasks.

This module provides efficient data loading and processing for knowledge tracing models,
with support for batching, shuffling, and train/validation/test splits.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SequenceBatch:
    """Batch of sequences for knowledge tracing.
    
    Attributes:
        skill_ids: Tensor of shape [batch_size, seq_len] containing skill IDs
        correct: Tensor of shape [batch_size, seq_len] containing correctness (0 or 1)
        mask: Boolean tensor of shape [batch_size, seq_len] indicating valid positions
    """
    skill_ids: torch.Tensor
    correct: torch.Tensor
    mask: torch.Tensor


class KTDataset(Dataset):
    """Memory-efficient dataset for knowledge tracing sequences.
    
    This dataset stores interaction sequences as numpy arrays for memory efficiency
    and only converts to PyTorch tensors when needed.
    """
    
    def __init__(self, sequences: List[List[Tuple[int, int]]], max_seq_len: int = 100) -> None:
        """Initialize dataset with sequences of (skill_id, correct) tuples.
        
        Args:
            sequences: List of interaction sequences, where each sequence is a list of
                     (skill_id, correct) tuples.
            max_seq_len: Maximum sequence length to use. Longer sequences will be truncated.
        """
        if not sequences:
            raise ValueError("Cannot create dataset with empty sequences list")
            
        # Store sequences as numpy arrays for memory efficiency
        self.sequences = [
            np.array(seq[:max_seq_len], dtype=np.int32) 
            for seq in sequences if seq
        ]
        self.max_seq_len = max_seq_len
        
        logger.debug(f"Initialized KTDataset with {len(self)} sequences "
                    f"(max_len={max_seq_len})")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence by index.
        
        Args:
            idx: Index of the sequence to retrieve.
            
        Returns:
            Dictionary containing:
                - 'skill_ids': Tensor of shape [seq_len]
                - 'correct': Tensor of shape [seq_len]
                - 'mask': Boolean tensor of shape [seq_len]
        """
        seq = self.sequences[idx]
        
        # Convert to tensors only when needed
        skill_ids = torch.from_numpy(seq[:, 0]).long()
        correct = torch.from_numpy(seq[:, 1]).long()
        mask = torch.ones(len(seq), dtype=torch.bool)
        
        return {
            'skill_ids': skill_ids,
            'correct': correct,
            'mask': mask
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> SequenceBatch:
        """Batch sequences together with padding.
        
        Args:
            batch: List of dataset items, each containing 'skill_ids', 'correct', and 'mask'.
            
        Returns:
            SequenceBatch with padded sequences and corresponding mask.
        """
        if not batch:
            raise ValueError("Cannot collate empty batch")
            
        # Get max sequence length in this batch
        max_len = max(item['skill_ids'].size(0) for item in batch)
        batch_size = len(batch)
        
        # Pre-allocate tensors
        skill_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
        correct = torch.zeros(batch_size, max_len, dtype=torch.long)
        mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        
        # Fill tensors efficiently
        for i, item in enumerate(batch):
            seq_len = item['skill_ids'].size(0)
            skill_ids[i, :seq_len] = item['skill_ids']
            correct[i, :seq_len] = item['correct']
            mask[i, :seq_len] = item['mask']
        
        return SequenceBatch(
            skill_ids=skill_ids,
            correct=correct,
            mask=mask
        )


class KTDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for knowledge tracing.
    
    Handles data loading, preprocessing, and preparation of DataLoaders for
    training, validation, and testing.
    """
    
    def __init__(
        self, 
        data_path: str, 
        batch_size: int = 64,
        max_seq_len: int = 100, 
        val_split: float = 0.2,
        num_workers: int = 4,
        seed: int = 42
    ) -> None:
        """Initialize the data module.
        
        Args:
            data_path: Path to the preprocessed data file (CSV format).
            batch_size: Number of sequences per batch.
            max_seq_len: Maximum sequence length (longer sequences will be truncated).
            val_split: Fraction of data to use for validation (and same for test).
            num_workers: Number of subprocesses to use for data loading.
            seed: Random seed for reproducibility.
        """
        super().__init__()
        self.data_path = Path(data_path).resolve()
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        
        # Validate parameters
        if not 0 < val_split < 0.5:
            raise ValueError(f"val_split must be between 0 and 0.5, got {val_split}")
        
        # Initialize datasets
        self.train_dataset: Optional[KTDataset] = None
        self.val_dataset: Optional[KTDataset] = None
        self.test_dataset: Optional[KTDataset] = None
        
        logger.info(f"Initialized KTDataModule with batch_size={batch_size}, "
                   f"max_seq_len={max_seq_len}, val_split={val_split}")

    def prepare_data(self) -> None:
        """Verify data file exists (download if needed in the future)."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}. "
                "Please ensure the file exists and is accessible."
            )
        logger.info(f"Verified data file: {self.data_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load and split the data into train/val/test datasets."""
        if self.train_dataset is not None and self.val_dataset is not None:
            logger.debug("Data already loaded, skipping setup")
            return
            
        logger.info(f"Loading data from {self.data_path}")
        
        # Load data with optimized dtypes
        try:
            df = pd.read_csv(self.data_path, dtype={
                'user_id': 'category',
                'skill_id': np.int32,
                'correct': np.int32
            })
        except Exception as e:
            raise ValueError(f"Failed to load data from {self.data_path}: {e}")
        
        # Create and split sequences
        sequences = self._create_sequences(df)
        
        # Set random seed for reproducibility
        rng = np.random.RandomState(self.seed)
        rng.shuffle(sequences)
        
        # Calculate split indices
        val_size = int(self.val_split * len(sequences))
        train_size = len(sequences) - 2 * val_size
        
        if train_size <= 0:
            raise ValueError(
                f"Insufficient data for splits. Total sequences: {len(sequences)}, "
                f"val_split: {self.val_split}"
            )
        
        logger.info(f"Splitting data: train={train_size}, "
                   f"val={val_size}, test={val_size}")
        
        # Create datasets based on stage
        if stage in (None, 'fit'):
            self.train_dataset = KTDataset(
                sequences[:train_size],
                self.max_seq_len
            )
            self.val_dataset = KTDataset(
                sequences[train_size:train_size + val_size],
                self.max_seq_len
            )
            
        if stage in (None, 'test'):
            self.test_dataset = KTDataset(
                sequences[train_size + val_size:],
                self.max_seq_len
            )
            
        logger.info("Data loading completed")

    def _create_sequences(self, df: pd.DataFrame) -> List[List[Tuple[int, int]]]:
        """Create interaction sequences from dataframe.
        
        Args:
            df: DataFrame with columns 'user_id', 'skill_id', and 'correct'.
                
        Returns:
            List of sequences, where each sequence is a list of (skill_id, correct) tuples.
        """
        required_columns = {'user_id', 'skill_id', 'correct'}
        if not required_columns.issubset(df.columns):
            missing = required_columns - set(df.columns)
            raise ValueError(f"Missing required columns in data: {missing}")
        
        sequences = []
        # Group by user and sort by index to maintain temporal order
        for _, group in df.sort_index().groupby('user_id', sort=False):
            seq = list(zip(
                group['skill_id'].to_numpy(),
                group['correct'].to_numpy()
            ))
            if seq:  # Only add non-empty sequences
                sequences.append(seq)
                
        if not sequences:
            raise ValueError("No valid sequences found in the data")
            
        logger.info(f"Created {len(sequences)} sequences from {len(df)} interactions")
        return sequences

    def _get_dataloader(
        self,
        dataset: Optional[KTDataset],
        shuffle: bool = False
    ) -> DataLoader:
        """Create a DataLoader with common settings.
        
        Args:
            dataset: Dataset to create loader for.
            shuffle: Whether to shuffle the data.
            
        Returns:
            Configured DataLoader instance.
            
        Raises:
            ValueError: If dataset is None.
        """
        if dataset is None:
            raise ValueError(
                "Dataset not initialized. Call setup() before requesting a DataLoader."
            )
            
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=KTDataset.collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0
        )

    def train_dataloader(self) -> DataLoader:
        """Create DataLoader for training data."""
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create DataLoader for validation data."""
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Create DataLoader for test data."""
        return self._get_dataloader(self.test_dataset, shuffle=False)
