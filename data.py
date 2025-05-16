"""Optimized data handling for Knowledge Tracing."""

from typing import Dict, List, Tuple, Optional, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

@dataclass
class SequenceBatch:
    skill_ids: torch.Tensor
    correct: torch.Tensor
    mask: torch.Tensor

class KTDataset(Dataset):
    """Memory-efficient dataset for knowledge tracing."""
    
    def __init__(self, sequences: List[List[Tuple[int, int]]], max_seq_len: int = 100):
        """Initialize dataset with sequences of (skill_id, correct) tuples.
        
        Args:
            sequences: List of interaction sequences
            max_seq_len: Maximum sequence length to use
        """
        # Store sequences as numpy arrays for memory efficiency
        self.sequences = [
            np.array(seq[:max_seq_len], dtype=np.int32) 
            for seq in sequences if seq
        ]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
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
        """Efficiently batch sequences together."""
        # Get max length in this batch
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
        
        return SequenceBatch(skill_ids=skill_ids, correct=correct, mask=mask)

class KTDataModule(pl.LightningDataModule):
    """Efficient data module for knowledge tracing."""
    
    def __init__(self, 
                 data_path: str, 
                 batch_size: int = 64,
                 max_seq_len: int = 100, 
                 val_split: float = 0.1,
                 num_workers: int = 4,
                 seed: int = 42):
        """Initialize data module.
        
        Args:
            data_path: Path to preprocessed data file
            batch_size: Batch size for training
            max_seq_len: Maximum sequence length
            val_split: Fraction of data to use for validation
            num_workers: Number of workers for data loading
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed
        
        # Initialize datasets
        self.train_dataset: Optional[KTDataset] = None
        self.val_dataset: Optional[KTDataset] = None
        self.test_dataset: Optional[KTDataset] = None

    def prepare_data(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

    def setup(self, stage: Optional[str] = None) -> None:
        # Load data efficiently
        df = pd.read_csv(self.data_path, dtype={
            'user_id': 'category',
            'skill_id': np.int32,
            'correct': np.int32
        })
        
        # Create and split sequences
        sequences = self._create_sequences(df)
        np.random.seed(self.seed)
        np.random.shuffle(sequences)
        
        # Calculate split sizes
        train_size = int((1 - self.val_split * 2) * len(sequences))
        val_size = int(self.val_split * len(sequences))
        
        # Create datasets
        if stage in (None, 'fit'):
            self.train_dataset = KTDataset(sequences[:train_size], self.max_seq_len)
            self.val_dataset = KTDataset(
                sequences[train_size:train_size + val_size], 
                self.max_seq_len
            )
        if stage in (None, 'test'):
            self.test_dataset = KTDataset(
                sequences[train_size + val_size:], 
                self.max_seq_len
            )

    def _create_sequences(self, df: pd.DataFrame) -> List[List[Tuple[int, int]]]:
        """Efficiently create sequences from dataframe."""
        sequences = []
        # Use itertuples for faster iteration
        for _, group in df.groupby('user_id'):
            seq = list(zip(
                group['skill_id'].to_numpy(),
                group['correct'].to_numpy()
            ))
            if seq:
                sequences.append(seq)
        return sequences

    def _get_dataloader(self, dataset: KTDataset, shuffle: bool) -> DataLoader:
        """Create dataloader with common settings."""
        if dataset is None:
            raise ValueError("Dataset not initialized. Call setup() first.")
            
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=KTDataset.collate_fn,
            pin_memory=True
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test_dataset, shuffle=False)
