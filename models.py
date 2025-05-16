"""Efficient Knowledge Tracing model implementation."""

from typing import Dict, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl

class DKT(pl.LightningModule):
    """Deep Knowledge Tracing model with optimized implementation."""
    
    def __init__(self, 
                 num_skills: int, 
                 hidden_dim: int = 128, 
                 num_layers: int = 2, 
                 dropout: float = 0.3, 
                 learning_rate: float = 1e-3):
        """Initialize the DKT model.
        
        Args:
            num_skills: Number of unique skills in the dataset
            hidden_dim: Dimension of hidden states in LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            learning_rate: Learning rate for optimization
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Optimize memory usage by combining skill and correctness embedding
        self.embedding = nn.Embedding(num_skills * 2, hidden_dim)
        
        # Use bidirectional LSTM for better sequence understanding
        self.lstm = nn.LSTM(
            hidden_dim, 
            hidden_dim // 2, # Half size for bidirectional
            num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_skills)
        self.learning_rate = learning_rate

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Combine skill and correctness into single input
        x = batch.skill_ids * 2 + batch.correct
        x = self.embedding(x)
        
        # Get sequence lengths for packing
        lengths = batch.mask.sum(1).cpu()
        
        # Pack sequence for efficient computation
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        # Process through LSTM
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        # Apply dropout and final layer
        output = self.dropout(output)
        return self.output(output)

    def _shared_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Shared step for training and validation."""
        logits = self(batch)
        
        # Get targets and masks
        target_ids = batch.skill_ids[:, 1:]
        target_correct = batch.correct[:, 1:]
        target_mask = batch.mask[:, 1:]
        
        # Get predictions
        pred = torch.gather(logits[:, :-1], 2, target_ids.unsqueeze(-1)).squeeze(-1)
        
        # Calculate metrics
        loss = F.binary_cross_entropy_with_logits(
            pred[target_mask],
            target_correct[target_mask].float()
        )
        
        with torch.no_grad():
            accuracy = ((torch.sigmoid(pred) >= 0.5) == target_correct)[target_mask].float().mean()
        
        return {'loss': loss, 'accuracy': accuracy}

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        metrics = self._shared_step(batch)
        self.log('train_loss', metrics['loss'])
        self.log('train_acc', metrics['accuracy'], prog_bar=True)
        return metrics['loss']

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        metrics = self._shared_step(batch)
        self.log('val_loss', metrics['loss'], prog_bar=True)
        self.log('val_acc', metrics['accuracy'], prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
