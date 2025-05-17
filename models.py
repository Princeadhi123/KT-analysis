"""Deep Knowledge Tracing (DKT) model implementation."""

from typing import Dict, Tuple, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl


class DKT(pl.LightningModule):
    """Deep Knowledge Tracing model with optimized implementation.
    
    This model predicts the probability of a student answering correctly
    based on their interaction history with different skills.
    """
    
    def __init__(
        self, 
        num_skills: int, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        dropout: float = 0.3, 
        learning_rate: float = 1e-3
    ) -> None:
        """Initialize the DKT model.
        
        Args:
            num_skills: Number of unique skills in the dataset.
            hidden_dim: Dimension of hidden states in LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate for regularization.
            learning_rate: Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Combine skill and correctness into a single embedding
        self.embedding = nn.Embedding(num_skills * 2, hidden_dim)
        
        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # Half size for bidirectional
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_skills)
        self.learning_rate = learning_rate

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for DKT model.
        
        Args:
            batch: Dictionary containing:
                - skill_ids: Tensor of shape [batch_size, seq_len]
                - correct: Tensor of shape [batch_size, seq_len]
                - mask: Tensor of shape [batch_size, seq_len]
                
        Returns:
            Tensor of shape [batch_size, seq_len, num_skills] with logits.
        """
        # Combine skill and correctness into single embedding
        x = batch.skill_ids * 2 + batch.correct
        x = self.embedding(x)
        
        # Process variable-length sequences
        lengths = batch.mask.sum(1).cpu()
        packed = pack_padded_sequence(
            x, 
            lengths, 
            batch_first=True, 
            enforce_sorted=False
        )
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        output = self.dropout(output)
        
        return self.output(output)

    def _shared_step(
        self, 
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Shared step for training and validation.
        
        Args:
            batch: Input batch of data.
            
        Returns:
            Dictionary containing loss and accuracy metrics.
        """
        logits = self(batch)
        
        # Prepare targets and masks
        target_ids = batch.skill_ids[:, 1:]
        target_correct = batch.correct[:, 1:]
        target_mask = batch.mask[:, 1:]
        
        # Get predictions for target skills
        pred = torch.gather(
            logits[:, :-1], 
            dim=2, 
            index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
        
        # Calculate loss and accuracy
        loss = F.binary_cross_entropy_with_logits(
            pred[target_mask],
            target_correct[target_mask].float()
        )
        
        with torch.no_grad():
            pred_probs = torch.sigmoid(pred)
            accuracy = ((pred_probs >= 0.5) == target_correct)[target_mask].float().mean()
        
        return {
            'loss': loss, 
            'accuracy': accuracy
        }

    def training_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Training step with logging."""
        metrics = self._shared_step(batch)
        self.log('train_loss', metrics['loss'])
        self.log('train_acc', metrics['accuracy'], prog_bar=True)
        return metrics['loss']

    def validation_step(
        self, 
        batch: Dict[str, torch.Tensor], 
        batch_idx: int
    ) -> None:
        """Validation step with logging."""
        metrics = self._shared_step(batch)
        self.log('val_loss', metrics['loss'], prog_bar=True)
        self.log('val_acc', metrics['accuracy'], prog_bar=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            },
            'monitor': 'val_loss'
        }
