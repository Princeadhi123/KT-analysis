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
    
import numpy as np # Added for np.load

class DKT(pl.LightningModule):
    """Deep Knowledge Tracing model with optimized implementation.
    
    This model predicts the probability of a student answering correctly
    based on their interaction history with different skills.
    It now uses pre-computed PAF skill embeddings.
    """
    
    def __init__(
        self,
        num_outputs: int, # Number of original skills to predict for (len(original_skill_id_to_factor_idx))
        embedding_dim: int, # Dimensionality of PAF embeddings (num_factors)
        skill_embeddings_path: str, # Path to 'skill_factor_embeddings.npy'
        hidden_dim: int = 128, # LSTM hidden dim
        num_layers: int = 2,
        dropout: float = 0.3,
        learning_rate: float = 1e-3
    ) -> None:
        """Initialize the DKT model with PAF embeddings.
        
        Args:
            num_outputs: Number of unique original skills to predict mastery for.
            embedding_dim: Dimensionality of the skill embeddings (from PAF).
            skill_embeddings_path: Path to the .npy file containing skill_factor_embeddings.
            hidden_dim: Dimension of hidden states in LSTM.
            num_layers: Number of LSTM layers.
            dropout: Dropout rate for regularization.
            learning_rate: Learning rate for the optimizer.
        """
        super().__init__()
        self.save_hyperparameters()

        # Load PAF embeddings
        # These are expected to be of shape [num_outputs, embedding_dim]
        paf_embeddings = torch.tensor(np.load(skill_embeddings_path), dtype=torch.float)
        
        if paf_embeddings.shape[0] != num_outputs:
            raise ValueError(
                f"Mismatch in num_outputs ({num_outputs}) and loaded paf_embeddings rows ({paf_embeddings.shape[0]}). "
                "num_outputs should be the number of unique skills from original_skill_id_to_factor_idx."
            )
        if paf_embeddings.shape[1] != embedding_dim:
            raise ValueError(
                f"Mismatch in embedding_dim ({embedding_dim}) and loaded paf_embeddings columns ({paf_embeddings.shape[1]}). "
                "embedding_dim should be the number of factors from PAF."
            )

        # Create expanded embedding matrix for (skill, correctness) pairs
        # Final embedding matrix shape: [num_outputs * 2, embedding_dim]
        expanded_embeddings = torch.zeros(num_outputs * 2, embedding_dim)
        for i in range(num_outputs):
            expanded_embeddings[2*i] = paf_embeddings[i]     # skill_i, incorrect
            expanded_embeddings[2*i + 1] = paf_embeddings[i] # skill_i, correct
        
        self.embedding = nn.Embedding(num_outputs * 2, embedding_dim)
        self.embedding.weight = nn.Parameter(expanded_embeddings, requires_grad=True) # Fine-tune

        # Bidirectional LSTM for sequence modeling
        # Input to LSTM is now the PAF embedding dimension
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_dim // 2,  # Half size for bidirectional
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout)
        # Output layer predicts mastery for each of the original skills
        self.output = nn.Linear(hidden_dim, num_outputs) 
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
