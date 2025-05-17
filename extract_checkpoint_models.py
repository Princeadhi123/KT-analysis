"""Script to extract and display information from model checkpoints."""
import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union

import torch
import pandas as pd
from tabulate import tabulate

from config import PATHS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_checkpoint(ckpt_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
    """Safely load a PyTorch checkpoint.
    
    Args:
        ckpt_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing checkpoint data or None if loading failed
    """
    try:
        # Try with weights_only=False first (for PyTorch 2.0+)
        return torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except TypeError:
        try:
            # Fall back to standard loading
            return torch.load(ckpt_path, map_location='cpu')
        except Exception as e:
            logger.error(f"Error loading checkpoint {ckpt_path}: {e}")
            return None

def extract_ckpt_info(ckpt_path: Union[str, Path]) -> Dict[str, Any]:
    """Extract relevant information from a checkpoint file.
    
    Args:
        ckpt_path: Path to the checkpoint file
        
    Returns:
        Dictionary containing extracted information
    """
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is None:
        return {}
        
    result = {
        'checkpoint': os.path.basename(ckpt_path),
        'hyperparameters': ckpt.get('hyper_parameters', {}),
        'val_loss': None,
        'epoch': ckpt.get('epoch'),
        'global_step': ckpt.get('global_step')
    }
    
    # Try to extract validation loss from various possible locations
    result['val_loss'] = ckpt.get('best_model_score')
    
    if result['val_loss'] is None and 'callbacks' in ckpt:
        for cb_val in ckpt['callbacks'].values():
            if not isinstance(cb_val, dict):
                continue
                
            # Check for common callback structures
            for k, v in cb_val.items():
                if any(term in k.lower() for term in ['best_model_score', 'best_model', 'val_loss']):
                    result['val_loss'] = v
                    break
            
            if result['val_loss'] is not None:
                break
    
    return result

def display_ckpt_info(info: Dict[str, Any]) -> None:
    """Display checkpoint information in a formatted way."""
    print(f"\n{'='*80}")
    print(f"Checkpoint: {info['checkpoint']}")
    
    if info.get('epoch') is not None:
        print(f"Epoch: {info['epoch']}")
    if info.get('global_step') is not None:
        print(f"Global Step: {info['global_step']}")
    
    print("\nHyperparameters:")
    print(json.dumps(info.get('hyperparameters', {}), indent=2))
    
    print("\nValidation Loss:")
    print(info.get('val_loss', 'Not found'))
    print("="*80)

def main() -> None:
    """Main function to process all checkpoints in the directory."""
    ckpt_dir = PATHS.checkpoint_dir
    
    if not ckpt_dir.exists():
        logger.error(f"Checkpoint directory not found: {ckpt_dir}")
        return
    
    ckpt_files = list(ckpt_dir.glob('*.ckpt'))
    if not ckpt_files:
        logger.warning(f"No checkpoint files found in {ckpt_dir}")
        return
    
    logger.info(f"Found {len(ckpt_files)} checkpoint(s) in {ckpt_dir}")
    
    all_info = []
    for ckpt_file in ckpt_files:
        info = extract_ckpt_info(ckpt_file)
        if info:  # Only process if checkpoint was loaded successfully
            all_info.append(info)
            display_ckpt_info(info)
    
    # Create a summary table if we have multiple checkpoints
    if len(all_info) > 1:
        summary = []
        for info in all_info:
            summary.append({
                'Checkpoint': info['checkpoint'],
                'Epoch': info.get('epoch', 'N/A'),
                'Step': info.get('global_step', 'N/A'),
                'Val Loss': f"{info.get('val_loss', 'N/A'):.6f}" if info.get('val_loss') is not None else 'N/A'
            })
        
        print("\nSummary of Checkpoints:")
        print(tabulate(summary, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    main()
