"""Efficient data preprocessing for Knowledge Tracing."""

import os
import json
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
import numpy as np
from tqdm import tqdm

from config import PATHS

@dataclass
class PreprocessingStats:
    num_records: int
    num_users: int
    num_skills: int
    avg_sequence_length: float
    overall_accuracy: float
    input_file_size_mb: float
    output_file_size_mb: float
    processing_time_seconds: float
    memory_used_mb: float
def create_skill_mapping(df: pd.DataFrame) -> Dict[int, int]:
    """
    Create mapping of original skill IDs to consecutive integers.
    Args:
        df (pd.DataFrame): DataFrame containing 'skill_id' column.
    Returns:
        Dict[int, int]: Mapping from original skill IDs to consecutive integers.
    """
    unique_skills = df['skill_id'].unique()
    return {skill: idx for idx, skill in enumerate(sorted(unique_skills))}

def preprocess_data(input_file: str, output_file: str) -> str:
    """
    Efficiently preprocess student interaction data for knowledge tracing.
    Args:
        input_file: Path to input CSV file
        output_file: Path to save preprocessed data
    Returns:
        Path to saved preprocessed file
    """
    start_time = time.time()
    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"Loading data from {input_file} ({file_size_mb:.2f} MB)...")

    # Efficiently load data
    df = pd.read_csv(input_path, low_memory=False)

    # Rename columns if needed
    column_mapping = {
        'ITEST_id': 'user_id',
        'skill': 'skill_id',
        'correct': 'correct',
        'startTime': 'timestamp'
    }
    df = df.rename(columns={col: new_col for col, new_col in column_mapping.items() if col in df.columns})

    # Ensure required columns
    required_columns = {'user_id', 'skill_id', 'correct', 'timestamp'}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert timestamp
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    # Convert correct to int (0/1)
    if df['correct'].dtype not in (np.int64, np.int32):
        correct_map = {'true': 1, 'True': 1, '1': 1, 1: 1, True: 1, 
                      'false': 0, 'False': 0, '0': 0, 0: 0, False: 0}
        df['correct'] = df['correct'].map(correct_map).fillna(0).astype(int)

    # Drop rows with missing values in required columns
    df = df.dropna(subset=required_columns)

    # Sort by user_id and timestamp
    df = df.sort_values(['user_id', 'timestamp'])

    # Create skill mapping
    skill_to_id = create_skill_mapping(df)
    id_to_skill = {v: k for k, v in skill_to_id.items()}
    df['skill_id'] = df['skill_id'].map(skill_to_id).astype(int)

    # Save mappings
    checkpoints_dir = Path('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)
    with (checkpoints_dir / 'skill_mapping.json').open('w') as f:
        json.dump(skill_to_id, f, indent=2)
    with (checkpoints_dir / 'id_to_skill_mapping.json').open('w') as f:
        json.dump(id_to_skill, f, indent=2)

    # Save processed data (with compression for large files)
    use_compression = file_size_mb > 50 and output_path.suffix == '.csv'
    if use_compression:
        compressed_output = output_path.with_suffix('.csv.gz')
        df.to_csv(compressed_output, index=False, compression='gzip')
        output_path = compressed_output
    else:
        df.to_csv(output_path, index=False)

    # Gather stats
    elapsed = time.time() - start_time
    stats = PreprocessingStats(
        num_records=len(df),
        num_users=df['user_id'].nunique(),
        num_skills=len(skill_to_id),
        avg_sequence_length=df.groupby('user_id').size().mean(),
        overall_accuracy=df['correct'].mean(),
        input_file_size_mb=file_size_mb,
        output_file_size_mb=output_path.stat().st_size / (1024 * 1024),
        processing_time_seconds=elapsed,
        memory_used_mb=0
    )
    with (checkpoints_dir / 'preprocessing_summary.json').open('w') as f:
        json.dump(stats.__dict__, f, indent=2)
    print("\nMissing values per column:")
    missing_values = df.isnull().sum()
    print(missing_values)
    
    # Drop rows with missing values in required columns only (more efficient)
    initial_count = len(df)
    df = df.dropna(subset=required_columns)
    if len(df) < initial_count:
        print(f"\nDropped {initial_count - len(df)} rows with missing values in required columns")
    
    # Sort by user_id and timestamp - use more efficient sorting with parallel algorithms
    if use_parallel and len(df) > 100000:
        # For large dataframes, use a more efficient sorting method
        df = df.sort_values(['user_id', 'timestamp'], kind='mergesort')
    else:
        df = df.sort_values(['user_id', 'timestamp'])
    
    # Create a mapping of skills to unique IDs starting from 1 - use more efficient method
    @lru_cache(maxsize=1)
    def create_skill_mapping(skills_tuple):
        """Create and cache skill mappings to avoid recomputation"""
        return {skill: i+1 for i, skill in enumerate(sorted(skills_tuple))}
    
    # Use the cached function if enabled
    if cache_mappings:
        # Convert unique skills to tuple for caching (since lists aren't hashable)
        unique_skills = tuple(sorted(df['skill_id'].unique()))
        skill_to_id = create_skill_mapping(unique_skills)
    else:
        skill_to_id = {skill: i+1 for i, skill in enumerate(sorted(df['skill_id'].unique()))}
    
    # Create reverse mapping for reference
    id_to_skill = {v: k for k, v in skill_to_id.items()}
    
    # Save the mappings to files
    os.makedirs('checkpoints', exist_ok=True)
    
    # Save skill to ID mapping with custom encoder for NumPy types
    with open('checkpoints/skill_mapping.json', 'w') as f:
        json.dump(skill_to_id, f, indent=2, cls=NumpyEncoder)
    
    # Save ID to skill mapping for reference
    with open('checkpoints/id_to_skill_mapping.json', 'w') as f:
        json.dump(id_to_skill, f, indent=2, cls=NumpyEncoder)
    
    # Map skills to IDs in the dataframe - use more efficient method
    # Convert mapping to a Series for faster mapping
    mapping_series = pd.Series(skill_to_id)
    df['skill_id'] = df['skill_id'].map(mapping_series).astype(int)
    
    # Add some basic statistics
    print("\nSkill distribution (top 10 most frequent):")
    skill_counts = df['skill_id'].value_counts().head(10)
    print(skill_counts)
    
    # Calculate correctness rate per skill
    print("\nAverage correctness per skill (top 10):")
    skill_accuracy = df.groupby('skill_id')['correct'].mean().sort_values(ascending=False).head(10)
    print(skill_accuracy)
    
    # Save the preprocessed data with compression for smaller file size
    print(f"\nSaving preprocessed data to {output_file}...")
    
    # Determine if we should use compression based on file size
    use_compression = file_size_mb > 50  # Use compression for files larger than 50MB
    
    if use_compression and output_file.endswith('.csv'):
        # Add .gz extension if not present
        compressed_output = f"{output_file}.gz"
        print(f"Using compression to save file as {compressed_output}")
        df.to_csv(compressed_output, index=False, compression='gzip')
        output_file = compressed_output
    else:
        df.to_csv(output_file, index=False)
    
    # Create a summary of the preprocessed data with additional performance metrics
    # Calculate performance metrics
    end_time = time.time()
    processing_time = end_time - start_time
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)  # Memory in MB
    
    summary = {
        'num_records': len(df),
        'num_users': df['user_id'].nunique(),
        'num_skills': len(skill_to_id),
        'avg_sequence_length': df.groupby('user_id').size().mean(),
        'overall_accuracy': df['correct'].mean(),
        'preprocessing_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'processing_time_seconds': processing_time,
        'memory_usage_mb': mem_after - mem_before,
        'file_size_mb': file_size_mb,
        'output_file_size_mb': os.path.getsize(output_file) / (1024 * 1024) if os.path.exists(output_file) else 0
    }
    
    with open('checkpoints/preprocessing_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    
    print("\nPreprocessing complete!")
    print(f"- Processed {len(df):,} interactions in {processing_time:.2f} seconds")
    print(f"- {len(skill_to_id):,} unique skills")
    print(f"- {df['user_id'].nunique():,} unique users")
    print(f"- Overall accuracy: {df['correct'].mean():.2%}")
    print(f"- Memory used: {mem_after - mem_before:.2f} MB")
    print(f"- Compression ratio: {(file_size_mb / summary['output_file_size_mb']):.2f}x" if use_compression else "")
    print("\nOutput files:")
    print(f"- Preprocessed data: {output_file}")
    print("- Skill mappings: checkpoints/skill_mapping.json")
    print("- Preprocessing summary: checkpoints/preprocessing_summary.json")
    
    return output_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Preprocess student interaction data for knowledge tracing')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output', required=True, help='Output CSV file path')
    args = parser.parse_args()
    preprocess_data(args.input, args.output)
