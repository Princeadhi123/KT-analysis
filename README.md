# Knowledge Tracing Model with Principal Axis Factoring

This project implements a Deep Knowledge Tracing (DKT) model using PyTorch and PyTorch Lightning. The model tracks students' knowledge states based on their interaction data and predicts their future performance. 
**A key feature of this implementation is the use of Principal Axis Factoring (PAF) as a preprocessing step to derive meaningful skill representations (embeddings) from the interaction data before training the DKT model.**

## Features
- LSTM-based DKT model
- **Advanced data preprocessing including Principal Axis Factoring (PAF) for skill representation**
- Efficient data loading and batching
- Modular, minimal codebase
- Model checkpointing and validation

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/KT-analysis.git
   cd KT-analysis
   ```
2. **Set up a virtual environment (recommended):**
   ```bash
   python -m venv venv
   # Windows
   .\venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preprocessing
Prepare your student interaction data as a CSV with columns:
- `user_id`: Unique student identifier
- `skill_id`: Original skill identifier (e.g., problem ID)
- `correct`: 0 (incorrect) or 1 (correct)
- `timestamp`: Timestamp of the interaction

Run the preprocessing script:
```bash
python preprocess_data.py --input path/to/your/input_data.csv --output path/to/your/processed_data.csv
```
This script performs several key operations:
- Basic data cleaning and formatting.
- **Principal Axis Factoring (PAF):**
    - Constructs a user-skill interaction matrix from the input data.
    - Applies PAF to this matrix to derive latent factors representing underlying skill dimensions. The number of factors is currently set to 10 (within `preprocess_data.py`).
    - The original skill IDs are then mapped to these factor-based representations.
- Outputs the main processed data file (e.g., `processed_data.csv`) where `skill_id` now refers to an index for the factor embeddings.
- **Key PAF-related outputs are saved in the `checkpoints/paf_outputs/` directory:**
    - `skill_factor_embeddings.npy`: A NumPy array containing the factor loadings for each original skill (shape: `[num_original_skills, num_factors]`). These are used as the initial weights for the DKT model's skill embedding layer.
    - `original_skill_id_to_factor_idx.json`: A JSON mapping from the original skill IDs to their corresponding row index in `skill_factor_embeddings.npy`.
    - `factor_idx_to_original_skill_id.json`: The reverse mapping.
    - `paf_model.joblib`: The saved scikit-learn PAF model object.
- A summary of the preprocessing, `checkpoints/preprocessing_summary.json`, is also generated.

### 2. Model Training
Train the DKT model on the processed data (which now includes factor-based skill IDs):
```bash
python main.py
```
Key aspects of the training process with PAF integration:
- **DKT Model (`models.py`):**
    - The DKT model is initialized with parameters derived from the PAF step:
        - `num_outputs`: The number of unique original skills (obtained from the shape of PAF embeddings).
        - `embedding_dim`: The dimensionality of the PAF skill embeddings (number of factors).
        - `skill_embeddings_path`: Path to the `skill_factor_embeddings.npy` file.
    - The DKT model's skill embedding layer is initialized using these pre-computed PAF factor loadings. These embeddings are set to be **fine-tuned** during the DKT training process.
- **Training Script (`main.py`):**
    - `main.py` reads the dimensions (number of skills and factor dimensionality) from the saved `skill_factor_embeddings.npy`.
    - It then passes these dimensions and the path to the embeddings file to the DKT model during its initialization.
This will train the model and save the best checkpoint (e.g., `best_model.ckpt`) in a subdirectory within `checkpoints/` (e.g., `checkpoints/model_checkpoints/`). Training logs are saved in `checkpoints/training_logs/`.

## Project Structure
```
KT-analysis/
├── data/                  # Raw and processed data files
├── checkpoints/           # Root directory for saved outputs
│   ├── paf_outputs/       # Outputs from the PAF preprocessing step
│   │   ├── skill_factor_embeddings.npy
│   │   ├── original_skill_id_to_factor_idx.json
│   │   ├── factor_idx_to_original_skill_id.json
│   │   └── paf_model.joblib
│   ├── model_checkpoints/ # Saved DKT model checkpoints
│   └── training_logs/     # Logs from PyTorch Lightning trainer
├── preprocess_data.py     # Data preprocessing script (includes PAF)
├── main.py                # Training script for DKT
├── models.py              # DKT model definition (adapted for PAF embeddings)
├── data.py                # PyTorch Lightning DataModule
├── config.py              # Configuration settings (model, training, paths)
├── utils.py               # Utility functions (if any)
├── requirements.txt       # Project dependencies
├── tests/                 # Unit and integration tests
│   └── test_paf_integration.py # Tests for PAF integration
└── README.md              # This file
```

## Requirements
- Python 3.8+
- PyTorch 1.9+ (or compatible)
- PyTorch Lightning 1.5+ (or compatible)
- pandas
- numpy
- scikit-learn (for FactorAnalysis and other utilities)
- tqdm
- joblib (for saving scikit-learn models)

## License
MIT License
