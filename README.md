# Knowledge Tracing Model

This project implements a Deep Knowledge Tracing (DKT) model using PyTorch and PyTorch Lightning. The model tracks students' knowledge states based on their interaction data and predicts their future performance.

## Features
- LSTM-based DKT model
- Efficient data preprocessing
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
- `skill_id`: Skill identifier
- `correct`: 0 (incorrect) or 1 (correct)
- `timestamp`: Timestamp of the interaction

Run the preprocessing script:
```bash
python preprocess_data.py --input data/student_log_2.csv --output data/processed_data.csv
```

### 2. Model Training
Train the DKT model on the processed data:
```bash
python main.py
```
This will train the model and save the best checkpoint in the `checkpoints/` directory.

## Project Structure
```
KT-analysis/
├── data/                  # Data files
├── checkpoints/           # Model checkpoints
├── preprocess_data.py     # Data preprocessing script
├── main.py                # Training script
├── models.py              # DKT model definition
├── data.py                # Data module
├── config.py              # Configuration
├── utils.py               # Utility functions
├── requirements.txt       # Dependencies
└── README.md              # This file
```

## Requirements
- Python 3.8+
- PyTorch 1.9+
- PyTorch Lightning 1.5+
- pandas
- numpy
- tqdm

## License
MIT License
