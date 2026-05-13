# Bitcoin Price Forecasting with Custom RNN and LSTM

A simple deep learning project that compares two sequence models (custom RNN and custom LSTM) for Bitcoin price forecasting.

## Project Goal

- Download and prepare historical Bitcoin data
- Build time-series sequences with a lookback window
- Train custom RNN and custom LSTM models
- Compare model quality and inference speed

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook
- yfinance

## Project Structure

```
.
|-- config.yaml                        # Main configuration (data split, training, model units)
|-- requirements.txt                   # Python dependencies
|-- src/                               # Reusable source code
|   |-- custom_cells.py                # Custom RNN/LSTM cell implementations
|   |-- data_loader.py                 # Bitcoin data download and missing-value checks
|   |-- preprocessing.py               # Scaling, sequence creation, dataset split, save/load arrays
|   |-- tf_dataset_loader.py           # tf.data Dataset pipeline
|   |-- models.py                      # Model architectures
|   `-- training.py                    # Training logic and callbacks
|-- notebooks/
|   |-- 01_data_preparation.ipynb
|   |-- 02_model_training.ipynb
|   `-- 03_model_comparison.ipynb
|-- data/
|   |-- raw/                           # Raw downloaded data
|   `-- processed/                     # Numpy arrays for train/valid/test
|-- models/                            # Saved trained models
`-- results/                           # Training history, comparison tables, and plots
```

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open and run notebooks in order:

- `notebooks/01_data_preparation.ipynb`
- `notebooks/02_model_training.ipynb`
- `notebooks/03_model_comparison.ipynb`

## Configuration

Main settings are in `config.yaml`, including:

- `DATA.lookback_window`
- `DATA.train_split` and `DATA.validation_split`
- `TRAINING.batch_size`, `TRAINING.epochs`, `TRAINING.early_stopping_patience`
- `MODELS.rnn_units`, `MODELS.lstm_units`, `MODELS.dropout`

## Current Comparison Snapshot

From `results/model_comparison/comparison_metrics.csv`:

- MAE: RNN = 5597.97, LSTM = 6615.25
- RMSE: RNN = 7571.66, LSTM = 7580.31
- MAPE: RNN = 6.95%, LSTM = 9.18%
- Prediction Time: RNN = 1.52s, LSTM = 1.64s

## Notes

- This project currently uses `BTC-USD` from Yahoo Finance (`yfinance`).
- Preprocessed arrays and trained model files are already included in this workspace.
- For reproducible experiments, keep notebook execution order and config values consistent.
