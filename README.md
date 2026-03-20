# README.md

# Industrial Instrument Fault Prediction

A machine learning project for predicting the health status of industrial instruments in a process plant.

## Target Classes
- Normal
- Warning
- Faulty

## Instrument Types
- Pressure Transmitter
- Temperature Transmitter
- Flow Meter
- Level Transmitter

## Dataset Files
Put these files inside the `data/` folder:
- `industrial_fault_training_dataset.csv`
- `industrial_fault_testing_dataset.csv`

## Important Leakage Note
The column `error_code` is removed from model training because it can directly reveal the target class. In many industrial datasets, error codes are generated after fault detection logic or alarm conditions, so keeping it as an input feature would cause data leakage and unrealistically high accuracy.

## Project Files
- `scripts/train_model.py` — trains and saves the best model
- `scripts/evaluate_model.py` — evaluates the saved model
- `scripts/predict_single.py` — predicts one manual sample
- `scripts/predict_batch.py` — predicts a CSV batch file
- `scripts/utils_preprocessing.py` — shared preprocessing utilities

## Setup

### 1. Create virtual environment
```bash
python -m venv venv