# Industrial Instrument Fault Prediction

This project is a machine learning system for predicting the health condition of industrial instruments used in a process plant.

## Project Goal

The goal of this project is to classify industrial instrument health into:
- Normal
- Warning
- Faulty

The system is designed for instruments such as:
- Pressure Transmitters
- Temperature Transmitters
- Flow Meters
- Level Transmitters

## Dataset Files

The project uses:
- `industrial_fault_training_dataset.csv`
- `industrial_fault_testing_dataset.csv`

## Input Features

The dataset contains the following features:
- timestamp
- instrument_id
- instrument_type
- process_variable
- setpoint
- measurement_error
- signal_noise
- drift_rate
- ambient_temperature
- humidity
- vibration_level
- power_supply_voltage
- calibration_age_days
- maintenance_overdue_days
- operating_pressure
- operating_temperature
- process_load_percent
- response_time_ms
- error_code
- fault_status

## Data Leakage Prevention

The `error_code` column was removed from model training.

Reason:
`error_code` can directly reveal the fault condition because normal records often have `0`, while warning and faulty records usually have non-zero values. Keeping it in training would cause data leakage and make the model performance unrealistically high.

## Machine Learning Workflow

The project performs the following steps:
- data loading
- data inspection
- missing value check
- duplicate check
- timestamp feature engineering
- categorical encoding
- numeric scaling
- model training
- model comparison
- model evaluation
- best model selection
- model saving
- single prediction
- batch prediction

## Models Trained

The following models were trained and compared:
- Random Forest
- Gradient Boosting
- Logistic Regression

## Best Model

Gradient Boosting was selected as the best-performing model.

## Project Structure

```text
industrial_instrument_fault_prediction/
│
├── data/
│   ├── industrial_fault_training_dataset.csv
│   └── industrial_fault_testing_dataset.csv
│
├── models/
│
├── outputs/
│
├── Scripts/
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_single.py
│   ├── predict_batch.py
│   └── utils_preprocessing.py
│
├── images/
├── requirements.txt
├── README.md
└── .gitignore