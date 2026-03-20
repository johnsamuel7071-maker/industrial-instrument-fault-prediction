# scripts/predict_single.py

import pandas as pd

from utils_preprocessing import load_artifact, preprocess_new_data


def main():
    best_model = load_artifact("best_model.pkl")
    label_encoder = load_artifact("label_encoder.pkl")
    feature_encoders = load_artifact("feature_encoders.pkl")
    scaler = load_artifact("scaler.pkl")
    selected_feature_columns = load_artifact("selected_feature_columns.pkl")
    categorical_cols = load_artifact("categorical_cols.pkl")
    numeric_cols = load_artifact("numeric_cols.pkl")

    sample = {
        "timestamp": "2025-05-14 10:30:00",
        "instrument_id": "PT101",
        "instrument_type": "Pressure Transmitter",
        "process_variable": 48.5,
        "setpoint": 50.0,
        "measurement_error": -1.5,
        "signal_noise": 0.18,
        "drift_rate": 0.04,
        "ambient_temperature": 31.5,
        "humidity": 62.0,
        "vibration_level": 0.35,
        "power_supply_voltage": 24.1,
        "calibration_age_days": 45,
        "maintenance_overdue_days": 0,
        "operating_pressure": 42.0,
        "operating_temperature": 120.0,
        "process_load_percent": 68.0,
        "response_time_ms": 320.0,
        "error_code": 0,
    }

    sample_df = pd.DataFrame([sample])

    try:
        processed_sample = preprocess_new_data(
            df=sample_df,
            feature_encoders=feature_encoders,
            scaler=scaler,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            selected_feature_columns=selected_feature_columns,
            leakage_columns=["error_code"],
        )

        prediction = best_model.predict(processed_sample)
        probabilities = best_model.predict_proba(processed_sample)[0]
        predicted_label = label_encoder.inverse_transform(prediction)[0]

        print("\n=== SINGLE PREDICTION RESULT ===")
        print("Predicted Fault Status:", predicted_label)
        print("\nClass Probabilities:")
        for class_name, prob in zip(label_encoder.classes_, probabilities):
            print(f"{class_name}: {prob:.4f}")

    except Exception as error:
        print(f"Prediction failed: {error}")


if __name__ == "__main__":
    main()