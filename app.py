import pandas as pd
import streamlit as st
from Scripts.utils_preprocessing import load_artifact, preprocess_new_data

st.set_page_config(page_title="Industrial Instrument Fault Prediction", layout="wide")

@st.cache_resource
def load_resources():
    best_model = load_artifact("best_model.pkl")
    label_encoder = load_artifact("label_encoder.pkl")
    feature_encoders = load_artifact("feature_encoders.pkl")
    scaler = load_artifact("scaler.pkl")
    selected_feature_columns = load_artifact("selected_feature_columns.pkl")
    categorical_cols = load_artifact("categorical_cols.pkl")
    numeric_cols = load_artifact("numeric_cols.pkl")
    return (
        best_model,
        label_encoder,
        feature_encoders,
        scaler,
        selected_feature_columns,
        categorical_cols,
        numeric_cols,
    )

st.title("Industrial Instrument Fault Prediction")
st.write("Predict instrument health: Normal, Warning, or Faulty")

(
    best_model,
    label_encoder,
    feature_encoders,
    scaler,
    selected_feature_columns,
    categorical_cols,
    numeric_cols,
) = load_resources()

col1, col2 = st.columns(2)

with col1:
    timestamp = st.text_input("Timestamp", "2025-05-14 10:30:00")
    instrument_id = st.text_input("Instrument ID", "PT101")
    instrument_type = st.selectbox(
        "Instrument Type",
        ["Pressure Transmitter", "Temperature Transmitter", "Flow Meter", "Level Transmitter"]
    )
    process_variable = st.number_input("Process Variable", value=48.5)
    setpoint = st.number_input("Setpoint", value=50.0)
    measurement_error = st.number_input("Measurement Error", value=-1.5)
    signal_noise = st.number_input("Signal Noise", value=0.18)
    drift_rate = st.number_input("Drift Rate", value=0.04)
    ambient_temperature = st.number_input("Ambient Temperature", value=31.5)
    humidity = st.number_input("Humidity", value=62.0)

with col2:
    vibration_level = st.number_input("Vibration Level", value=0.35)
    power_supply_voltage = st.number_input("Power Supply Voltage", value=24.1)
    calibration_age_days = st.number_input("Calibration Age Days", value=45, step=1)
    maintenance_overdue_days = st.number_input("Maintenance Overdue Days", value=0, step=1)
    operating_pressure = st.number_input("Operating Pressure", value=42.0)
    operating_temperature = st.number_input("Operating Temperature", value=120.0)
    process_load_percent = st.number_input("Process Load Percent", value=68.0)
    response_time_ms = st.number_input("Response Time ms", value=320.0)
    error_code = st.number_input("Error Code", value=0, step=1)

if st.button("Predict"):
    try:
        sample = pd.DataFrame([{
            "timestamp": timestamp,
            "instrument_id": instrument_id,
            "instrument_type": instrument_type,
            "process_variable": process_variable,
            "setpoint": setpoint,
            "measurement_error": measurement_error,
            "signal_noise": signal_noise,
            "drift_rate": drift_rate,
            "ambient_temperature": ambient_temperature,
            "humidity": humidity,
            "vibration_level": vibration_level,
            "power_supply_voltage": power_supply_voltage,
            "calibration_age_days": calibration_age_days,
            "maintenance_overdue_days": maintenance_overdue_days,
            "operating_pressure": operating_pressure,
            "operating_temperature": operating_temperature,
            "process_load_percent": process_load_percent,
            "response_time_ms": response_time_ms,
            "error_code": error_code,
        }])

        processed = preprocess_new_data(
            df=sample,
            feature_encoders=feature_encoders,
            scaler=scaler,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            selected_feature_columns=selected_feature_columns,
            leakage_columns=["error_code"],
        )

        pred = best_model.predict(processed)
        probs = best_model.predict_proba(processed)[0]
        label = label_encoder.inverse_transform(pred)[0]

        st.success(f"Predicted Fault Status: {label}")
        prob_df = pd.DataFrame({
            "Class": label_encoder.classes_,
            "Probability": probs
        })
        st.dataframe(prob_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")