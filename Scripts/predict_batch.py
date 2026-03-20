import os
import pandas as pd

from utils_preprocessing import (
    get_outputs_dir,
    load_artifact,
    preprocess_new_data,
)


def main():
    best_model = load_artifact("best_model.pkl")
    label_encoder = load_artifact("label_encoder.pkl")
    feature_encoders = load_artifact("feature_encoders.pkl")
    scaler = load_artifact("scaler.pkl")
    selected_feature_columns = load_artifact("selected_feature_columns.pkl")
    categorical_cols = load_artifact("categorical_cols.pkl")
    numeric_cols = load_artifact("numeric_cols.pkl")

    batch_file = input("Enter path to batch CSV file: ").strip()

    if not os.path.exists(batch_file):
        print(f"File not found: {batch_file}")
        return

    try:
        raw_df = pd.read_csv(batch_file)

        processed_df = preprocess_new_data(
            df=raw_df,
            feature_encoders=feature_encoders,
            scaler=scaler,
            categorical_cols=categorical_cols,
            numeric_cols=numeric_cols,
            selected_feature_columns=selected_feature_columns,
            leakage_columns=["error_code"],
        )

        predictions = best_model.predict(processed_df)
        probabilities = best_model.predict_proba(processed_df)

        result_df = raw_df.copy()
        result_df["predicted_fault_status"] = label_encoder.inverse_transform(predictions)

        for i, class_name in enumerate(label_encoder.classes_):
            result_df[f"prob_{class_name}"] = probabilities[:, i]

        output_path = os.path.join(get_outputs_dir(), "batch_predictions.csv")
        result_df.to_csv(output_path, index=False)

        print(f"\nBatch prediction completed. Saved to: {output_path}")
        print(result_df.head())

    except Exception as error:
        print(f"Batch prediction failed: {error}")


if __name__ == "__main__":
    main()