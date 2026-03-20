# scripts/evaluate_model.py

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from utils_preprocessing import (
    get_data_paths,
    load_artifact,
    load_datasets,
    preprocess_train_test,
)


def main():
    print("Evaluating saved model...")

    train_path, test_path = get_data_paths()
    train_df, test_df = load_datasets(train_path, test_path)

    processed = preprocess_train_test(
        train_df=train_df,
        test_df=test_df,
        target_col="fault_status",
        leakage_columns=["error_code"],
    )

    X_test = processed["X_test"]
    y_test = processed["y_test_encoded"]

    best_model = load_artifact("best_model.pkl")
    label_encoder = load_artifact("label_encoder.pkl")

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    print("\n=== SAVED MODEL EVALUATION ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=label_encoder.classes_,
            zero_division=0,
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if __name__ == "__main__":
    main()