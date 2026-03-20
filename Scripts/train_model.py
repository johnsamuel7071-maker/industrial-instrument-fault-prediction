# scripts/train_model.py

import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from utils_preprocessing import (
    get_data_paths,
    get_outputs_dir,
    inspect_data,
    load_datasets,
    preprocess_train_test,
    save_artifact,
)

warnings.filterwarnings("ignore")


def evaluate_model(model_name, model, X_train, y_train, X_test, y_test, label_encoder):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_weighted")

    report = classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "model": model,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "cv_mean_f1": cv_scores.mean(),
        "cv_std_f1": cv_scores.std(),
        "classification_report": report,
    }


def save_confusion_matrix(y_true, y_pred, class_names, filename):
    cm = confusion_matrix(y_true, y_pred)
    output_dir = get_outputs_dir()

    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    path = os.path.join(output_dir, filename)
    plt.savefig(path)
    plt.close()
    print(f"Saved confusion matrix: {path}")


def save_feature_importance(model, feature_names, csv_name, img_name):
    if not hasattr(model, "feature_importances_"):
        print("Best model does not support feature importance.")
        return

    output_dir = get_outputs_dir()
    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values(by="importance", ascending=False)

    csv_path = os.path.join(output_dir, csv_name)
    importance_df.to_csv(csv_path, index=False)
    print(f"Saved feature importance CSV: {csv_path}")

    top_n = min(15, len(importance_df))
    plot_df = importance_df.head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["feature"], plot_df["importance"])
    plt.gca().invert_yaxis()
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()

    img_path = os.path.join(output_dir, img_name)
    plt.savefig(img_path)
    plt.close()
    print(f"Saved feature importance image: {img_path}")


def save_evaluation_report(results):
    output_dir = get_outputs_dir()
    report_path = os.path.join(output_dir, "evaluation_report.txt")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Industrial Instrument Fault Prediction - Evaluation Report\n")
        f.write("=" * 70 + "\n\n")

        for result in results:
            f.write(f"Model: {result['model_name']}\n")
            f.write(f"Accuracy: {result['accuracy']:.4f}\n")
            f.write(f"Precision: {result['precision']:.4f}\n")
            f.write(f"Recall: {result['recall']:.4f}\n")
            f.write(f"F1 Score: {result['f1_score']:.4f}\n")
            f.write(f"CV Mean F1: {result['cv_mean_f1']:.4f}\n")
            f.write(f"CV Std F1: {result['cv_std_f1']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(result["classification_report"])
            f.write("\n" + "=" * 70 + "\n\n")

    print(f"Saved evaluation report: {report_path}")


def main():
    print("Starting model training...")

    print("\nLeakage prevention:")
    print(
        "Dropping 'error_code' because it may directly reveal the target class. "
        "Using it would cause data leakage and unrealistic model performance."
    )

    train_path, test_path = get_data_paths()
    train_df, test_df = load_datasets(train_path, test_path)
    inspect_data(train_df, test_df)

    processed = preprocess_train_test(
        train_df=train_df,
        test_df=test_df,
        target_col="fault_status",
        leakage_columns=["error_code"],
    )

    X_train = processed["X_train"]
    X_test = processed["X_test"]
    y_train = processed["y_train_encoded"]
    y_test = processed["y_test_encoded"]
    label_encoder = processed["label_encoder"]

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=3,
            random_state=42,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
    }

    results = []

    for model_name, model in models.items():
        print(f"\n=== Training {model_name} ===")
        result = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            label_encoder=label_encoder,
        )

        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"Precision: {result['precision']:.4f}")
        print(f"Recall: {result['recall']:.4f}")
        print(f"F1 Score: {result['f1_score']:.4f}")
        print(f"CV Mean F1: {result['cv_mean_f1']:.4f}")
        print("\nClassification Report:")
        print(result["classification_report"])

        print("Confusion Matrix:")
        print(confusion_matrix(y_test, result["y_pred"]))

        results.append(result)

    best_result = max(results, key=lambda x: x["f1_score"])
    print(f"\nBest model selected: {best_result['model_name']}")

    save_confusion_matrix(
        y_true=y_test,
        y_pred=best_result["y_pred"],
        class_names=label_encoder.classes_,
        filename="confusion_matrix.png",
    )

    save_feature_importance(
        model=best_result["model"],
        feature_names=processed["selected_feature_columns"],
        csv_name="feature_importance.csv",
        img_name="feature_importance.png",
    )

    save_evaluation_report(results)

    save_artifact(best_result["model"], "best_model.pkl")
    save_artifact(processed["label_encoder"], "label_encoder.pkl")
    save_artifact(processed["feature_encoders"], "feature_encoders.pkl")
    save_artifact(processed["scaler"], "scaler.pkl")
    save_artifact(processed["selected_feature_columns"], "selected_feature_columns.pkl")
    save_artifact(processed["categorical_cols"], "categorical_cols.pkl")
    save_artifact(processed["numeric_cols"], "numeric_cols.pkl")

    print("\nTraining complete.")


if __name__ == "__main__":
    main()