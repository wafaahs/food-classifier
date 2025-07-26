import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def load_predictions(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at: {csv_path}")
    return pd.read_csv(csv_path)


def compute_accuracy(df):
    correct = (df["prediction"] == df["ground_truth"]).sum()
    total = len(df)
    return correct / total if total > 0 else 0


def plot_confusion_matrix(df, output_dir):
    y_true = df["ground_truth"]
    y_pred = df["prediction"]
    labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()
    return cm_path


def generate_report(df, output_dir):
    report_dict = classification_report(
        df["ground_truth"], df["prediction"], output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_csv_path = os.path.join(output_dir, "classification_report.csv")
    report_df.to_csv(report_csv_path)
    return report_df, report_csv_path


def plot_per_class_confusion(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    classes = sorted(df["ground_truth"].unique())

    for cls in classes:
        # Create a 2x2 matrix for binary classification: this class vs all others
        df["binary_true"] = df["ground_truth"].apply(lambda x: cls if x == cls else f"not_{cls}")
        df["binary_pred"] = df["prediction"].apply(lambda x: cls if x == cls else f"not_{cls}")

        cm = confusion_matrix(df["binary_true"], df["binary_pred"], labels=[cls, f"not_{cls}"])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
                    xticklabels=[cls, f"not_{cls}"], yticklabels=[cls, f"not_{cls}"])
        plt.title(f"Binary Confusion Matrix for: {cls}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"confusion_{cls}.png")
        plt.savefig(plot_path)
        plt.close()

        df.drop(columns=["binary_true", "binary_pred"], inplace=True)


def plot_metrics_bar_chart(report_df, output_dir):
    filtered = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore")
    metrics = filtered[["precision", "recall", "f1-score"]]
    metrics.plot(kind="bar", figsize=(14, 6))
    plt.title("Per-Class Metrics")
    plt.xlabel("Class")
    plt.ylabel("Score")
    plt.xticks(rotation=90)
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "class_metrics_bar_chart.png")
    plt.savefig(chart_path)
    plt.close()
    return chart_path


def write_summary_file(accuracy, report_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Overall Accuracy: {accuracy * 100:.2f}%\n\n")
        f.write("Per-Class Precision, Recall, F1-score:\n")
        filtered = report_df.drop(index=["accuracy", "macro avg", "weighted avg"], errors="ignore").copy()
        f.write(filtered.to_string())

        worst_classes = filtered.sort_values("f1-score").head(5)
        f.write("\n\nWorst 5 Classes by F1-score:\n")
        f.write(worst_classes.to_string())

    return summary_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", required=True, help="Model subdir name (e.g. e10_b64_adam_lr0p001)")
    args = parser.parse_args()

    input_csv = os.path.join("output", args.subdir, "predictions.csv")
    output_dir = os.path.join("analysis", args.subdir)
    os.makedirs(output_dir, exist_ok=True)

    df = load_predictions(input_csv)
    accuracy = compute_accuracy(df)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    report_df, report_path = generate_report(df, output_dir)
    print(report_df)
    print(f"Report saved to: {report_path}")

    cm_path = plot_confusion_matrix(df, output_dir)
    print(f"Confusion matrix saved to: {cm_path}")

    plot_per_class_confusion(df, output_dir)
    print("Per-class 2x2 confusion matrices saved.")

    chart_path = plot_metrics_bar_chart(report_df, output_dir)
    print(f"Metrics bar chart saved to: {chart_path}")

    summary_path = write_summary_file(accuracy, report_df, output_dir)
    print(f"Summary saved to: {summary_path}")
