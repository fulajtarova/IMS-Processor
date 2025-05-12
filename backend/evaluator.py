"""This module evaluates the model's performance on a given dataset and saves the results."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from backend.setup_logger import logger
from backend.plotter import plot_metrics


def evaluate_model(
    model,
    X,
    y_true,
    model_name,
    enabled_steps,
    dataset_name,
    output_dir,
    temp_dir,
    plot,
    label_encoder,
    write_steps,
):
    """
    Evaluate the model on the given dataset and save the results.
    """
    model_name = model_name.replace("(Eval)", "")
    y_pred = model.predict(X)
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    labels = list(map(int, np.unique(y_true)))

    result_data = {
        "Dataset": dataset_name,
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "macro avg_Precision": report.get("macro avg", {}).get("precision"),
        "macro avg_Recall": report.get("macro avg", {}).get("recall"),
        "macro avg_F1": report.get("macro avg", {}).get("f1-score"),
        "weighted avg_Precision": report.get("weighted avg", {}).get("precision"),
        "weighted avg_Recall": report.get("weighted avg", {}).get("recall"),
        "weighted avg_F1": report.get("weighted avg", {}).get("f1-score"),
    }

    result_data = {
        k: (round(v, 4) if isinstance(v, float) else v) for k, v in result_data.items()
    }

    result_df = pd.DataFrame([result_data])

    results_file = os.path.join(output_dir, "results.csv")

    file_exists = os.path.exists(results_file)
    is_new_file = not file_exists or os.stat(results_file).st_size == 0

    if is_new_file:
        with open(results_file, "w", encoding="utf-8") as f:
            header = ",".join(result_df.columns)
            f.write(header + "\n")

    with open(results_file, "a", encoding="utf-8") as f:
        if write_steps:
            f.write("\n")
            steps_text = str(enabled_steps)
            f.write(f'"{steps_text}"\n')

    result_df.to_csv(results_file, mode="a", header=False, index=False)

    logger.info(
        f"{model_name} on {dataset_name} - Accuracy: {accuracy:.4f}, "
        f"F1: {report['accuracy']:.4f}, "
        f"Macro F1: {report['macro avg']['f1-score']:.4f}, "
        f"Weighted F1: {report['weighted avg']['f1-score']:.4f}"
    )

    plot_paths = []
    if plot and temp_dir:
        cm_path, report_path = plot_metrics(
            y_true,
            y_pred,
            labels=labels,
            title=f"{model_name}-{dataset_name}",
            save_path=temp_dir,
            label_encoder=label_encoder,
        )
        plot_paths = [cm_path, report_path]

    return plot_paths
