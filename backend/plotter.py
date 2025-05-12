"""This module plots evaluation metrics for machine learning models, including confusion matrices and classification reports."""

import os
from io import BytesIO
import base64
import ims
import json
import datetime
import re
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


import seaborn as sns
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)


def plot_metrics(y_true, y_pred, labels, title, save_path, label_encoder):
    """
    Plot confusion matrix and classification report as heatmap.
    Saves both images to disk and ensures resources are closed properly.
    """

    safe_title = re.sub(r"[^\w\-\.]", "", title).replace("_", "")
    os.makedirs(save_path, exist_ok=True)

    try:
        y_pred_display = label_encoder.inverse_transform(y_pred)
        y_true_display = label_encoder.inverse_transform(y_true)
        display_labels = label_encoder.classes_

        cm = confusion_matrix(y_true_display, y_pred_display, labels=display_labels)
        cm_path = os.path.join(save_path, f"{safe_title}-confusion.png")

        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=display_labels
        )
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
        ax_cm.set_title(f"{safe_title} - Confusion Matrix")
        fig_cm.tight_layout()
        fig_cm.savefig(cm_path)
        plt.close(fig_cm)

    except Exception as e:
        print(f"[Error] Failed to save confusion matrix: {e}")
        cm_path = None

    report_path = os.path.join(save_path, f"{safe_title}-report.png")

    try:
        report_dict = classification_report(
            y_true_display, y_pred_display, labels=display_labels, output_dict=True
        )
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.drop(index=["accuracy"], errors="ignore", inplace=True)
        report_df.drop(columns=["support"], errors="ignore", inplace=True)

        fig_cr, ax_cr = plt.subplots(figsize=(7, max(4, len(report_df) * 0.6)))
        sns.heatmap(
            report_df,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=False,
            ax=ax_cr,
            linewidths=0.5,
            linecolor="white",
        )
        ax_cr.set_title(f"{safe_title} - Classification Report")
        fig_cr.tight_layout()
        fig_cr.savefig(report_path)
        plt.close(fig_cr)

    except Exception as e:
        print(f"[Error] Failed to save classification report: {e}")
        report_path = None

    return cm_path, report_path


# Function to plot the spectrum
def get_spectrum_plot_data(file_path: str):
    """
    Read the spectrum data from a .MEA or .json file and plot it.
    """
    if not file_path or not os.path.exists(file_path):
        raise ValueError("Invalid file path provided.")

    if file_path.endswith(".mea"):
        spectrum = ims.Spectrum.read_mea(file_path)

    elif file_path.endswith(".json"):
        with open(file_path, "r") as json_file:
            entry = json.load(json_file)

        spectrum = ims.Spectrum(
            name=entry["name"],
            values=np.array(entry["values"], dtype=np.float32),
            ret_time=np.array(entry["ret_time"], dtype=np.float32),
            drift_time=np.array(entry["drift_time"], dtype=np.float32)[::-1],
            time=datetime.datetime.now(),
        )

    result = spectrum.plot()
    fig = result[0] if isinstance(result, tuple) else result
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"
