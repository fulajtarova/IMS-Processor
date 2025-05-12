"""This module shows results of the custom ML pipeline.
It displays the evaluation metrics and plots generated during the pipeline execution."""

from nicegui import ui
import os
import pandas as pd
from frontend.navbar import add_navbar
from frontend.custom import results_data


def show():
    """Displays the results of the custom ML pipeline."""
    add_navbar()
    with ui.column().classes("items-center w-full max-w-full mx-auto pt-12 gap-4"):

        ui.label("Pipeline Results").classes("text-2xl font-bold mb-4")

        ui.label(results_data.get("message", "No message")).classes("text-sm")

        # Show CSV results table if available
        results_file = results_data.get("results_file")
        if results_file and os.path.exists(results_file):
            df = pd.read_csv(results_file)
            if not df.empty:
                ui.label("Evaluation Metrics").classes("text-lg font-semibold")
                ui.table(
                    columns=[
                        {"name": col, "label": col, "field": col} for col in df.columns
                    ],
                    rows=df.to_dict("records"),
                    row_key="Model",
                ).classes("w-full")

        # Show plots side by side
        if results_data.get("plots"):
            ui.label("Evaluation Plots").classes("text-lg font-semibold")

            for i in range(0, len(results_data["plots"]), 2):
                with ui.row().classes("justify-center w-full gap-6 mb-4"):
                    for j in range(2):
                        if i + j < len(results_data["plots"]):
                            plot_path = results_data["plots"][i + j]
                            if os.path.exists(plot_path):
                                ui.image(plot_path).classes(
                                    "max-w-[600px] w-full rounded shadow"
                                )
        else:
            ui.label("No plots to display.")
