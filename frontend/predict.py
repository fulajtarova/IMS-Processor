"""This module handles the prediction functionality of the IMS Processor application.
It allows users to upload a configuration file and a trained model file,
and then makes predictions using the uploaded model on the specified dataset.
"""

from nicegui import ui
from backend.pipeline import predict_and_evaluate
import asyncio

from frontend.results_predict import predictions_data
from frontend.navbar import add_navbar
from utils import TEMP_DIR


def show():
    """Displays the prediction page of the IMS Processor application."""
    add_navbar()
    uploaded_file_path = None
    joblib_uploaded_file_path = None

    with ui.column().classes("items-center w-full max-w-4xl mx-auto pt-12 gap-4"):

        ui.label("Predict Using Trained Model").classes("text-2xl font-bold")

        # Upload .txt config file
        ui.label("Upload a .txt config file for dataset").classes("text-lg font-bold")
        uploaded_label = ui.label("").classes("text-gray-600")

        def handle_upload(e):
            """Handles the upload of a configuration file."""
            nonlocal uploaded_file_path
            if uploaded_file_path and uploaded_file_path.exists():
                uploaded_file_path.unlink()
            uploaded_file_path = TEMP_DIR / e.name
            with open(uploaded_file_path, "wb") as f:
                f.write(e.content.read())
            uploaded_label.text = f"Uploaded: {uploaded_file_path.name}"
            ui.notify(f"File uploaded: {uploaded_file_path.name}", color="positive")

        ui.upload(label="Select a .txt File", on_upload=handle_upload).props(
            "accept=.txt"
        ).classes("w-full")

        # Upload .joblib model file
        ui.label("Upload a .joblib file for model").classes("text-lg font-bold mt-4")
        joblib_uploaded_label = ui.label("").classes("text-gray-600")

        def handle_joblib_upload(e):
            """Handles the upload of a .joblib model file."""
            nonlocal joblib_uploaded_file_path
            if joblib_uploaded_file_path and joblib_uploaded_file_path.exists():
                joblib_uploaded_file_path.unlink()
            joblib_uploaded_file_path = TEMP_DIR / e.name
            with open(joblib_uploaded_file_path, "wb") as f:
                f.write(e.content.read())
            joblib_uploaded_label.text = f"Uploaded: {joblib_uploaded_file_path.name}"
            ui.notify(
                f"File uploaded: {joblib_uploaded_file_path.name}", color="positive"
            )

        ui.upload(label="Select a .joblib File", on_upload=handle_joblib_upload).props(
            "accept=.joblib"
        ).classes("w-full")

        async def predict():
            """Handles the prediction process."""
            if uploaded_file_path and joblib_uploaded_file_path:
                predict_button.disable()
                ui.notify("Predicting...", color="positive")
                try:
                    y_dec, y_dec_proba, sample_names = await asyncio.to_thread(
                        predict_and_evaluate,
                        str(joblib_uploaded_file_path),
                        str(uploaded_file_path),
                    )
                    if (
                        y_dec is None
                        or y_dec_proba is None
                        or len(y_dec) == 0
                        or len(y_dec_proba) == 0
                    ):
                        ui.notify(
                            "Prediction failed. Please check your inputs.",
                            color="negative",
                        )
                        predict_button.enable()
                        return

                    ui.notify("Prediction completed!", color="positive")

                    predictions_data["predictions"] = y_dec
                    predictions_data["probabilities"] = y_dec_proba
                    predictions_data["sample_names"] = sample_names

                    ui.navigate.to("/results_predict")

                except Exception as e:
                    ui.notify(f"Error during prediction: {str(e)}", color="negative")
                finally:
                    predict_button.enable()
            else:
                ui.notify("Please upload both files.", color="negative")

        predict_button = ui.button("Predict", on_click=predict).classes("mt-4")
