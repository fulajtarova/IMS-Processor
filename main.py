"""
This module serves as the entry point for the ML Pipeline GUI application.

It sets up the web application using NiceGUI and defines the main pages of the application.
"""

from nicegui import ui
from frontend import (
    home,
    dataset_info,
    predict,
    custom,
    results_custom,
    results_predict,
)
import shutil
import pathlib
from utils import clear_temp_dir


def setup_gui():
    """
    Registers all GUI pages and their corresponding route handlers.
    This function should be called only when running the app.
    """

    @ui.page("/")
    def landing():
        """Landing page of the application."""
        clear_temp_dir()
        home.show()

    @ui.page("/dataset-info")
    def dataset_info_page():
        """Page to show basic information about the dataset."""
        dataset_info.show()

    @ui.page("/custom")
    def custom_page():
        """Page to build a custom ML pipeline."""
        custom.show()

    @ui.page("/results_custom")
    def results_custom_page():
        """Page to show the results of the custom ML pipeline."""
        results_custom.show()

    @ui.page("/predict")
    def predict_page():
        """Page to make predictions using the trained model."""
        predict.show()

    @ui.page("/results_predict")
    def results_predict_page():
        """Page to show the results of the predictions."""
        clear_temp_dir()
        results_predict.show()


if __name__ in {"__main__", "__mp_main__"}:
    pycache_path = pathlib.Path(__file__).parent / "__pycache__"
    if pycache_path.exists():
        shutil.rmtree(pycache_path)

    setup_gui()
    ui.run(title="ML Pipeline GUI", reload=False)
