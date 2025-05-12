"""This module is the home page of the IMS Processor application.
It provides the main interface for users to navigate through the application.
The home page includes buttons to access different functionalities of the application,
such as showing basic information about the dataset, building a custom ML pipeline,
and making predictions using a trained model.
"""

from nicegui import ui
from frontend.navbar import add_navbar


def show():
    """Displays the home page of the IMS Processor application."""
    add_navbar()

    with ui.column().classes("items-center w-full max-w-4xl mx-auto pt-12 mt-4 gap-5"):
        ui.label("IMS Processor").classes("text-3xl font-bold mb-6")

        with ui.column().classes("w-full max-w-sm gap-3"):
            ui.button(
                "1. Show Basic Info About Dataset",
                on_click=lambda: ui.navigate.to("/dataset-info"),
            ).classes("w-full py-4 text-lg mb-2")

            ui.button(
                "2. Build Your Custom ML Pipeline",
                on_click=lambda: ui.navigate.to("/custom"),
            ).classes("w-full py-4 text-lg mb-2")

            ui.button(
                "3. Predict Using Trained Model",
                on_click=lambda: ui.navigate.to("/predict"),
            ).classes("w-full py-4 text-lg mb-2")
