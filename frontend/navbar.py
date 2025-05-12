"""This module sets up the navigation bar for the IMS Processor application.
It provides a consistent header across all pages of the application,
allowing users to navigate back to the home page easily.
The navigation bar includes the application title and a home button.
"""

from nicegui import ui


def add_navbar():
    """Adds a navigation bar to the top of the page."""
    with ui.header().classes("bg-primary text-white"):
        with ui.row().classes("items-center gap-4"):
            with ui.button(on_click=lambda: ui.navigate.to("/")).props("flat").classes(
                "text-white"
            ):
                ui.icon("home")
            ui.label("IMS PROCESSOR").classes("text-xl font-bold")
