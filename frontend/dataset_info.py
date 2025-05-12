"""This module shows basic information about the dataset.
It allows the user to upload a dataset file and displays information such as the number of samples, sample shape, class distribution, and a preview of the samples.
It also allows the user to upload a spectrum file and displays a plot of the spectrum.
"""

from nicegui import ui
from pathlib import Path
from backend.load_data import read_dataset, get_dataset_info
from backend.plotter import get_spectrum_plot_data
import asyncio

from frontend.navbar import add_navbar
from utils import clear_temp_dir

from utils import TEMP_DIR

uploaded_file_path = None
spectrum_file_path = None


def show():
    """Displays the page for showing basic information about the dataset."""
    add_navbar()
    global uploaded_file_path

    with ui.column().classes("items-center w-full max-w-4xl mx-auto pt-12 gap-4"):

        ui.label("Show Basic Info About Dataset").classes("text-2xl font-bold mb-4")

        # Dataset upload section
        ui.label("Upload a Dataset (.txt file)").classes("text-lg font-bold")
        uploaded_label = ui.label("")

        def handle_upload(e):
            """Handles the upload of a dataset file."""
            global uploaded_file_path
            for f in TEMP_DIR.glob("*"):
                f.unlink()
            uploaded_file_path = TEMP_DIR / e.name
            with open(uploaded_file_path, "wb") as f:
                f.write(e.content.read())
            uploaded_label.text = f"Uploaded: {uploaded_file_path.name}"
            ui.notify(f"File uploaded: {uploaded_file_path.name}", color="positive")

        ui.upload(label="Select a .txt File", on_upload=handle_upload).props(
            "accept=.txt"
        ).classes("w-full")

        info_container = ui.column().classes("w-full mt-4 gap-1")

        dialog = ui.dialog().props("persistent")
        with dialog, ui.card():
            dialog_label = ui.label("Processing dataset, please wait...")
            ui.spinner()

        ui.button(
            "Show Info",
            on_click=lambda: asyncio.create_task(
                show_info(info_container, dialog, dialog_label)
            ),
        ).classes("mt-2")

        # Image plot section
        ui.separator()
        ui.label("Upload a Spectrum File (.mea or .json)").classes("text-lg font-bold")
        spectrum_uploaded_label = ui.label("")

        def handle_spectrum_upload(e):
            """Handles the upload of a spectrum file."""
            global spectrum_file_path
            for f in TEMP_DIR.glob("*.mea"):
                f.unlink()
            for f in TEMP_DIR.glob("*.json"):
                f.unlink()

            spectrum_file_path = TEMP_DIR / e.name
            with open(spectrum_file_path, "wb") as f:
                f.write(e.content.read())

            spectrum_uploaded_label.text = f"Uploaded: {spectrum_file_path.name}"
            ui.notify(f"File uploaded: {spectrum_file_path.name}", color="positive")

            spectrum_container.clear()

        ui.upload(
            label="Select a .mea or .json File", on_upload=handle_spectrum_upload
        ).props("accept=.mea,.json").classes("w-full")

        spectrum_container = ui.column().classes("items-center w-full mt-4 gap-1")

        def plot_mea_sample():
            """Plots the spectrum of the uploaded file."""
            global spectrum_file_path
            if not spectrum_file_path or not spectrum_file_path.exists():
                ui.notify("Please upload a .mea or .json file first.", color="warning")
                return
            try:
                image_url = get_spectrum_plot_data(str(spectrum_file_path))
                spectrum_container.clear()
                with spectrum_container:
                    ui.image(image_url).classes("w-full max-w-3xl")

                clear_temp_dir()

            except Exception as e:
                ui.notify(f"Failed to plot file: {e}", color="negative")

        ui.button("Plot Sample", on_click=plot_mea_sample).classes("mt-2 mb-4")


async def show_info(info_container, dialog, dialog_label):
    """Asynchronously shows information about the dataset."""
    global uploaded_file_path

    if not uploaded_file_path or not uploaded_file_path.exists():
        with info_container:
            ui.notify("Please upload a file first.", color="warning")
        return

    dialog_label.text = "Processing dataset, please wait..."
    dialog.open()

    try:
        processing_config = {
            "average": {"enabled": True},
        }

        result, _ = await asyncio.to_thread(
            read_dataset, [uploaded_file_path], processing_config
        )
        if not result:
            with info_container:
                ui.notify("Failed to process dataset.", color="negative")
            return

        dataset_name, (X, y) = next(iter(result.items()))
        summary = get_dataset_info(X, y)

        info_container.clear()
        with info_container:
            ui.label(f"Dataset: {dataset_name}").classes("text-lg font-semibold")
            ui.label(f"Total Samples: {summary['samples']}")
            ui.label(f"Sample Shape: {summary['sample_shape']}")

            ui.label("Class Distribution:").classes("text-lg font-semibold mt-2")
            for cls, count in summary["class_distribution"].items():
                ui.label(f"  Class '{cls}': {count} samples")

            ui.label("Sample Preview:").classes("text-lg font-semibold mt-2")
            for i, sample in enumerate(summary["sample_preview"], 1):
                ui.label(f"Sample {i} — Class: {sample['label']}").classes(
                    "font-semibold"
                )
                s = sample["stats"]
                ui.label(
                    f"  Min: {s['min']}, Max: {s['max']}, "
                    f"Mean: {s['mean']}, Median: {s['median']}, Std: {s['std']}"
                ).classes("text-sm")

        for temp_file in TEMP_DIR.iterdir():
            if temp_file.is_file():
                temp_file.unlink()

    except Exception as e:
        with info_container:
            ui.notify(f"Error: {e}", color="negative")
    finally:
        dialog.close()
