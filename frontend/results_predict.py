"""This module shows results of predictions made using a trained model.
It displays the classification results, including the predicted labels and their probabilities.
"""

from nicegui import ui
from frontend.navbar import add_navbar

predictions_data = {
    "predictions": [],
    "probabilities": [],
    "sample_names": [],
}


def show():
    """Displays the results of predictions made using a trained model."""
    add_navbar()
    predictions = predictions_data["predictions"]
    probabilities = predictions_data["probabilities"]
    sample_names = predictions_data["sample_names"]

    with ui.column().classes("items-center w-full max-w-4xl mx-auto pt-12 gap-6"):
        ui.label("Classification Results").classes("text-2xl font-bold mb-4")

        for idx, (sample_name, label, prob_dict) in enumerate(
            zip(sample_names, predictions, probabilities), start=1
        ):
            with ui.column().classes("w-full items-center gap-2"):
                ui.label(f"Sample {idx}: {sample_name}").classes("text-xl font-bold")

                main_prob = prob_dict.get(label, 0)
                ui.label(f"Prediction: {label} ({main_prob:.2f}%)").classes(
                    "text-lg font-semibold"
                )

                labels = list(prob_dict.keys())
                values = list(prob_dict.values())
                ui.echart(
                    {
                        "tooltip": {"trigger": "item", "formatter": "{b}: {d}%"},
                        "series": [
                            {
                                "type": "pie",
                                "radius": ["40%", "70%"],
                                "avoidLabelOverlap": False,
                                "label": {
                                    "show": True,
                                    "formatter": "{b|{b}}\n{d}%",
                                    "fontSize": 12,
                                    "overflow": "truncate",
                                    "rich": {
                                        "b": {
                                            "width": 100,
                                            "overflow": "break",
                                        }
                                    },
                                },
                                "labelLine": {"show": True},
                                "data": [
                                    {"value": v, "name": k}
                                    for k, v in zip(labels, values)
                                ],
                            }
                        ],
                    }
                ).classes("w-80 h-80")
