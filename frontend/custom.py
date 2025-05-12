"""This module creates a custom ML pipeline using NiceGUI.
It allows users to upload datasets, select preprocessing methods,
choose classifiers, and perform hyperparameter tuning.
"""

from nicegui import ui
import asyncio
from copy import deepcopy


from backend.pipeline import run_pipeline
from utils import (
    smoothing_methods,
    dimensionality_reduction_methods,
    transformation_methods,
    feature_selection_filters,
    feature_selection_wrappers_estimators,
    available_classifiers,
    hyperparameter_methods,
    hyperparameter_cv_strategies,
    hyperparameter_scoring,
    classifiers_dict_all_default,
    TEMP_DIR,
    RESULTS_DIR,
)
from frontend.navbar import add_navbar

results_data = {"message": "", "plots": []}
uploaded_files = []
selected_classifiers = {}
parameter_sections = {}
classifier_checkboxes = {}


def show():
    """Displays the page for building a custom ML pipeline."""
    add_navbar()

    uploaded_files.clear()
    selected_classifiers.clear()
    parameter_sections.clear()
    classifier_checkboxes.clear()

    with ui.column().classes("items-center w-full max-w-4xl mx-auto pt-12 gap-4"):

        ui.label("Machine Learning Pipeline Builder").classes("text-2xl font-bold mb-4")

        # Upload .txt config files
        with ui.card().classes("w-full p-4"):
            ui.label("Step 1: Upload Config Files (.txt)").classes("text-lg font-bold")

            upload_list = ui.column().classes("w-full mt-2")

            def refresh_upload_list():
                """Refreshes the list of uploaded files."""
                upload_list.clear()
                for file_path in uploaded_files:
                    with upload_list:
                        with ui.row().classes("w-full items-center"):
                            with ui.button(
                                on_click=lambda p=file_path: remove_file(p)
                            ).classes("bg-transparent p-1"):
                                ui.icon("delete").classes("text-red-500 text-sm")
                            ui.label(file_path.name)

            def handle_upload(e):
                """Handles the upload of a dataset file."""
                path = TEMP_DIR / e.name

                with open(path, "wb") as f:
                    f.write(e.content.read())

                if any(f.name == path.name for f in uploaded_files):
                    ui.notify(
                        f"File '{path.name}' already exists in the list.",
                        color="warning",
                    )
                    return

                uploaded_files.append(path)
                ui.notify(f"Uploaded: {path.name}", color="positive")
                refresh_upload_list()

            def remove_file(path):
                """Removes a file from the uploaded files list."""
                if path.exists():
                    path.unlink()
                if path in uploaded_files:
                    uploaded_files.remove(path)
                ui.notify(f"Removed: {path.name}", color="negative")
                refresh_upload_list()

            ui.upload(on_upload=handle_upload, multiple=True).props(
                "accept=.txt"
            ).classes("w-full")

        # Processing Options
        with ui.card().classes("w-full p-4"):
            ui.label("Step 2: Preprocessing").classes("text-lg font-bold")
            ui.label(
                "You can select one or multiple preprocessing methods below. "
                "Processing will be applied in the listed order.\n"
                "Note: 'Spectral Average' is always enabled by default."
            ).classes("text-sm text-gray-600")
            # RIP removal
            global rip_remove, rr_start, rr_end
            rip_remove = ui.checkbox("Remove RIP")
            rr_start = create_number("RIP Start", 1.05)
            rr_end = create_number("RIP End", 2.00)
            rr_start.bind_visibility_from(rip_remove, "value")
            rr_end.bind_visibility_from(rip_remove, "value")

            # RIP relative
            global rip_relative
            rip_relative = ui.checkbox("Relative RIP")

            # Cutoff
            global cutoff_enabled, cutoff_percentage, cutoff_value
            cutoff_enabled = ui.checkbox("Enable Cutoff")

            cutoff_percentage = create_number("Cut Below % of Max", 0.2)
            cutoff_value = create_number("Cutoff Value (optional)", None)

            cutoff_hint = ui.label(
                "- 'Cutoff Value': Removes values below a fixed numeric threshold.\n"
                "- 'Cut Below % of Max': Removes values below a percentage of the maximum value.\n"
                "- If both are set, 'Cutoff Value' is used. Cutoff below % is applied if both are empty."
            ).classes("text-xs text-gray-500 whitespace-pre-wrap")

            cutoff_percentage.bind_visibility_from(cutoff_enabled, "value")
            cutoff_value.bind_visibility_from(cutoff_enabled, "value")
            cutoff_hint.bind_visibility_from(cutoff_enabled, "value")

            # Smoothing
            global smoothing_enabled, smoothing_method, smoothing_size
            smoothing_enabled = ui.checkbox("Enable Smoothing")
            smoothing_method = ui.select(
                smoothing_methods, label="Smoothing Method", value="savitzky_golay"
            ).classes("w-full")
            smoothing_size = create_number("Size (optional)", None)

            smoothing_hint = ui.label(
                "Recommended smoothing sizes:\n"
                "- Savitzky-Golay: 7, 9, 11 (odd numbers; default: 9)\n"
                "- Uniform Filter: 3, 5, 7 (default: 5)\n"
                "- Gaussian Blur: 1.0–3.0 (default: 2.0)"
            ).classes("text-xs text-gray-500 whitespace-pre-wrap")

            smoothing_method.bind_visibility_from(smoothing_enabled, "value")
            smoothing_size.bind_visibility_from(smoothing_enabled, "value")
            smoothing_hint.bind_visibility_from(smoothing_enabled, "value")

            # Scaling
            global scaling_enabled, scaling_method
            scaling_enabled = ui.checkbox("Enable Scaling")
            scaling_method = ui.select(
                transformation_methods,
                label="Scaling Method",
                value="zscore",
            ).classes("w-full")
            scaling_method.bind_visibility_from(scaling_enabled, "value")

            # FS filter
            global fs_filter_enabled, fs_method, fs_threshold
            fs_filter_enabled = ui.checkbox("Enable Feature Selection: Filter")

            fs_method = ui.select(
                feature_selection_filters, label="Method", value="mutual_info"
            ).classes("w-full")
            fs_threshold = create_number("Threshold", 0.2).classes("w-full")
            fs_hint = ui.label(
                "Feature selection threshold:\n"
                "- 'variance_threshold': Use a float (default: 1e-5)\n"
                "- 'mutual_info': Use a float < 1 to select a fraction of features (e.g., 0.2 = 20%), or an integer to select a fixed number of top features (e.g., 10). Default: 0.2"
            ).classes("text-xs text-gray-500 whitespace-pre-wrap")

            fs_method.bind_visibility_from(fs_filter_enabled, "value")
            fs_threshold.bind_visibility_from(fs_filter_enabled, "value")
            fs_hint.bind_visibility_from(fs_filter_enabled, "value")

            # FS wrapper
            global fs_wrapper_enabled, fs_estimator, fs_n_features, fs_step
            fs_wrapper_enabled = ui.checkbox("Enable Feature Selection: Wrapper (RFE)")

            fs_estimator = create_select(
                "Estimator",
                feature_selection_wrappers_estimators,
                value="Support Vector Classifier",
            ).classes("w-full")

            fs_n_features = create_number("Number of Features to Select", 0.2).classes(
                "w-full"
            )
            fs_step = create_number("Step Size", 0.05).classes("w-full")
            fs_wrapper_hint = ui.label(
                "- 'Number of Features to Select': Use a float < 1 to select a fraction of features (e.g., 0.2 = 20%), or an integer to select an exact number. Default: 0.2 (20%).\n"
                "- 'Step Size': Proportion (float < 1) or count (int) of features to remove per iteration. Default: 0.05 (5%)."
            ).classes("text-xs text-gray-500 whitespace-pre-wrap")

            fs_estimator.bind_visibility_from(fs_wrapper_enabled, "value")
            fs_n_features.bind_visibility_from(fs_wrapper_enabled, "value")
            fs_step.bind_visibility_from(fs_wrapper_enabled, "value")
            fs_wrapper_hint.bind_visibility_from(fs_wrapper_enabled, "value")

            # Dimensionality Reduction
            global dr_enabled, dr_method, dr_n_components
            dr_enabled = ui.checkbox("Enable Dimensionality Reduction")

            dr_method = ui.select(
                dimensionality_reduction_methods,
                label="Method",
                value="LDA",
            ).classes("w-full")

            dr_n_components = create_number("Number of Components (optional)", None)

            dr_hint = ui.label(
                "Number of components:\n"
                "- PCA: Use a float < 1 to retain that percentage of variance (e.g., 0.95 = 95%), or an integer to select a fixed number of components. Default: 0.95 (95% variance).\n"
                "- LDA: Maximum allowed is (number of classes - 1). Default: automatically calculated.\n"
                "- PCA-LDA: PCA keeps 95% variance by default, followed by LDA using the max allowed components."
            ).classes("text-xs text-gray-500 whitespace-pre-wrap")

            dr_method.bind_visibility_from(dr_enabled, "value")
            dr_n_components.bind_visibility_from(dr_enabled, "value")
            dr_hint.bind_visibility_from(dr_enabled, "value")

        # Data Split
        with ui.card().classes("w-full p-4"):
            ui.label("Step 3: Data Split (%)").classes("text-lg font-bold")
            global train_split, val_split, test_split
            with ui.row():
                train_split = ui.number("Train (%)", value=80)
                val_split = ui.number("Val (%)", value=10)
                test_split = ui.number("Test (%)", value=10)

        # Classifier Selection
        with ui.card().classes("w-full p-4"):
            ui.label("Step 4: Select Classifiers").classes("text-lg font-bold")

            def make_toggle_callback(classifier_name):
                """Creates a callback function for toggling classifiers."""

                def toggle(e):

                    toggle_classifier(classifier_name, e.value)

                return toggle

            for name in available_classifiers:
                checkbox = ui.checkbox(name, on_change=make_toggle_callback(name))
                classifier_checkboxes[name] = checkbox

            with ui.expansion("Customize Parameters (Optional)").classes("w-full"):
                global parameters_container
                parameters_container = ui.column().classes("w-full gap-2")

        # Hyperparameter Tuning
        with ui.card().classes("w-full p-4"):
            ui.label("Step 5: Hyperparameter Tuning").classes("text-lg font-bold")

            global hyper_enabled, method_select, strategy_select, folds_input, iter_input, scoring_select, verbose_input

            hyper_enabled = ui.checkbox("Enable Tuning")

            method_select = create_select(
                label="Search Method",
                options=hyperparameter_methods,
                value="random_search",
            )

            strategy_select = create_select(
                label="CV Strategy",
                options=hyperparameter_cv_strategies,
                value="stratified",
            )

            folds_input = create_number("CV Folds", value=5)
            iter_input = create_number("Random Search Iterations", value=10)

            scoring_select = create_select(
                label="Scoring Metric",
                options=hyperparameter_scoring,
                value="f1_macro",
            )

            iter_input.bind_visibility_from(
                method_select, "value", lambda v: v == "random_search"
            )

            for el in [
                method_select,
                strategy_select,
                folds_input,
                iter_input,
                scoring_select,
            ]:
                el.bind_visibility_from(hyper_enabled, "value")

        with ui.row().classes("justify-center mt-4 gap-4"):
            ui.button("Run Pipeline", on_click=start_pipeline)


def toggle_classifier(name, enabled):
    """Toggles the visibility of classifier parameters."""
    if not enabled:
        selected_classifiers.pop(name, None)
        if name in parameter_sections:
            parameter_sections[name].clear()
            parameter_sections[name].delete()
            del parameter_sections[name]
        return

    with parameters_container:
        section = ui.column().classes("w-full border p-2")
        parameter_sections[name] = section
        with section:
            ui.label(f"{name} Parameters")

            default_params = deepcopy(classifiers_dict_all_default.get(name, {}))
            params = {}

            def add_param(key, field):
                """Adds a parameter field to the section."""
                params[key] = field

            # LOGISTIC REGRESSION
            default_params = deepcopy(classifiers_dict_all_default.get(name, {}))
            params = {}

            # LOGISTIC REGRESSION
            if name == "Logistic Regression":
                add_param("C", create_number("C", default_params.get("C", 1.0)))
                ui.label(
                    "Inverse of regularization strength. Smaller values specify stronger regularization."
                ).classes("text-xs text-gray-500")

                add_param(
                    "penalty",
                    create_select(
                        "Penalty",
                        ["l2", "l1", "elasticnet", None],
                        value=default_params.get("penalty", "l2"),
                    ),
                )
                ui.label(
                    "Used to specify the norm used in the penalization. 'None' is a no penalty."
                ).classes("text-xs text-gray-500")

                add_param(
                    "solver",
                    create_select(
                        "Solver",
                        ["lbfgs", "liblinear", "saga", "newton-cg"],
                        value=default_params.get("solver", "lbfgs"),
                    ),
                )
                ui.label(
                    "Algorithm to use in the optimization problem. 'liblinear' is a good choice for small datasets."
                    "Solver compatibility:\n"
                    "- 'lbfgs', 'saga', 'newton-cg': support binary and multinomial (multiclass)\n"
                    "- 'liblinear': only supports binary or one-vs-rest (OvR), not multinomial\n"
                    "- 'saga': required for elasticnet penalty\n"
                ).classes("text-xs text-gray-500 whitespace-pre-wrap")

                add_param(
                    "max_iter",
                    create_number("Max Iter", default_params.get("max_iter", 100)),
                )
                ui.label(
                    "Maximum number of iterations for the solver to converge."
                ).classes("text-xs text-gray-500")

            # SUPPORT VECTOR
            elif name == "Support Vector Classifier":
                add_param("C", create_number("C", default_params.get("C")))
                ui.label(
                    "Regularization parameter. The strength of the regularization is inversely proportional to C."
                ).classes("text-xs text-gray-500")
                add_param(
                    "kernel",
                    create_select(
                        "Kernel",
                        ["linear", "poly", "rbf", "sigmoid"],
                        value=default_params.get("kernel"),
                    ),
                )
                ui.label(
                    "Specifies the kernel type to be used in the algorithm."
                ).classes("text-xs text-gray-500")

                add_param(
                    "degree",
                    create_number("Degree", default_params.get("degree")).classes(
                        "w-full"
                    ),
                )
                ui.label(
                    "Degree of the polynomial kernel function ('poly'). Ignored by all other kernels."
                ).classes("text-xs text-gray-500")

                add_param(
                    "gamma",
                    create_select(
                        "Gamma",
                        ["scale", "auto"],
                        value=default_params.get("gamma"),
                    ),
                )
                ui.label(
                    "Kernel coefficient for 'rbf', 'poly', and 'sigmoid'."
                ).classes("text-xs text-gray-500")

            # DECISION TREE
            elif name == "Decision Tree":
                add_param(
                    "criterion",
                    create_select(
                        "Criterion",
                        ["gini", "entropy"],
                        value=default_params.get("criterion"),
                    ),
                )
                ui.label("Function to measure the quality of a split.").classes(
                    "text-xs text-gray-500"
                )

                add_param(
                    "max_depth",
                    create_number("Max Depth", default_params.get("max_depth")).classes(
                        "w-full"
                    ),
                )
                ui.label(
                    "The maximum depth of the tree. If None, nodes are expanded until all leaves are pure."
                ).classes("text-xs text-gray-500")

                add_param(
                    "min_samples_split",
                    create_number(
                        "Min Samples Split",
                        default_params.get("min_samples_split"),
                    ).classes("w-full"),
                )
                ui.label(
                    "The minimum number of samples required to split an internal node."
                ).classes("text-xs text-gray-500")

                add_param(
                    "min_samples_leaf",
                    create_number(
                        "Min Samples Leaf",
                        default_params.get("min_samples_leaf"),
                    ).classes("w-full"),
                )
                ui.label(
                    "The minimum number of samples required to be at a leaf node."
                ).classes("text-xs text-gray-500")

                add_param(
                    "max_features",
                    create_select(
                        "Max Features",
                        [None, "sqrt", "log2"],
                        default_params.get("max_features"),
                    ),
                )
                ui.label(
                    "The number of features to consider when looking for the best split."
                ).classes("text-xs text-gray-500")

            # K-NEAREST NEIGHBORS
            elif name == "K-Nearest Neighbors":
                add_param(
                    "n_neighbors",
                    create_number(
                        "N Neighbors", default_params.get("n_neighbors")
                    ).classes("w-full"),
                )
                ui.label("Number of neighbors to use for kneighbors queries.").classes(
                    "text-xs text-gray-500"
                )

                add_param(
                    "weights",
                    create_select(
                        "Weights",
                        ["uniform", "distance"],
                        value=default_params.get("weights"),
                    ),
                )
                ui.label(
                    "Weight function used in prediction. 'uniform' weights all points equally."
                ).classes("text-xs text-gray-500")
                add_param(
                    "algorithm",
                    create_select(
                        "Algorithm",
                        ["auto", "ball_tree", "kd_tree", "brute"],
                        value=default_params.get("algorithm"),
                    ),
                )
                ui.label(
                    "Algorithm used to compute the nearest neighbors:\n"
                    "- 'auto': automatically decides the best algorithm based on the input data.\n"
                    "- 'ball_tree': uses BallTree algorithm.\n"
                    "- 'kd_tree': uses KDTree algorithm.\n"
                    "- 'brute': brute-force search."
                ).classes("text-xs text-gray-500 whitespace-pre-wrap")
                add_param(
                    "leaf_size",
                    create_number("Leaf Size", default_params.get("leaf_size")).classes(
                        "w-full"
                    ),
                )
                ui.label(
                    "Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as the memory required to store the tree."
                ).classes("text-xs text-gray-500")

                add_param(
                    "metric",
                    create_select(
                        "Metric",
                        [
                            "euclidean",
                            "manhattan",
                            "minkowski",
                        ],
                        value=default_params.get("metric"),
                    ),
                )

            # NAIVE BAYES
            elif name == "Naive Bayes":
                add_param(
                    "var_smoothing",
                    create_number(
                        "Var Smoothing",
                        default_params.get("var_smoothing"),
                    ).classes("w-full"),
                )
                ui.label(
                    "Variance smoothing parameter. This is added to the variance of each feature to prevent division by zero."
                ).classes("text-xs text-gray-500")

            # RANDOM FOREST
            elif name == "Random Forest":
                add_param(
                    "n_estimators",
                    create_number(
                        "N Estimators", default_params.get("n_estimators")
                    ).classes("w-full"),
                )
                ui.label("The number of trees in the forest. Default is 100.").classes(
                    "text-xs text-gray-500"
                )

                add_param(
                    "max_depth",
                    create_number("Max Depth", default_params.get("max_depth")).classes(
                        "w-full"
                    ),
                )
                ui.label(
                    "The maximum depth of the tree. If None, nodes are expanded until all leaves are pure."
                ).classes("text-xs text-gray-500")

                add_param(
                    "min_samples_split",
                    create_number(
                        "Min Samples Split",
                        default_params.get("min_samples_split"),
                    ).classes("w-full"),
                )
                ui.label(
                    "The minimum number of samples required to split an internal node."
                ).classes("text-xs text-gray-500")
                add_param(
                    "min_samples_leaf",
                    create_number(
                        "Min Samples Leaf",
                        default_params.get("min_samples_leaf"),
                    ).classes("w-full"),
                )
                ui.label(
                    "The minimum number of samples required to be at a leaf node."
                ).classes("text-xs text-gray-500")
                add_param(
                    "max_features",
                    create_select(
                        "Max Features",
                        [None, "sqrt", "log2"],
                        value=default_params.get("max_features"),
                    ),
                )
                ui.label(
                    "The number of features to consider when looking for the best split."
                ).classes("text-xs text-gray-500")

            # GRADIENT BOOSTING
            elif name == "Gradient Boosting":
                add_param(
                    "n_estimators",
                    create_number(
                        "N Estimators", default_params.get("n_estimators")
                    ).classes("w-full"),
                )
                ui.label(
                    "The number of boosting stages to be run. Default is 100."
                ).classes("text-xs text-gray-500")

                add_param(
                    "learning_rate",
                    create_number(
                        "Learning Rate",
                        default_params.get("learning_rate"),
                    ).classes("w-full"),
                )
                ui.label(
                    "Learning rate shrinks the contribution of each tree. Default is 0.1."
                ).classes("text-xs text-gray-500")
                add_param(
                    "max_depth",
                    create_number("Max Depth", default_params.get("max_depth")).classes(
                        "w-full"
                    ),
                )
                ui.label(
                    "The maximum depth of the individual regression estimators. Default is 3."
                ).classes("text-xs text-gray-500")
                add_param(
                    "subsample",
                    create_number(
                        "Subsample",
                        default_params.get("subsample"),
                    ).classes("w-full"),
                )
                ui.label(
                    "The fraction of samples to be used for fitting the individual base learners. Default is 1.0 (use all samples)."
                ).classes("text-xs text-gray-500")
                add_param(
                    "loss",
                    create_select(
                        "Loss",
                        ["log_loss", "exponential"],
                        value=default_params.get("loss"),
                    ),
                )
                ui.label(
                    "The loss function to be optimized. 'log_loss' is the default."
                ).classes("text-xs text-gray-500")

            # XGBOOST
            elif name == "XGBoost":
                add_param(
                    "n_estimators",
                    create_number(
                        "Boosting Rounds", default_params.get("n_estimators")
                    ),
                )
                ui.label(
                    "Number of boosting rounds. Default is 100. If None, will use the default value."
                ).classes("text-xs text-gray-500")
                add_param(
                    "max_depth",
                    create_number("Max Depth", default_params.get("max_depth")),
                )
                ui.label(
                    "Maximum depth of a tree. Default is 6. If None, will use the default value."
                ).classes("text-xs text-gray-500")
                add_param(
                    "learning_rate",
                    create_number("Learning Rate", default_params.get("learning_rate")),
                )
                ui.label(
                    "Learning rate. Default is 0.3. If None, will use the default value."
                ).classes("text-xs text-gray-500")

            selected_classifiers[name] = {
                "params": params,
                "default_params": default_params,
            }


def create_number(label, value):
    """Creates a number input field with a label and a default value."""
    parse_as = float if isinstance(value, float) else int
    field = ui.input(label=label, value="" if value is None else str(value)).classes(
        "w-full"
    )

    def parse():
        """Parses the input value and returns it as a number."""
        val = field.value.strip()
        if not val:
            return None
        try:
            return parse_as(val)
        except ValueError:
            ui.notify(f"Invalid input for {label}", color="negative")
            raise

    field.get_parsed_value = parse
    return field


def create_select(label, options, value=None):
    """Creates a select input field with a label and a default value."""
    options = [str(opt) if opt is not None else "None" for opt in options]
    value = str(value) if value is not None else "None"

    field = ui.select(options, label=label, value=value).classes("w-full")

    def parse():
        """Parses the input value and returns it."""
        return None if field.value == "None" else field.value

    field.get_parsed_value = parse
    return field


async def start_pipeline():
    """Asynchronously starts the ML pipeline."""
    if not uploaded_files:
        ui.notify("Upload at least one dataset")
        return

    total = train_split.value + val_split.value + test_split.value

    if total != 100:
        ui.notify("Train/Val/Test must sum to 100")
        return

    if not selected_classifiers:
        ui.notify("Select at least one classifier")
        return

    classifiers = {}

    for name, meta in selected_classifiers.items():
        parsed_params = {}
        defaults = meta.get("default_params", {})
        widgets = meta.get("params", {})

        for k in defaults:
            if k in widgets:
                w = widgets[k]
                val = w.get_parsed_value() if hasattr(w, "get_parsed_value") else w

                raw_val = getattr(w, "value", "")
                if (
                    val is None
                    and isinstance(raw_val, str)
                    and raw_val.strip() not in ("", "None")
                ):
                    ui.notify(f"Invalid value for '{k}' in {name}", color="negative")
                    return

                parsed_params[k] = val
            else:
                parsed_params[k] = defaults[k]

        classifiers[name] = parsed_params

    processing_config = {
        "rip_remove": {
            "enabled": rip_remove.value,
            "cut_start": rr_start.get_parsed_value(),
            "cut_end": rr_end.get_parsed_value(),
        },
        "rip_rel": {
            "enabled": rip_relative.value,
        },
        "cutoff": {
            "enabled": cutoff_enabled.value,
            "keep_percentage": cutoff_percentage.get_parsed_value(),
            "cutoff_value": cutoff_value.get_parsed_value(),
        },
        "smoothing": {
            "enabled": smoothing_enabled.value,
            "method": smoothing_method.value,
            "size": smoothing_size.get_parsed_value(),
        },
        "average": {"enabled": True},
        "transformation": {
            "enabled": scaling_enabled.value,
            "method": scaling_method.value,
        },
        "feature_selection_filter": {
            "enabled": fs_filter_enabled.value,
            "method": fs_method.value,
            "threshold": fs_threshold.get_parsed_value(),
        },
        "feature_selection_wrapper": {
            "enabled": fs_wrapper_enabled.value,
            "estimator": fs_estimator.get_parsed_value(),
            "n_features": fs_n_features.get_parsed_value(),
        },
        "dimensionality_reduction": {
            "enabled": dr_enabled.value,
            "method": dr_method.value,
            "n_components": dr_n_components.get_parsed_value(),
        },
    }

    hyper_config = {
        "enabled": hyper_enabled.value,
        "method": method_select.value,
        "cv_strategy": strategy_select.value,
        "cv_folds": folds_input.get_parsed_value(),
        "scoring": scoring_select.value,
    }
    if method_select.value == "random_search":
        hyper_config["n_iter"] = iter_input.get_parsed_value()

    split_ratios = {
        "train": train_split.value / 100,
        "val": val_split.value / 100,
        "test": test_split.value / 100,
    }

    with ui.dialog() as dialog, ui.card():
        ui.label("Running pipeline...")
        ui.spinner()
        dialog.open()

        try:
            result = await asyncio.to_thread(
                run_pipeline,
                uploaded_files,
                split_ratios,
                classifiers,
                processing_config,
                hyper_config,
                str(RESULTS_DIR),
                str(TEMP_DIR),
            )
        except Exception as e:
            result = {"message": f"Pipeline failed: {e}", "plots": []}
        dialog.close()

    results_data["message"] = result["message"]
    ui.notify(result["message"])

    results_data["plots"] = result.get("plots", [])
    ui.navigate.to("/results_custom")
