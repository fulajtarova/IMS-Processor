"""This module is the core of the machine learning pipeline. It orchestrates the entire process, including loading data, preprocessing, splitting, training, tuning, and evaluating models.
It also handles predictions on new datasets.
"""

import os
import numpy as np
import random
import joblib
import shutil

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC

from backend.setup_logger import logger

from backend.load_data import read_dataset
from backend.hyperparameters import hyperparameter_tuning
from backend.evaluator import evaluate_model
from backend.splitting import split_data
from backend.load_data import read_dataset
from backend.processor import process_data
from utils import (
    find_enabled_steps,
    hyper_param_grids,
    update_bundle,
    ensure_feature_compatibility,
)


random.seed(3)
np.random.seed(3)


CLASSIFIER_MAP = {
    "Logistic Regression": LogisticRegression,
    "Support Vector Classifier": SVC,
    "Decision Tree": DecisionTreeClassifier,
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Naive Bayes": GaussianNB,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "XGBoost": XGBClassifier,
}


def run_pipeline(
    datasets: list,
    split_ratios: dict,
    classifiers_dict: dict,
    processing_config: dict,
    hyper_config: dict,
    output_dir: str,
    temp_dir: str,
):
    """
    Run the entire pipeline: load data, preprocess, split, process, train, tune, and evaluate.
    """

    # Load and preprocess datasets
    datasets_dict, _ = read_dataset(datasets, processing_config)
    param_grids = hyper_param_grids
    dataset_splits = {}
    plot_paths = []
    hyperparameters_results_file = os.path.join(
        output_dir, "hyperparameters_results.csv"
    )

    for dataset_name, (X, y) in datasets_dict.items():
        logger.info(f"Dataset: {dataset_name} | Shape: {X.shape}")
        # Split the dataset into train, validation, and test sets
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder = split_data(
            X,
            y,
            split_ratios["train"],
            split_ratios["val"],
            split_ratios["test"],
            random_state=3,
        )

        logger.info(f"X_train: {X_train[:5]}")
        logger.info(f"y_train: {y_train[:5]}")

        # Process the data (scaling, feature selection, dimensionality reduction)
        X_train, X_val, X_test, bundle_path = process_data(
            processing_config, X_train, X_val, X_test, y_train, output_dir, dataset_name
        )

        dataset_splits[dataset_name] = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "bundle_path": bundle_path,
            "label_encoder": label_encoder,
        }

    steps_written = False

    # Iterate over classifiers and datasets
    for clf_name, params in classifiers_dict.items():
        try:
            if "random_state" not in params and clf_name in [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
            ]:
                params["random_state"] = 3

            clf_class = CLASSIFIER_MAP[clf_name]

            for dataset_name in datasets_dict.keys():
                logger.info(f"Model: {clf_name}, Dataset: {dataset_name}")
                data = dataset_splits[dataset_name]

                clf_instance = clf_class(**params)

                # Handle multi-class classification for XGBoost
                if clf_name == "XGBoost":
                    num_classes = len(np.unique(data["y_train"]))
                    if num_classes > 2:
                        clf_instance.set_params(
                            objective="multi:softprob", num_class=num_classes
                        )
                    else:
                        clf_instance.set_params(objective="binary:logistic")
                if clf_name == "Support Vector Classifier":
                    clf_instance.set_params(probability=True)

                # Train initial model
                try:
                    model = clf_instance.fit(data["X_train"], data["y_train"])

                    # Update the bundle with the trained model
                    bundle_path = data["bundle_path"]
                    label_encoder = data["label_encoder"]
                    bundle_path = update_bundle(
                        bundle_path, "model", model, dataset_name, label_encoder
                    )

                except Exception as e:
                    logger.error(f"Failed to train or save model: {e}")
                    print(f"Failed to train or save model: {e}")

                # Hyperparameter tuning (if enabled)
                if hyper_config["enabled"]:
                    logger.info(
                        f"Tuning hyperparameters for {clf_name} on {dataset_name}"
                    )
                    model = hyperparameter_tuning(
                        classifier=model,
                        param_grid=param_grids[clf_name],
                        X_train=data["X_train"],
                        y_train=data["y_train"],
                        X_val=data["X_val"],
                        y_val=data["y_val"],
                        dataset_name=dataset_name,
                        results_file=hyperparameters_results_file,
                        settings=hyper_config,
                    )

                    # Update the bundle with the tuned model
                    bundle_path = update_bundle(
                        bundle_path, "tuned_model", model, clf_name
                    )

                    X_eval = data["X_test"]
                    y_eval = data["y_test"]

                # If hyperparameter tuning is not enabled, concatenate val and test sets
                else:
                    X_eval = np.concatenate([data["X_val"], data["X_test"]])
                    y_eval = np.concatenate([data["y_val"], data["y_test"]])

                # Evaluate the model
                plot_paths_new = evaluate_model(
                    model,
                    X_eval,
                    y_eval,
                    f"{clf_name} (Eval)",
                    find_enabled_steps(processing_config),
                    dataset_name,
                    output_dir,
                    temp_dir,
                    plot=True,
                    label_encoder=label_encoder,
                    write_steps=not steps_written,
                )
                plot_paths.extend(plot_paths_new)
                steps_written = True

        except Exception as e:
            logger.error(f"Failed to initialize or train {clf_name}: {e}")
            continue
    print(f"Pipeline completed for all classifiers and datasets.")

    return {
        "message": "Pipeline completed. Check logs and CSV results.",
        "plots": plot_paths,
        "results_file": output_dir,
        "bundle_path": bundle_path,
    }


def predict_and_evaluate(joblib_path: str, dataset_path: str):
    """
    Load a trained model and predict on a new dataset.
    """
    print(f"Predicting with model from: {joblib_path}")
    if not os.path.exists(joblib_path):
        print(f"Model file not found: {joblib_path}")
        return None
    if not os.path.exists(dataset_path):
        print(f"Dataset file not found: {dataset_path}")
        return None

    print(f"Loading bundle from: {joblib_path}")
    try:
        bundle = joblib.load(joblib_path)
    except Exception as e:
        print(f"Failed to load bundle: {e}")
        return None

    processing_config = bundle["bundle"]["processing_config"]

    datasets_dict, sample_names = read_dataset([dataset_path], processing_config, True)

    X, _ = datasets_dict["honey_p"]

    scaler = bundle["bundle"].get("scaler", None)
    feature_selector_filter = bundle["bundle"].get("feature_selector_filter", None)
    feature_selector_wrapper = bundle["bundle"].get("feature_selector_wrapper", None)
    dimensionality_reducer = bundle["bundle"].get("dimensionality_reducer", None)
    model = bundle["bundle"].get("model", None)
    tuned_model = bundle["bundle"].get("tuned_model", None)
    label_encoder = bundle["bundle"].get("label_encoder", None)

    if scaler:
        print(f"Transforming data with scaler: {scaler}")
        X = ensure_feature_compatibility(X, scaler, "Scaler")
        X = scaler.transform(X)
    if feature_selector_filter:
        print(
            f"Transforming data with feature selector filter: {feature_selector_filter}"
        )
        X = ensure_feature_compatibility(
            X, feature_selector_filter, "Feature Selector Filter"
        )
        X = feature_selector_filter.transform(X)
    if feature_selector_wrapper:
        print(
            f"Transforming data with feature selector wrapper: {feature_selector_wrapper}"
        )
        X = ensure_feature_compatibility(
            X, feature_selector_wrapper, "Feature Selector Wrapper"
        )
        X = feature_selector_wrapper.transform(X)
    if dimensionality_reducer:
        print(
            f"Transforming data with dimensionality reducer: {dimensionality_reducer}"
        )
        X = ensure_feature_compatibility(
            X, dimensionality_reducer, "Dimensionality Reducer"
        )
        X = dimensionality_reducer.transform(X)

    if tuned_model:
        print(f"Using tuned model: {tuned_model}")
        model = tuned_model

    if not model:
        print("No model found in the bundle.")
        return
    try:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)
        y_dec = label_encoder.inverse_transform(y_pred).tolist()
        y_dec_proba = [
            {
                label_encoder.classes_[i]: round(prob * 100, 2)
                for i, prob in enumerate(probs)
            }
            for probs in y_pred_proba
        ]
        return y_dec, y_dec_proba, sample_names
    except Exception as e:
        print(f"Prediction failed: {e}")
        return None, None
