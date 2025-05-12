"""This modele contains utility functions and constants for the ML Pipeline GUI application.
It includes functions for clearing temporary directories, saving and updating processing bundles,
"""

from backend.setup_logger import logger
import hashlib
import dill
import os
from pathlib import Path
import numpy as np
import pandas as pd

# Path organization
TEMP_DIR = Path("temp").resolve()
TEMP_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("results").resolve()
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Function to clear the temporary directory
def clear_temp_dir():
    """
    Clear the temporary directory by removing all files.
    """
    for temp_file in TEMP_DIR.iterdir():
        if temp_file.is_file():
            temp_file.unlink()


BASE_CONFIG = {
    "rip_remove": {"enabled": False, "cut_start": None, "cut_end": None},
    "rip_rel": {"enabled": False},
    "cutoff": {"enabled": False, "cut_percentage": None, "cutoff_value": None},
    "smoothing": {"enabled": False, "method": None, "size": None},
    "average": {"enabled": True},
    "transformation": {"enabled": False, "method": None},
    "feature_selection_filter": {
        "enabled": False,
        "method": None,
        "threshold": None,
    },
    "feature_selection_wrapper": {
        "enabled": False,
        "estimator": None,
        "n_features": None,
        "step": None,
    },
    "dimensionality_reduction": {
        "enabled": False,
        "method": None,
        "n_components": None,
    },
}

BASE_HYPER = {
    "enabled": False,
    "method": "random_search",
    "cv_strategy": "stratified",
    "cv_folds": 5,
    "scoring": "accuracy",
    "n_iter": 10,
}


# Smoothing methods
smoothing_methods = [
    "uniform_filter",
    "gaussian_blurring",
    "savitzky_golay",
]

# Scaling methods
transformation_methods = [
    "min_max",
    "max_abs",
    "zscore",
    "robust",
]

# Feature selection filters
feature_selection_filters = ["variance_threshold", "mutual_info"]

# Feature selection wrappers
feature_selection_wrappers_estimators = [
    "Logistic Regression",
    "Random Forest",
    "Gradient Boosting",
    "Support Vector Classifier",
]

# Dimensionality reduction methods
dimensionality_reduction_methods = ["PCA", "LDA", "PCA-LDA"]


# Algorithms for classification
available_classifiers = [
    "Logistic Regression",
    "Support Vector Classifier",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Naive Bayes",
    "Random Forest",
    "Gradient Boosting",
    "XGBoost",
]

# Hyperparameter tuning methods
hyperparameter_methods = ["grid_search", "random_search"]
hyperparameter_cv_strategies = ["stratified", "kfold"]
hyperparameter_scoring = [
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "roc_auc_ovr",
]


# Default parameters for classifiers
lr_default_params = {
    "C": 1.0,
    "class_weight": None,
    "dual": False,
    "fit_intercept": True,
    "intercept_scaling": 1,
    "l1_ratio": None,
    "max_iter": 100,
    "multi_class": "auto",
    "n_jobs": None,
    "penalty": "l2",
    "random_state": None,
    "solver": "lbfgs",
    "tol": 0.0001,
    "verbose": 0,
    "warm_start": False,
}

svc_default_params = {
    "C": 1.0,
    "break_ties": False,
    "cache_size": 200,
    "class_weight": None,
    "coef0": 0.0,
    "decision_function_shape": "ovr",
    "degree": 3,
    "gamma": "scale",
    "kernel": "rbf",
    "max_iter": -1,
    "probability": True,
    "random_state": None,
    "shrinking": True,
    "tol": 0.001,
    "verbose": False,
}


dt_default_params = {
    "ccp_alpha": 0.0,
    "class_weight": None,
    "criterion": "gini",
    "max_depth": None,
    "max_features": None,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": None,
    "random_state": None,
    "splitter": "best",
}

knn_default_params = {
    "algorithm": "auto",
    "leaf_size": 30,
    "metric": "minkowski",
    "metric_params": None,
    "n_jobs": None,
    "n_neighbors": 5,
    "p": 2,
    "weights": "uniform",
}

nb_default_params = {"priors": None, "var_smoothing": 1e-09}

rf_default_params = {
    "bootstrap": True,
    "ccp_alpha": 0.0,
    "class_weight": None,
    "criterion": "gini",
    "max_depth": None,
    "max_features": "sqrt",
    "max_leaf_nodes": None,
    "max_samples": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "monotonic_cst": None,
    "n_estimators": 100,
    "n_jobs": None,
    "oob_score": False,
    "random_state": None,
    "verbose": 0,
    "warm_start": False,
}

gb_default_params = {
    "ccp_alpha": 0.0,
    "criterion": "friedman_mse",
    "init": None,
    "learning_rate": 0.1,
    "loss": "log_loss",
    "max_depth": 3,
    "max_features": None,
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0.0,
    "min_samples_leaf": 1,
    "min_samples_split": 2,
    "min_weight_fraction_leaf": 0.0,
    "n_estimators": 100,
    "n_iter_no_change": None,
    "random_state": None,
    "subsample": 1.0,
    "tol": 0.0001,
    "validation_fraction": 0.1,
    "verbose": 0,
    "warm_start": False,
}

xgb_default_params = {
    "objective": "binary:logistic",
    "base_score": None,
    "booster": None,
    "callbacks": None,
    "colsample_bylevel": None,
    "colsample_bynode": None,
    "colsample_bytree": None,
    "device": None,
    "early_stopping_rounds": None,
    "enable_categorical": False,
    "eval_metric": None,
    "feature_types": None,
    "gamma": None,
    "grow_policy": None,
    "importance_type": None,
    "interaction_constraints": None,
    "learning_rate": None,
    "max_bin": None,
    "max_cat_threshold": None,
    "max_cat_to_onehot": None,
    "max_delta_step": None,
    "max_depth": None,
    "max_leaves": None,
    "min_child_weight": None,
    "monotone_constraints": None,
    "multi_strategy": None,
    "n_estimators": None,
    "n_jobs": None,
    "num_parallel_tree": None,
    "random_state": None,
    "reg_alpha": None,
    "reg_lambda": None,
    "sampling_method": None,
    "scale_pos_weight": None,
    "subsample": None,
    "tree_method": None,
    "validate_parameters": None,
    "verbosity": None,
}

classifiers_dict_all_default = {
    "Logistic Regression": lr_default_params,
    "Support Vector Classifier": svc_default_params,
    "Decision Tree": dt_default_params,
    "K-Nearest Neighbors": knn_default_params,
    "Naive Bayes": nb_default_params,
    "Random Forest": rf_default_params,
    "Gradient Boosting": gb_default_params,
    "XGBoost": xgb_default_params,
}


classifiers_dict_best = {
    "Logistic Regression": lr_default_params,
    "Random Forest": rf_default_params,
    "XGBoost": xgb_default_params,
}

# Hyperparameter tuning grids
param_grid_logistic_regression = [
    {
        "solver": ["lbfgs"],
        "penalty": ["l2"],
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [200, 500, 1000],
    },
    {
        "solver": ["saga"],
        "penalty": ["l1", "l2", "elasticnet"],
        "C": [0.01, 0.1, 1, 10],
        "max_iter": [200, 500, 1000],
        "l1_ratio": [0.1, 0.5, 0.9],
    },
]


param_grid_random_forest = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [
        "auto",
        "sqrt",
        "log2",
    ],
    "bootstrap": [True, False],
}
param_grid_svc = {
    "C": [0.01, 0.1, 1, 10, 100],
    "kernel": ["linear", "poly", "rbf", "sigmoid"],
    "degree": [2, 3, 4],
    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
    "coef0": [
        0.0,
        0.1,
        0.5,
        1.0,
    ],
    "shrinking": [True, False],
    "tol": [1e-4, 1e-3, 1e-2],
    "probability": [True],
    "class_weight": [None, "balanced"],
    "decision_function_shape": ["ovr", "ovo"],
    "max_iter": [-1, 100, 500, 1000],
}

param_grid_decision_tree = {
    "max_depth": [3, 5, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": [
        None,
        "sqrt",
        "log2",
    ],
    "criterion": ["gini", "entropy", "log_loss"],
    "splitter": [
        "best",
        "random",
    ],
}

param_grid_knn = {
    "n_neighbors": [3, 5, 7, 9, 11],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan", "minkowski"],
    "leaf_size": [30, 40, 50],
    "algorithm": [
        "auto",
        "ball_tree",
        "kd_tree",
        "brute",
    ],
}

param_grid_gaussian_nb = {"var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]}

param_grid_gradient_boosting = {
    "loss": ["log_loss", "exponential"],
    "n_estimators": [50, 100, 200],
    "learning_rate": [
        0.01,
        0.05,
        0.1,
        0.2,
    ],
    "criterion": ["friedman_mse", "squared_error"],
    "max_depth": [3, 5, 7, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.8, 0.9, 1.0],
    "max_features": [None, "sqrt", "log2"],
    "n_iter_no_change": [None, 10, 20],
    "max_depth": [3, 5, 7, 10],
}


param_grid_xgboost = {
    "n_estimators": [None, 50, 100, 200],
    "learning_rate": [None, 0.01, 0.1, 0.2],
    "max_depth": [None, 3, 6, 9],
    "max_leaves": [None, 0, 15, 31],
    "min_child_weight": [None, 1, 3, 5],
    "gamma": [None, 0, 1, 5],
    "subsample": [None, 0.8, 1.0],
    "colsample_bytree": [None, 0.5, 0.7, 1.0],
    "reg_alpha": [None, 0, 0.1, 1],
    "reg_lambda": [None, 0, 0.1, 1],
    "scale_pos_weight": [None, 1, 2, 5],
    "grow_policy": [None, "depthwise", "lossguide"],
    "booster": [None, "gbtree", "dart"],
}


hyper_param_grids = {
    "Logistic Regression": param_grid_logistic_regression,
    "Random Forest": param_grid_random_forest,
    "Support Vector Classifier": param_grid_svc,
    "Decision Tree": param_grid_decision_tree,
    "K-Nearest Neighbors": param_grid_knn,
    "Naive Bayes": param_grid_gaussian_nb,
    "Gradient Boosting": param_grid_gradient_boosting,
    "XGBoost": param_grid_xgboost,
}


# Function to find enabled steps in the processing configuration
def find_enabled_steps(processing_config):
    """
    Find enabled steps in the processing configuration.
    """
    enabled_steps = {
        key: {k: v for k, v in value.items() if k != "enabled" and v is not None}
        for key, value in processing_config.items()
        if value.get("enabled", False)
    }

    if enabled_steps:
        formatted_steps = []
        for step, details in enabled_steps.items():
            if details:
                formatted_steps.append(f"{step}: {details}")
            else:
                formatted_steps.append(f"{step}")

        return ", ".join(formatted_steps)
    else:
        return "No processing steps enabled"


# Function to save the entire processing bundle
def save_bundle(bundle, output_dir, dataset_name):
    """
    Save the entire processing bundle to a specified directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    bundles_dir = os.path.join(output_dir, "bundles")
    os.makedirs(bundles_dir, exist_ok=True)

    bundle_hash = hashlib.md5(dill.dumps(bundle)).hexdigest()

    bundle_filename = f"{dataset_name}_{bundle_hash}.joblib"
    bundle_path = os.path.join(bundles_dir, bundle_filename)

    with open(bundle_path, "wb") as f:
        dill.dump({"bundle": bundle}, f)

    logger.info(f"Bundle saved at: {bundle_path}")

    return bundle_path


# Function to update the processing bundle
def update_bundle(
    bundle_path, new_model_key, new_model, dataset_name, label_encoder=None
):
    """
    Update the processing bundle with a new model or label encoder.
    """

    with open(bundle_path, "rb") as f:
        existing_bundle = dill.load(f)["bundle"]

    classifier_name = new_model.__class__.__name__

    if new_model_key == "tuned_model":
        os.remove(bundle_path)
        logger.info(f"Deleted previous bundle: {bundle_path}")

    if label_encoder is not None:
        existing_bundle["label_encoder"] = label_encoder

    existing_bundle[new_model_key] = new_model

    updated_bundle_hash = hashlib.md5(dill.dumps(existing_bundle)).hexdigest()

    bundle_filename = f"{dataset_name}_{classifier_name}_{updated_bundle_hash}.joblib"
    updated_bundle_path = os.path.join(
        os.path.dirname(bundle_path), "models", bundle_filename
    )

    os.makedirs(os.path.dirname(updated_bundle_path), exist_ok=True)

    logger.info(f"Saving updated bundle to: {updated_bundle_path}")

    with open(updated_bundle_path, "wb") as f:
        dill.dump({"bundle": existing_bundle}, f)

    return updated_bundle_path


def ensure_feature_compatibility(X, transformer, name):
    """
    Ensure that the input data X has the same number of features as expected by the transformer.
    If the number of features is different, adjust the input data accordingly.
    """
    expected = getattr(transformer, "n_features_in_", None)
    if expected is None:
        expected = getattr(transformer, "n_input_features_", None)

    if expected is None:
        logger.warning(
            f"[{name}] Transformer does not have n_features_in_ or n_input_features_ attribute. Skipping feature compatibility check."
        )
        return X

    current = X.shape[1]

    if current == expected:
        return X

    if isinstance(X, pd.DataFrame):
        if current < expected:
            for i in range(current, expected):
                X[f"padded_{i}"] = 0
        elif current > expected:
            X = X.iloc[:, :expected]
        logger.warning(
            f"[{name}] Feature count adjusted: {X.shape[1]} (from {current} to {expected})"
        )
        return X

    if current < expected:
        padding = np.zeros((X.shape[0], expected - current))
        X = np.hstack((X, padding))
    elif current > expected:
        X = X[:, :expected]
    logger.warning(
        f"[{name}] Feature count adjusted: {X.shape[1]} (from {current} to {expected})"
    )

    return X
