"""This module serves as test cases for the IMS processor pipeline, generating configurations for individual and combined processing steps."""

import copy
from utils import BASE_CONFIG


def build_config(**kwargs):
    """
    Build a configuration dictionary based on the base configuration and additional parameters.
    """
    cfg = copy.deepcopy(BASE_CONFIG)

    for key, value in kwargs.items():
        if key in cfg:
            cfg[key]["enabled"] = True
            if isinstance(value, dict):
                cfg[key].update(value)

    return cfg


def generate_individual_step_tests():
    """
    Generate individual step tests for the pipeline. Each test is a configuration dictionary that specifies the processing steps to be applied.
    """
    configs = []

    # Default configuration
    configs.append(build_config())

    # Rip-related configurations
    configs.append(build_config(rip_remove={}))
    configs.append(build_config(rip_rel={}))

    # Cutoff configurations
    configs.append(build_config(cutoff={}))

    # Smoothing configurations
    smoothing_methods = ["uniform_filter", "gaussian_blurring", "savitzky_golay"]
    for method in smoothing_methods:
        configs.append(build_config(smoothing={"method": method}))

    # Transformation configurations
    transformation_methods = ["min_max", "max_abs", "zscore", "robust"]
    for method in transformation_methods:
        configs.append(build_config(transformation={"method": method}))

    # Feature selection filter configurations
    fs_filter_methods = ["variance_threshold", "mutual_info"]
    for method in fs_filter_methods:
        configs.append(build_config(feature_selection_filter={"method": method}))

    # Feature selection wrapper configurations
    fs_wrapper_estimators = [
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
        "Support Vector Machine",
    ]
    for estimator in fs_wrapper_estimators:
        configs.append(build_config(feature_selection_wrapper={"estimator": estimator}))

    # Dimensionality reduction configurations
    dr_methods = ["PCA", "LDA", "PCA-LDA"]
    for method in dr_methods:
        configs.append(build_config(dimensionality_reduction={"method": method}))

    return configs


def generate_combined_step_tests():
    """
    Generate combined step tests for the pipeline. Each test is a configuration dictionary that specifies multiple processing steps to be applied.
    """
    return [
        # Baseline (only average)
        build_config(),
        # 2-step tests
        build_config(
            transformation={"method": "zscore"},
            feature_selection_wrapper={"estimator": "Gradient Boosting"},
        ),
        build_config(
            transformation={"method": "zscore"},
            feature_selection_filter={"method": "mutual_info"},
        ),
        build_config(
            transformation={"method": "zscore"},
            dimensionality_reduction={"method": "LDA"},
        ),
        build_config(
            rip_remove={},
            feature_selection_wrapper={"estimator": "Gradient Boosting"},
        ),
        build_config(rip_remove={}, feature_selection_filter={"method": "mutual_info"}),
        build_config(
            smoothing={"method": "savitzky_golay"},
            feature_selection_wrapper={"estimator": "Gradient Boosting"},
        ),
        build_config(
            smoothing={"method": "savitzky_golay"},
            feature_selection_filter={"method": "mutual_info"},
        ),
        build_config(rip_remove={}, transformation={"method": "zscore"}),
        build_config(rip_remove={}, dimensionality_reduction={"method": "LDA"}),
        build_config(
            smoothing={"method": "savitzky_golay"}, transformation={"method": "zscore"}
        ),
        build_config(transformation={"method": "zscore"}, cutoff={}),
        build_config(rip_remove={}, cutoff={}),
        # 3-step tests
        build_config(
            transformation={"method": "zscore"},
            rip_remove={},
            feature_selection_wrapper={"estimator": "Gradient Boosting"},
        ),
        build_config(
            transformation={"method": "zscore"},
            rip_remove={},
            feature_selection_filter={"method": "mutual_info"},
        ),
        build_config(
            transformation={"method": "zscore"},
            smoothing={"method": "savitzky_golay"},
            feature_selection_wrapper={"estimator": "Gradient Boosting"},
        ),
        build_config(
            transformation={"method": "zscore"},
            smoothing={"method": "savitzky_golay"},
            feature_selection_filter={"method": "mutual_info"},
        ),
        build_config(
            transformation={"method": "zscore"},
            rip_remove={},
            dimensionality_reduction={"method": "LDA"},
        ),
        build_config(
            transformation={"method": "zscore"},
            smoothing={"method": "savitzky_golay"},
            dimensionality_reduction={"method": "LDA"},
        ),
    ]
