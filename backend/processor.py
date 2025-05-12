"""This module processes data for machine learning tasks, including scaling, feature selection, and dimensionality reduction."""

import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    MinMaxScaler,
    MaxAbsScaler,
)
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    mutual_info_classif,
    RFE,
)
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


from backend.setup_logger import logger
from utils import save_bundle

# MAPS
classifier_map = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "Gradient Boosting": GradientBoostingClassifier,
    "Support Vector Classifier": SVC,
}

reducer_map = {
    "PCA": PCA,
    "LDA": LDA,
}


# HELPER FUNCTIONS
# Function to fit and transform the data
def fit_transform_on_train(transformer, X_train, X_val, X_test, y=None):
    """
    Fit the transformer on the training data and transform the training, validation, and test data.
    """
    if y is not None:
        transformer.fit(X_train, y)
    else:
        transformer.fit(X_train)

    X_train_transformed = transformer.transform(X_train)
    X_val_transformed = transformer.transform(X_val)
    X_test_transformed = transformer.transform(X_test)
    return X_train_transformed, X_val_transformed, X_test_transformed, transformer


# PROCESSING FUNCTIONS
# Function to apply scaling
def apply_scaling(config, X_train, X_val, X_test):
    """
    Apply scaling to the data based on the specified method.
    """
    method = config.get("method")

    if method is None:
        method = "zscore"

    logger.info(f"Applying scaling: {method}")
    print(f"Applying scaling: {method}")

    if method == "zscore":
        scaler = StandardScaler()
    elif method == "robust":
        scaler = RobustScaler()
    elif method == "min_max":
        scaler = MinMaxScaler()
    elif method == "max_abs":
        scaler = MaxAbsScaler()
    else:
        raise ValueError(f"Unsupported scaling method: {method}")

    X_train, X_val, X_test, scaler = fit_transform_on_train(
        scaler, X_train, X_val, X_test
    )
    return X_train, X_val, X_test, scaler, method


# Function to apply filter feature selection
def apply_filter_feature_selection(config, X_train, X_val, X_test, y_train):
    """
    Apply filter-based feature selection based on the specified method.
    Supports 'variance_threshold' and 'mutual_info'.
    """
    method = config.get("method", "mutual_info")
    threshold = config.get("threshold")

    logger.info(f"Applying filter feature selection: {method}")
    print(f"Applying filter feature selection: {method}")

    if method == "variance_threshold":
        if threshold is None:
            threshold = 1e-5
        selector = VarianceThreshold(threshold=threshold)
        y = None

    elif method == "mutual_info":
        if threshold is None:
            threshold = 0.2

        if isinstance(threshold, float) and 0 < threshold < 1:
            k = max(int(threshold * X_train.shape[1]), 1)
        elif isinstance(threshold, int) and threshold > 0:
            k = threshold
        else:
            raise ValueError(
                "Threshold must be a float between 0 and 1, or a positive integer."
            )

        if k > X_train.shape[1]:
            raise ValueError(
                "Threshold (k) cannot be greater than the number of features."
            )

        selector = SelectKBest(score_func=mutual_info_classif, k=k)
        y = y_train

    else:
        raise ValueError(f"Unsupported feature selection method: '{method}'")

    X_train, X_val, X_test, selector = fit_transform_on_train(
        selector, X_train, X_val, X_test, y=y
    )
    return X_train, X_val, X_test, selector, method


# Function to apply wrapper feature selection
def apply_wrapper_feature_selection(config, X_train, X_val, X_test, y):
    """
    Apply wrapper feature selection based on the specified method.
    """
    estimator = config.get("estimator")
    n_features = config.get("n_features")
    step = config.get("step")

    if estimator is None:
        estimator = "Support Vector Classifier"

    logger.info(f"Applying wrapper feature selection: {estimator}")
    print(f"Applying wrapper feature selection: {estimator}")

    if estimator not in classifier_map:
        estimator = "Support Vector Classifier"

    if n_features is None or n_features <= 0:
        n_features = max(int(0.2 * X_train.shape[1]), 1)

    if n_features > X_train.shape[1]:
        raise ValueError(f"n_features cannot be greater than number of features")

    if step is None:
        step = 0.05
    if step <= 0 or step >= 1:
        raise ValueError(f"Step must be in range (0, 1)")

    estimator_cls = classifier_map[estimator]
    if estimator == "Logistic Regression":
        base_estimator = estimator_cls(
            solver="liblinear", random_state=3, max_iter=1000
        )
    elif estimator == "Support Vector Classifier":
        base_estimator = SVC(kernel="linear", probability=False, random_state=3)

    else:
        base_estimator = estimator_cls(random_state=3)

    wrapper = RFE(estimator=base_estimator, n_features_to_select=n_features, step=step)
    X_train, X_val, X_test, wrapper = fit_transform_on_train(
        wrapper, X_train, X_val, X_test, y
    )
    return X_train, X_val, X_test, wrapper, estimator


# Function to apply dimensionality reduction
def apply_dimensionality_reduction(config, X_train, X_val, X_test, y_train):
    """
    Apply dimensionality reduction based on the specified method.
    """
    method = config.get("method")
    n_components = config.get("n_components")

    if method is None:
        method = "LDA"

    if method == "PCA-LDA":
        # First apply PCA
        pca = PCA(n_components=0.95)
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        X_test = pca.transform(X_test)

        # Then apply LDA
        n_components_lda = min(X_train.shape[1], len(np.unique(y_train)) - 1)
        lda = LDA(n_components=n_components_lda)
        X_train = lda.fit_transform(X_train, y_train)
        X_val = lda.transform(X_val)
        X_test = lda.transform(X_test)

        # Return both reducers if needed later
        return X_train, X_val, X_test, {"PCA": pca, "LDA": lda}, method

    if method not in reducer_map:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")

    if n_components is None:
        if method == "PCA":
            n_components = 0.95
        elif method == "LDA":
            n_components = min(X_train.shape[1], len(np.unique(y_train)) - 1)

    if n_components > X_train.shape[1] or n_components <= 0:
        raise ValueError(f"Invalid n_components")

    print(f"Applying {method} with n_components={n_components}")
    logger.info(f"Applying {method} with n_components={n_components}")

    # Standard single reducer (PCA or LDA)
    reducer = reducer_map[method](n_components=n_components)
    X_train, X_val, X_test, reducer = fit_transform_on_train(
        reducer, X_train, X_val, X_test, y_train
    )
    return X_train, X_val, X_test, reducer, method


# MAIN FUNCTION TO PROCESS SPLITTED DATA
def process_data(steps, X_train, X_val, X_test, y_train, output_dir, dataset_name):
    """
    Process the data by applying scaling, feature selection, and dimensionality reduction.
    """
    bundle = {
        "scaler": None,
        "feature_selector_filter": None,
        "feature_selector_wrapper": None,
        "dimensionality_reducer": None,
        "processing_config": steps,
    }

    methods_used = {}

    print("Processing data...")
    logger.info("Processing data...")

    # Transformation
    if steps["transformation"]["enabled"]:
        X_train, X_val, X_test, scaler, transformation_method = apply_scaling(
            steps["transformation"], X_train, X_val, X_test
        )
        bundle["scaler"] = scaler
        methods_used["transformation"] = transformation_method

    # Feature Selection Filter
    if steps.get("feature_selection_filter", {}).get("enabled", False):
        X_train, X_val, X_test, selector, filter_method = (
            apply_filter_feature_selection(
                steps["feature_selection_filter"],
                X_train,
                X_val,
                X_test,
                y_train,
            )
        )
        bundle["feature_selector_filter"] = selector
        methods_used["feature_selection_filter"] = filter_method

    # Feature Selection Wrapper
    if steps.get("feature_selection_wrapper", {}).get("enabled", False):
        X_train, X_val, X_test, wrapper, wrapper_method = (
            apply_wrapper_feature_selection(
                steps["feature_selection_wrapper"], X_train, X_val, X_test, y_train
            )
        )
        bundle["feature_selector_wrapper"] = wrapper
        methods_used["feature_selection_wrapper"] = wrapper_method

    # Dimensionality Reduction
    if steps.get("dimensionality_reduction", {}).get("enabled", False):
        X_train, X_val, X_test, reducer, reduction_method = (
            apply_dimensionality_reduction(
                steps["dimensionality_reduction"],
                X_train,
                X_val,
                X_test,
                y_train,
            )
        )
        bundle["dimensionality_reducer"] = reducer
        methods_used["dimensionality_reduction"] = reduction_method

    # Print Final Shapes
    logger.info(f"Final shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")
    print(f"Final shapes: {X_train.shape}, {X_val.shape}, {X_test.shape}")

    # Save the processing bundle with methods used

    bundle_path = save_bundle(bundle, output_dir, dataset_name)
    logger.info(f"Processing bundle saved at: {bundle_path}")
    print(f"Processing bundle saved at: {bundle_path}")

    return X_train, X_val, X_test, bundle_path
