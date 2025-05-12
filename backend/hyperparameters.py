"""This module manages hyperparameter tuning for machine learning models."""

import os
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold,
    GridSearchCV,
    RandomizedSearchCV,
    KFold,
)
from backend.setup_logger import logger

import warnings
from sklearn.exceptions import ConvergenceWarning

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def hyperparameter_tuning(
    classifier: object,
    param_grid: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    dataset_name: str,
    results_file: str,
    settings: dict,
):
    """
    Perform hyperparameter tuning using GridSearchCV or RandomizedSearchCV.
    """
    method = settings.get("method", "grid_search")
    cv_folds = settings.get("cv_folds", 5)
    n_iter = settings.get("n_iter", 10)
    scoring = settings.get("scoring", "accuracy")

    if settings.get("cv_strategy", "stratified") == "kfold":
        cv_strategy = KFold(n_splits=cv_folds, shuffle=True)
        logger.info("Using KFold cross-validation strategy.")
    else:
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True)
        logger.info("Using StratifiedKFold cross-validation strategy.")

    if method == "random_search":
        search = RandomizedSearchCV(
            estimator=classifier,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=-1,
            refit=scoring,
        )
        logger.info(f"Using RandomizedSearchCV with {n_iter} iterations.")
    else:
        search = GridSearchCV(
            estimator=classifier,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_strategy,
            n_jobs=-1,
            refit=scoring,
        )
        logger.info("Using GridSearchCV.")

    try:
        search.fit(X_train, y_train)
    except Exception as e:
        print(f"Search failed: {e}")
        raise

    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score_cv = search.best_score_
    val_score = best_model.score(X_val, y_val)

    logger.info(
        f"Best parameters for {classifier.__class__.__name__} on {dataset_name}: {best_params}"
    )
    logger.info(f"Best CV score ({scoring}): {best_score_cv}")
    logger.info(f"Validation score: {val_score}")

    result_data = {
        "Dataset": dataset_name,
        "Model": classifier.__class__.__name__,
        "Best Parameters": str(best_params),
        "Best Score (CV)": best_score_cv,
        "Validation Score": val_score,
    }

    results_df = pd.DataFrame([result_data])
    results_df.to_csv(
        results_file, mode="a", header=not os.path.exists(results_file), index=False
    )

    print(
        f"Results saved to {results_file}. Best parameters: {best_params}, Best CV score: {best_score_cv}, Validation score: {val_score}"
    )
    return best_model
