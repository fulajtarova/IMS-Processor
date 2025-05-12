"""This module splits the dataset into training, validation, and test sets with stratification if possible."""

import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from backend.setup_logger import logger


def split_data(X, y, train_size=0.8, val_size=0.1, test_size=0.1, random_state=3):
    """
    Splits the dataset into training, validation, and test sets with stratification if possible. This function also encodes the labels.
    """
    assert (
        train_size + val_size + test_size == 1
    ), "Train, val, and test sizes must sum to 1."

    # Encode the labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Determine whether stratification is possible
    class_counts = np.bincount(y_encoded)
    min_class_count = class_counts.min()
    total_samples = len(y)

    test_split_size = int(total_samples * test_size)
    val_split_size = int(total_samples * val_size)

    can_stratify = (
        min_class_count >= 2
        and test_split_size >= len(np.unique(y_encoded))
        and val_split_size >= len(np.unique(y_encoded))
    )
    stratify_labels = y_encoded if can_stratify else None

    if not can_stratify:
        logger.warning("Stratified split disabled due to too few samples per class.")

    # First split (train and temp)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y_encoded,
        train_size=train_size,
        stratify=stratify_labels,
        random_state=random_state,
    )

    # Second split (val and test)
    # Try stratifying temp set if still feasible
    if can_stratify and len(np.unique(y_temp)) <= min(np.bincount(y_temp)):
        stratify_labels_temp = y_temp
    else:
        stratify_labels_temp = None

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=test_size / (val_size + test_size),
        stratify=stratify_labels_temp,
        random_state=random_state,
    )

    logger.info(
        f"Train split: {X_train.shape[0]} samples | "
        f"Val split: {X_val.shape[0]} samples | "
        f"Test split: {X_test.shape[0]} samples\n"
        f"Train class count: {dict(zip(*np.unique(y_train, return_counts=True)))} | "
        f"Val class count: {dict(zip(*np.unique(y_val, return_counts=True)))} | "
        f"Test class count: {dict(zip(*np.unique(y_test, return_counts=True)))}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
