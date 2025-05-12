"""This module loads and processes datasets for machine learning tasks."""

import os
import json
import datetime
import numpy as np
import pandas as pd
import ims
import random

from backend.preprocessor import (
    preprocess_data,
    apply_rip_processing,
    set_rip_config,
)
from utils import find_enabled_steps
from backend.setup_logger import logger


# Function to open a text file and read its contents
def open_txt_file(file_path):
    """
    Open a text file and read its contents. The first line contains the annotation table information, and the second line contains the sample folder path.
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None, None, None

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        annotation_table_info = lines[0].strip().split("|")

        annotation_table = annotation_table_info[1].lower()
        csv_table_path = (
            annotation_table_info[2] if len(annotation_table_info) > 2 else None
        )
        sample_folder_path = lines[1].strip()

        return annotation_table, csv_table_path, sample_folder_path

    except Exception as e:
        print(f"Error reading file: {e}")
        return None, None, None


# Function to create a dataset from the sample folder
def create_dataset(sample_folder_path, processing_config, predict, csv_table_path=None):
    """
    Create a dataset from the sample folder. The dataset is created by reading .mea and .json files from the folder. The function also applies RIP processing and returns the dataset as numpy arrays.
    """
    X, y = [], []
    sample_names = []

    rip_rel, rip_remove, cut_split = set_rip_config(processing_config)

    df = None
    if csv_table_path:
        try:
            df = pd.read_csv(csv_table_path, encoding="utf-8")
            df.columns = df.columns.str.strip().str.lower()
            if "sampleid" not in df.columns:
                logger.error(
                    f"'SampleID' column not found in the CSV file: {csv_table_path}"
                )
                print(
                    f"Error: 'SampleID' column not found in the CSV file: {csv_table_path}"
                )
                return None, None
            df["sampleid"] = df["sampleid"].astype(str)
        except Exception as e:
            logger.error(f"Error reading CSV file: {e}")
            print(f"Error reading CSV file: {e}")
            return None, None

    for root, _, files in os.walk(sample_folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            if not os.path.isfile(file_path):
                continue

            ext = file.lower().split(".")[-1]
            if ext not in {"mea", "json"}:
                continue

            rel_path = os.path.relpath(root, sample_folder_path)
            class_label = rel_path.split(os.sep)[0] if rel_path != "." else "unknown"

            if ext == "mea":
                process_mea_file(
                    file_path, df, class_label, rip_rel, rip_remove, cut_split, X, y
                )
                # Append only the file name without the extension
                sample_names.append(os.path.basename(file_path))
            elif ext == "json":
                process_json_file(
                    file_path, df, class_label, rip_rel, rip_remove, cut_split, X, y
                )
                # Append only the file name without the extension
                sample_names.append(os.path.basename(file_path))

    if rip_remove and X:
        min_rows = min(val.shape[0] for val in X)
        min_cols = min(val.shape[1] for val in X)
        X = [val[:min_rows, :min_cols] for val in X]

    return np.array(X, dtype=np.float32), np.array(y), sample_names


# Function to process .mea files
def process_mea_file(file_path, df, class_folder, rip_rel, rip_remove, cut_split, X, y):
    """
    Process a .mea file and extract the spectrum data.
    """
    try:
        file_sample_id = os.path.basename(file_path).replace(".mea", "")

        class_label = get_class_label(file_sample_id, df, class_folder)

        sample = ims.Spectrum.read_mea(file_path)
        sample = apply_rip_processing(sample, rip_rel, rip_remove, cut_split)

        if sample.values is not None:
            X.append(sample.values)
            y.append(class_label)
    except Exception as e:
        logger.error(f"Error reading .mea file {file_path}: {e}")
        print(f"Error reading .mea file {file_path}: {e}")


# Function to process .json files
def process_json_file(
    file_path, df, class_folder, rip_rel, rip_remove, cut_split, X, y
):
    """
    Process a .json file and extract the spectrum data.
    """
    try:
        with open(file_path, "r") as json_file:
            entry = json.load(json_file)

        spectrum_obj = ims.Spectrum(
            name=entry["name"],
            values=np.array(entry["values"], dtype=np.float32),
            ret_time=np.array(entry["ret_time"], dtype=np.float32),
            drift_time=np.array(entry["drift_time"], dtype=np.float32)[::-1],
            time=datetime.datetime.now(),
        )
        spectrum_obj = apply_rip_processing(
            spectrum_obj, rip_rel, rip_remove, cut_split
        )

        file_sample_id = entry["name"]
        class_label = get_class_label(file_sample_id, df, class_folder)

        if spectrum_obj.values is not None:
            X.append(spectrum_obj.values)
            y.append(class_label)
    except Exception as e:
        logger.error(f"Error reading .json file {file_path}: {e}")
        print(f"Error reading .json file {file_path}: {e}")


# Function to get the class label from the DataFrame
def get_class_label(file_sample_id, df, default_class_label):
    """
    Get the class label for a given sample ID.
    """
    if df is not None and file_sample_id in df["sampleid"].values:
        return df.loc[df["sampleid"] == file_sample_id, "class"].values[0]
    return default_class_label


# Function to get dataset information
def get_dataset_info(X: np.ndarray, y: np.ndarray, num_preview: int = 5):
    """
    Get dataset information including sample count, shape, class distribution, and a preview of samples.
    """
    summary = {
        "samples": X.shape[0],
        "sample_shape": X.shape[1] if len(X.shape) > 1 else 1,
    }

    try:
        y_series = pd.Series(y)
        summary["class_distribution"] = y_series.value_counts().to_dict()
    except Exception:
        summary["class_distribution"] = {}

    try:
        indices = random.sample(range(X.shape[0]), min(num_preview, X.shape[0]))
        summary["sample_preview"] = [
            {
                "label": y[i],
                "stats": {
                    "min": round(float(np.min(X[i])), 2),
                    "max": round(float(np.max(X[i])), 2),
                    "mean": round(float(np.mean(X[i])), 2),
                    "median": round(float(np.median(X[i])), 2),
                    "std": round(float(np.std(X[i])), 2),
                },
            }
            for i in indices
        ]
    except Exception:
        summary["sample_preview"] = []

    return summary


# Function to read datasets from file paths
def read_dataset(file_paths: list, processing_config: dict, predict: bool = False):
    """
    Read datasets from the provided file paths. The datasets are created by reading .mea and .json files from the folders. The function also applies RIP processing and returns the datasets as numpy arrays.
    """
    datasets = {}

    logger.info("Starting dataset loading process...")
    print("Starting dataset loading process...")

    enabled_steps = find_enabled_steps(processing_config)
    logger.info(f"Processing configuration: {enabled_steps}")
    print(f"Processing configuration: {enabled_steps}")

    for file_path in file_paths:
        print(f"\nReading dataset from: {file_path}")
        logger.info(f"Reading dataset from: {file_path}")

        annotation_table, csv_table_path, sample_folder_path = open_txt_file(file_path)

        if annotation_table and csv_table_path:
            logger.info(
                f"Annotations found. Creating dataset with annotations from: {sample_folder_path}"
            )
            X, y, sample_names = create_dataset(
                sample_folder_path, processing_config, predict, csv_table_path
            )
        else:
            logger.info(
                f"No annotations found. Creating dataset without annotations from: {sample_folder_path}"
            )
            X, y, sample_names = create_dataset(
                sample_folder_path, processing_config, predict
            )

        if X is not None and y is not None:
            print(f"Dataset loaded successfully. Shape: {X.shape}")
            logger.info(
                f"Dataset loaded successfully for {file_path}. Shape: {X.shape}"
            )

            X = preprocess_data(X, processing_config)

            dataset_name = os.path.basename(file_path).split(".")[0]
            datasets[dataset_name] = (X, y)
            print(f"Dataset preprocessing completed. Shape: {X.shape}")
            logger.info(
                f"Dataset preprocessing completed for {file_path}. Shape: {X.shape}"
            )
        else:
            print(f"Failed to load dataset from {file_path}")
            logger.error(f"Failed to load dataset from {file_path}. Check input files.")

    logger.info("All datasets have been successfully loaded and processed.")
    return datasets, sample_names
