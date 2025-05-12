"""This module preprocesses raw data for machine learning tasks, including RIP processing, smoothing, cutoff, and spectrum averaging."""

import numpy as np
from scipy.ndimage import uniform_filter, gaussian_filter
from scipy.signal import savgol_filter

from backend.setup_logger import logger


# PREPROCESSING FUNCTIONS
# Function to apply RIP relative or remove
def apply_rip_processing(sample, rip_rel, rip_remove, cut_split):
    """
    Apply RIP relative or remove processing to the sample.
    """
    if rip_remove:
        return sample.riprel().cut_dt(cut_split[0], cut_split[1])
    elif rip_rel:
        return sample.riprel()
    return sample


def set_rip_config(processing_config):
    """
    Set the RIP configuration based on the provided processing configuration.
    """
    rip_rel = processing_config.get("rip_rel", {}).get("enabled", False)
    rip_remove = processing_config.get("rip_remove", {}).get("enabled", False)
    cut_start = processing_config.get("rip_remove", {}).get("cut_start", 1.05)
    cut_end = processing_config.get("rip_remove", {}).get("cut_end", 2.5)

    if rip_rel and rip_remove:
        rip_rel = False
        logger.warning("Both RIP rel and RIP remove enabled — disabling RIP rel.")
        print("Warning: Both RIP rel and remove enabled — disabling RIP rel.")

    cut_split = (
        cut_start if cut_start is not None else 1.05,
        cut_end if cut_end is not None else 2.0,
    )

    if rip_remove:
        logger.info(f"RIP remove enabled with cutting range: {cut_split}")
        print(f"RIP remove enabled with cutting range: {cut_split}")
    elif rip_rel:
        logger.info("RIP relative enabled")
        print("RIP relative enabled")

    return rip_rel, rip_remove, cut_split


# Function to apply smoothing
def apply_smoothing(arr, config, first_print):
    """
    Apply smoothing to the input array using the specified method and size.
    """
    method = config.get("method") or "savitzky_golay"
    size = config.get("size")

    if first_print:
        logger.info(f"Applying smoothing with method: {method}")
        print(f"Applying smoothing with method: {method}")

    if size is None:
        if method == "gaussian_blurring":
            size = 2
        elif method == "savitzky_golay":
            size = 9
        elif method == "uniform_filter":
            size = 5
        else:
            raise ValueError(f"Unknown smoothing method: {method}")

    if method == "gaussian_blurring":
        return gaussian_filter(arr, sigma=size)

    elif method == "savitzky_golay":
        polyorder = min(3, size - 2)

        if size % 2 == 0:
            size += 1
        if size <= polyorder:
            size = polyorder + 2 if (polyorder + 2) % 2 != 0 else polyorder + 3

        smoothed = []
        for row in arr:
            if size >= len(row):
                raise ValueError(
                    f"Savitzky-Golay window_length ({size}) cannot be >= row length ({len(row)})"
                )
            smoothed_row = savgol_filter(row, window_length=size, polyorder=polyorder)
            smoothed.append(smoothed_row)
        return np.array(smoothed)

    elif method == "uniform_filter":
        return np.array([uniform_filter(row, size=size) for row in arr])

    else:
        raise ValueError(f"Unsupported smoothing method: {method}")


# Function to apply cutoff
def apply_cutoff(data, config):
    """
    Apply cutoff to the data based on the provided configuration.
    """
    cut_percentage = config.get("cut_percentage")
    cutoff_value = config.get("cutoff_value")

    if data.ndim != 3:
        raise ValueError("Expected input data to have 3 dimensions")

    if cut_percentage is None and cutoff_value is None:
        cut_percentage = 0.2
        print("No cutoff percentage or value provided, defaulting to 0.2")
        logger.info("No cutoff percentage or value provided, defaulting to 0.2")

    logger.info(f"Applying cutoff")
    print(f"Applying cutoff")

    if cutoff_value is not None:
        keep_columns = np.unique(np.where(data >= cutoff_value)[2])
    else:
        max_value = np.max(data)
        keep_value = max_value * cut_percentage
        keep_columns = np.unique(np.where(data > keep_value)[2])

    if len(keep_columns) == 0:
        raise ValueError("No columns meet the cutoff condition.")

    left_index = keep_columns[0]
    right_index = keep_columns[-1]

    data_new = np.delete(
        data, np.r_[0:left_index, right_index + 1 : data.shape[2]], axis=2
    )

    return data_new


# Function to apply spectrum averaging
def apply_spectrum_averaging(data, axis=1):
    """
    Apply spectrum averaging to the data along the specified axis.
    """
    logger.info("Averaging the spectrum")
    print("Averaging the spectrum")

    averaged_spectrum = np.mean(data, axis=axis)

    return averaged_spectrum


# MAIN FUNCTION TO PREPROCESS RAW DATA
def preprocess_data(data, steps):
    """
    Preprocess the raw data based on the provided steps. This function applies cutoff, smoothing, and spectrum averaging to the data as specified in the steps.
    """
    if steps is None:
        logger.info("No preprocessing steps provided, returning original data")
        print("No preprocessing steps provided, returning original data")
        return data

    logger.info(f"Example of raw data sample: {data[0]}")

    logger.info("Preprocessing data...")
    print("Preprocessing data...")

    # Cutoff
    if steps.get("cutoff", {}).get("enabled", False):
        if steps["rip_rel"]["enabled"] or steps["rip_remove"]["enabled"]:
            logger.warning("Skipping cutoff because RIP relative or remove is enabled")
            print("Skipping cutoff because RIP relative or remove is enabled")
        else:
            data = apply_cutoff(data, steps["cutoff"])

    # Smoothing
    if steps.get("smoothing", {}).get("enabled", False):

        first_print = True
        for i in range(len(data)):
            data[i] = apply_smoothing(data[i], steps["smoothing"], first_print)
            first_print = False

    # Averaging
    if steps.get("spectrum_averaging", {}).get("enabled", True):
        data = apply_spectrum_averaging(data)

    return data
