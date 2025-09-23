import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils.config import *

def compute_mean_class_4_then_subtract(
    df,
    class_column,
    chip_column,
    columns_to_normalize,
    target_class=4,
    save_stats_json=None
):
    """
    Compute the mean and std of the target class (e.g., class 4) per chip,
    normalize other-class rows using those stats, and save stats to JSON.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        class_column (str): Column name for class labels.
        chip_column (str): Column name for chip IDs.
        columns_to_normalize (list): Names of feature columns to normalize.
        target_class (int): Class to compute normalization stats from.
        save_stats_json (str): Optional JSON path to save normalization statistics.

    Returns:
        tuple: (normalized_df, mean_stats, std_stats)
    """
    df_copy = df.copy()
    means_target_rows = []
    stds_target_rows = []
    stats_per_chip = []

    for chip, chip_group in df_copy.groupby(chip_column):

        target_rows = chip_group[chip_group[class_column] == target_class]
        if target_rows.empty:
            continue  # Skip if no class 4 in this chip

        mean_target = target_rows[columns_to_normalize].mean()
        std_target = target_rows[columns_to_normalize].std().replace(0, 1)

        means_target_rows.append(mean_target.values)
        stds_target_rows.append(std_target.values)

        # Save per-chip stats for JSON
        chip_stats = {
            'chip': int(chip),
            'features': dict(zip(columns_to_normalize, mean_target.values)),
            'std_values': dict(zip(columns_to_normalize, std_target.values))
        }
        stats_per_chip.append(chip_stats)

        # Normalize ALL samples in same chip: (x - mean) / std
        # This includes both target class and other classes
        chip_mask = (df_copy[chip_column] == chip)
        df_copy.loc[chip_mask, columns_to_normalize] = (
            df_copy.loc[chip_mask, columns_to_normalize] - mean_target
        ) / std_target

    # Compute overall statistics
    mean_class_4_overall = np.mean(np.stack(means_target_rows), axis=0)
    std_class_4_overall = np.mean(np.stack(stds_target_rows), axis=0)

    # Save statistics to JSON if path provided
    if save_stats_json:
        stats_dict = {
            "overall_statistics": {
                "mean": mean_class_4_overall.tolist(),
                "std": std_class_4_overall.tolist(),
                "feature_names": columns_to_normalize,
                "target_class": target_class,
            },
            "per_chip_statistics": stats_per_chip,
            "metadata": {
                "creation_date": pd.Timestamp.now().isoformat(),
                "total_chips_processed": len(stats_per_chip),
                "feature_count": len(columns_to_normalize)
            }
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_stats_json), exist_ok=True)
        
        with open(save_stats_json, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"Normalization statistics saved to: {save_stats_json}")

    return df_copy, mean_class_4_overall, std_class_4_overall


def compute_minmax_class_4_then_normalize(
    df,
    class_column,
    chip_column,
    columns_to_normalize,
    target_class=4,
    save_stats_json=None
):
    """
    Compute the min and max of the target class (e.g., class 4) per chip,
    normalize other-class rows using min-max scaling, and save stats to JSON.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        class_column (str): Column name for class labels.
        chip_column (str): Column name for chip IDs.
        columns_to_normalize (list): Names of feature columns to normalize.
        target_class (int): Class to compute normalization stats from.
        save_stats_json (str): Optional JSON path to save normalization statistics.

    Returns:
        tuple: (normalized_df, min_stats, max_stats)
    """
    df_copy = df.copy()
    mins_target_rows = []
    maxs_target_rows = []
    stats_per_chip = []

    for chip, chip_group in df_copy.groupby(chip_column):

        target_rows = chip_group[chip_group[class_column] == target_class]
        if target_rows.empty:
            continue  # Skip if no class 4 in this chip

        min_target = target_rows[columns_to_normalize].min()
        max_target = target_rows[columns_to_normalize].max()
        
        # Handle cases where min == max (avoid division by zero)
        range_target = max_target - min_target
        range_target = range_target.replace(0, 1)

        mins_target_rows.append(min_target.values)
        maxs_target_rows.append(max_target.values)

        # Save per-chip stats for JSON
        chip_stats = {
            'chip': int(chip),
            'min_values': dict(zip(columns_to_normalize, min_target.values)),
            'max_values': dict(zip(columns_to_normalize, max_target.values)),
            'range_values': dict(zip(columns_to_normalize, range_target.values))
        }
        stats_per_chip.append(chip_stats)

        # Min-max normalize ALL samples in same chip: (x - min) / (max - min)
        # This includes both target class and other classes
        chip_mask = (df_copy[chip_column] == chip)
        df_copy.loc[chip_mask, columns_to_normalize] = (
            df_copy.loc[chip_mask, columns_to_normalize] - min_target
        ) / range_target

    # Compute overall statistics
    min_class_4_overall = np.mean(np.stack(mins_target_rows), axis=0)
    max_class_4_overall = np.mean(np.stack(maxs_target_rows), axis=0)

    # Save statistics to JSON if path provided
    if save_stats_json:
        stats_dict = {
            "overall_statistics": {
                "min": min_class_4_overall.tolist(),
                "max": max_class_4_overall.tolist(),
                "feature_names": columns_to_normalize,
                "target_class": target_class,
            },
            "per_chip_statistics": stats_per_chip,
            "metadata": {
                "creation_date": pd.Timestamp.now().isoformat(),
                "total_chips_processed": len(stats_per_chip),
                "feature_count": len(columns_to_normalize),
                "normalization_type": "minmax"
            }
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_stats_json), exist_ok=True)
        
        with open(save_stats_json, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"Min-max normalization statistics saved to: {save_stats_json}")

    return df_copy, min_class_4_overall, max_class_4_overall


def compute_robust_class_4_then_normalize(
    df,
    class_column,
    chip_column,
    columns_to_normalize,
    target_class=4,
    save_stats_json=None
):
    """
    Compute the median and MAD (Median Absolute Deviation) of the target class (e.g., class 4) per chip,
    normalize other-class rows using robust scaling, and save stats to JSON.
    
    Robust scaling uses median and MAD instead of mean and std, making it less sensitive to outliers.
    Formula: (x - median) / MAD, where MAD = median(|x - median|)

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        class_column (str): Column name for class labels.
        chip_column (str): Column name for chip IDs.
        columns_to_normalize (list): Names of feature columns to normalize.
        target_class (int): Class to compute normalization stats from.
        save_stats_json (str): Optional JSON path to save normalization statistics.

    Returns:
        tuple: (normalized_df, median_stats, mad_stats)
    """
    df_copy = df.copy()
    medians_target_rows = []
    mads_target_rows = []
    stats_per_chip = []

    for chip, chip_group in df_copy.groupby(chip_column):

        target_rows = chip_group[chip_group[class_column] == target_class]
        if target_rows.empty:
            continue  # Skip if no class 4 in this chip

        # Compute median and MAD for robust scaling
        median_target = target_rows[columns_to_normalize].median()
        
        # MAD calculation: median(|x - median|)
        deviations = np.abs(target_rows[columns_to_normalize] - median_target)
        mad_target = deviations.median()
        
        # Handle cases where MAD is 0 (avoid division by zero)
        mad_target = mad_target.replace(0, 1)

        medians_target_rows.append(median_target.values)
        mads_target_rows.append(mad_target.values)

        # Save per-chip stats for JSON
        chip_stats = {
            'chip': int(chip),
            'median_values': dict(zip(columns_to_normalize, median_target.values)),
            'mad_values': dict(zip(columns_to_normalize, mad_target.values))
        }
        stats_per_chip.append(chip_stats)

        # Robust normalize ALL samples in same chip: (x - median) / MAD
        # This includes both target class and other classes
        chip_mask = (df_copy[chip_column] == chip)
        df_copy.loc[chip_mask, columns_to_normalize] = (
            df_copy.loc[chip_mask, columns_to_normalize] - median_target
        ) / mad_target

    # Compute overall statistics
    median_class_4_overall = np.mean(np.stack(medians_target_rows), axis=0)
    mad_class_4_overall = np.mean(np.stack(mads_target_rows), axis=0)

    # Save statistics to JSON if path provided
    if save_stats_json:
        stats_dict = {
            "overall_statistics": {
                "median": median_class_4_overall.tolist(),
                "mad": mad_class_4_overall.tolist(),
                "feature_names": columns_to_normalize,
                "target_class": target_class,
            },
            "per_chip_statistics": stats_per_chip,
            "metadata": {
                "creation_date": pd.Timestamp.now().isoformat(),
                "total_chips_processed": len(stats_per_chip),
                "feature_count": len(columns_to_normalize),
                "normalization_type": "robust_scaling",
                "description": "Robust scaling using median and MAD (Median Absolute Deviation)"
            }
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_stats_json), exist_ok=True)
        
        with open(save_stats_json, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"Robust scaling statistics saved to: {save_stats_json}")

    return df_copy, median_class_4_overall, mad_class_4_overall