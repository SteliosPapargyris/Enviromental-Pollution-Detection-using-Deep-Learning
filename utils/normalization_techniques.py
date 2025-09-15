import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from utils.config import *

def save_statistics_json(mean_data, std_data, columns, base_path):
    """Save statistics in JSON format only"""
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats_dict = {
        "mean": mean_data.tolist(),
        "std": std_data.tolist(), 
        "feature_names": columns,
        "creation_date": pd.Timestamp.now().isoformat(),
        "shape": mean_data.shape
    }
    
    json_path = f"{base_path}_stats.json"
    with open(json_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    
    print(f"Statistics saved to: {json_path}")

def compute_mean_class_4_then_subtract(
    df,
    chip_exclude,
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
        if chip == chip_exclude:
            continue

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
                "excluded_chip": chip_exclude
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
    chip_exclude,
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
        chip_exclude: Chip to exclude from processing.
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
        if chip == chip_exclude:
            continue

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
                "excluded_chip": chip_exclude
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
    chip_exclude,
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
        chip_exclude: Chip to exclude from processing.
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
        if chip == chip_exclude:
            continue

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
                "excluded_chip": chip_exclude
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


def _apply_class_based_robust_normalization(X, y, df, target_class_encoded, normalization_columns, 
                                           stats_source, stats_path, normalize_target_class):
    """Apply class-based robust scaling normalization using target class statistics."""
    
    # Get normalization statistics based on chosen method
    if stats_source == 'json':
        if stats_path is None:
            raise ValueError("stats_path must be provided when stats_source='json'")
        
        # Load statistics from JSON
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Check if this is robust stats
        if 'median' in stats.get('overall_statistics', {}):
            # Robust scaling statistics
            peak_median = np.array(stats['overall_statistics']['median'])
            peak_mad = np.array(stats['overall_statistics']['mad'])
        else:
            raise ValueError("JSON stats file does not contain robust scaling statistics. Please use robust normalization stats.")
        
        # Handle temperature stats
        if 'Temperature' in X.columns:
            normalization_mask = (df['Chip'] == chip_exclude) & (df['Class'] == target_class_encoded)
            normalization_data = df[normalization_mask]
            
            if normalization_data.empty:
                raise ValueError(f"No data found for target class {target_class} in chip {chip_exclude}")
            
            temp_median = normalization_data['Temperature'].median()
            temp_mad = np.median(np.abs(normalization_data['Temperature'] - temp_median))
            temp_mad = temp_mad if temp_mad != 0 else 1
        
    elif stats_source == 'compute':
        # Calculate normalization statistics from data
        normalization_mask = (df['Chip'] == chip_exclude) & (df['Class'] == target_class_encoded)
        normalization_data = df[normalization_mask]
        
        if normalization_data.empty:
            raise ValueError(f"No data found for target class {target_class} in chip {chip_exclude}")
        
        # Separate peak columns and temperature columns for consistent calculation
        peak_columns = [col for col in normalization_columns if col.startswith('Peak')]
        peak_median = normalization_data[peak_columns].median().values
        
        # Calculate MAD for peak columns
        deviations = np.abs(normalization_data[peak_columns] - peak_median)
        peak_mad = deviations.median().values
        peak_mad = np.where(peak_mad == 0, 1, peak_mad)  # Handle zero MAD
        
        # Calculate temperature stats separately
        temp_median = normalization_data['Temperature'].median()
        temp_mad = np.median(np.abs(normalization_data['Temperature'] - temp_median))
        temp_mad = temp_mad if temp_mad != 0 else 1
        
    else:
        raise ValueError("stats_source must be either 'compute' or 'json'")
    
    # Apply normalization
    peak_columns = [col for col in normalization_columns if col.startswith('Peak')]
    _normalize_features_robust(X, y, target_class_encoded, peak_columns, 
                              peak_median, peak_mad, temp_median, temp_mad, normalize_target_class)
    

def _apply_class_based_normalization(X, y, df, target_class_encoded, normalization_columns, 
                                   stats_source, stats_path, normalize_target_class):
    """Apply class-based normalization using target class statistics."""
    
    # Get normalization statistics based on chosen method
    if stats_source == 'json':
        if stats_path is None:
            raise ValueError("stats_path must be provided when stats_source='json'")
        
        # Load statistics from JSON
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        peak_mean = np.array(stats['mean'])
        peak_std = np.array(stats['std'])
        
        # Handle temperature stats
        if 'Temperature' in X.columns:
            normalization_mask = (df['Chip'] == chip_exclude) & (df['Class'] == target_class_encoded)
            normalization_data = df[normalization_mask]
            
            if normalization_data.empty:
                raise ValueError(f"No data found for target class {target_class} in chip {chip_exclude}")
            
            temp_mean = normalization_data['Temperature'].mean()
            temp_std = normalization_data['Temperature'].std()
        
    elif stats_source == 'compute':
        # Calculate normalization statistics from data
        normalization_mask = (df['Chip'] == chip_exclude) & (df['Class'] == target_class_encoded)
        normalization_data = df[normalization_mask]
        
        if normalization_data.empty:
            raise ValueError(f"No data found for target class {target_class} in chip {chip_exclude}")
        
        # Separate peak columns and temperature columns for consistent calculation
        peak_columns = [col for col in normalization_columns if col.startswith('Peak')]
        peak_mean = normalization_data[peak_columns].mean().values
        peak_std = normalization_data[peak_columns].std().values
        
        # Calculate temperature stats separately
        temp_mean = normalization_data['Temperature'].mean()
        temp_std = normalization_data['Temperature'].std()
        
    else:
        raise ValueError("stats_source must be either 'compute' or 'json'")
    
    # Apply normalization
    peak_columns = [col for col in normalization_columns if col.startswith('Peak')]
    _normalize_features(X, y, target_class_encoded, peak_columns, 
                       peak_mean, peak_std, temp_mean, temp_std, normalize_target_class)


def _apply_standard_normalization(X, normalization_columns):
    """Apply standard z-score normalization to all features."""
    from sklearn.preprocessing import StandardScaler
    
    # Normalize peak features
    if normalization_columns:
        scaler_peaks = StandardScaler()
        X[normalization_columns] = scaler_peaks.fit_transform(X[normalization_columns])
    
    # Normalize temperature
    if 'Temperature' in X.columns:
        scaler_temp = StandardScaler()
        X[['Temperature']] = scaler_temp.fit_transform(X[['Temperature']])


def _apply_minmax_normalization(X, normalization_columns):
    """Apply min-max normalization to all features."""
    from sklearn.preprocessing import MinMaxScaler
    
    # Normalize peak features
    if normalization_columns:
        scaler_peaks = MinMaxScaler()
        X[normalization_columns] = scaler_peaks.fit_transform(X[normalization_columns])
    
    # Normalize temperature
    if 'Temperature' in X.columns:
        scaler_temp = MinMaxScaler()
        X[['Temperature']] = scaler_temp.fit_transform(X[['Temperature']])


def _apply_class_based_minmax_normalization(X, y, df, target_class_encoded, normalization_columns, 
                                          stats_source, stats_path, normalize_target_class):
    """Apply class-based min-max normalization using target class statistics."""
    
    # Get normalization statistics based on chosen method
    if stats_source == 'json':
        if stats_path is None:
            raise ValueError("stats_path must be provided when stats_source='json'")
        
        # Load statistics from JSON
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Check if this is min-max stats or mean/std stats
        if 'min' in stats.get('overall_statistics', {}):
            # Min-max statistics
            peak_min = np.array(stats['overall_statistics']['min'])
            peak_max = np.array(stats['overall_statistics']['max'])
        else:
            raise ValueError("JSON stats file does not contain min-max statistics. Please use minmax normalization stats.")
        
        # Handle temperature stats
        if 'Temperature' in X.columns:
            normalization_mask = (df['Chip'] == chip_exclude) & (df['Class'] == target_class_encoded)
            normalization_data = df[normalization_mask]
            
            if normalization_data.empty:
                raise ValueError(f"No data found for target class {target_class} in chip {chip_exclude}")
            
            temp_min = normalization_data['Temperature'].min()
            temp_max = normalization_data['Temperature'].max()
        
    elif stats_source == 'compute':
        # Calculate normalization statistics from data
        normalization_mask = (df['Chip'] == chip_exclude) & (df['Class'] == target_class_encoded)
        normalization_data = df[normalization_mask]
        
        if normalization_data.empty:
            raise ValueError(f"No data found for target class {target_class} in chip {chip_exclude}")
        
        # Separate peak columns and temperature columns for consistent calculation
        peak_columns = [col for col in normalization_columns if col.startswith('Peak')]
        peak_min = normalization_data[peak_columns].min().values
        peak_max = normalization_data[peak_columns].max().values
        
        # Calculate temperature stats separately
        temp_min = normalization_data['Temperature'].min()
        temp_max = normalization_data['Temperature'].max()
        
    else:
        raise ValueError("stats_source must be either 'compute' or 'json'")
    
    # Apply normalization
    peak_columns = [col for col in normalization_columns if col.startswith('Peak')]
    _normalize_features_minmax(X, y, target_class_encoded, peak_columns, 
                              peak_min, peak_max, temp_min, temp_max, normalize_target_class)


def _normalize_features_minmax(X, y, target_class_encoded, peak_columns, 
                              peak_min, peak_max, temp_min, temp_max, normalize_target_class=True):
    """
    Helper function to apply class-based min-max normalization to features.
    Min-max scaling normalizes ALL classes using Class 4 statistics.
    
    Args:
        X (pd.DataFrame): Feature matrix to normalize
        y (pd.Series): Labels (not used in min-max scaling as we normalize all)
        target_class_encoded (int): Target class value (not used as we normalize all)
        peak_columns (list): List of peak column names
        peak_min, peak_max (np.array): Min and max for peak normalization
        temp_min, temp_max (float): Min and max for temperature normalization
        normalize_target_class (bool): Always True for min-max scaling
    """
    # For min-max scaling, normalize ALL samples (using Class 4 statistics)
    # This ensures all samples including Class 4 are normalized
    
    if peak_columns:
        # Avoid division by zero
        peak_range = peak_max - peak_min
        peak_range_safe = np.where(peak_range == 0, 1, peak_range)
        X[peak_columns] = (X[peak_columns] - peak_min) / peak_range_safe
    
    # Normalize temperature for all samples
    if 'Temperature' in X.columns:
        temp_range = temp_max - temp_min
        temp_range_safe = temp_range if temp_range != 0 else 1
        X['Temperature'] = (X['Temperature'] - temp_min) / temp_range_safe


def _normalize_features(X, y, target_class_encoded, peak_columns, 
                       peak_mean, peak_std, temp_mean, temp_std, normalize_target_class=True):
    """
    Helper function to apply class-based normalization to features.
    Mean/std scaling normalizes ALL classes using Class 4 statistics.
    
    Args:
        X (pd.DataFrame): Feature matrix to normalize
        y (pd.Series): Labels (not used in mean/std scaling as we normalize all)
        target_class_encoded (int): Target class value (not used as we normalize all)
        peak_columns (list): List of peak column names
        peak_mean, peak_std (np.array): Mean and std for peak normalization
        temp_mean, temp_std (float): Mean and std for temperature normalization
        normalize_target_class (bool): Always True for mean/std scaling
    """
    # For mean/std scaling, normalize ALL samples (using Class 4 statistics)
    # This ensures all samples including Class 4 are normalized
    
    if peak_columns:
        # Avoid division by zero
        peak_std_safe = np.where(peak_std == 0, 1, peak_std)
        X[peak_columns] = (X[peak_columns] - peak_mean) / peak_std_safe
    
    # Normalize temperature for all samples
    if 'Temperature' in X.columns:
        temp_std_safe = temp_std if temp_std != 0 else 1
        X['Temperature'] = (X['Temperature'] - temp_mean) / temp_std_safe


def _normalize_features_robust(X, y, target_class_encoded, peak_columns, 
                              peak_median, peak_mad, temp_median, temp_mad, normalize_target_class=True):
    """
    Helper function to apply class-based robust scaling normalization to features.
    Robust scaling normalizes ALL classes using Class 4 statistics.
    
    Args:
        X (pd.DataFrame): Feature matrix to normalize
        y (pd.Series): Labels (not used in robust scaling as we normalize all)
        target_class_encoded (int): Target class value (not used as we normalize all)
        peak_columns (list): List of peak column names
        peak_median, peak_mad (np.array): Median and MAD for peak normalization
        temp_median, temp_mad (float): Median and MAD for temperature normalization
        normalize_target_class (bool): Always True for robust scaling
    """
    # For robust scaling, normalize ALL samples (using Class 4 statistics)
    # This ensures all samples including Class 4 are normalized
    
    if peak_columns:
        # Apply robust normalization: (x - median) / MAD
        X[peak_columns] = (X[peak_columns] - peak_median) / peak_mad
    
    # Normalize temperature for all samples
    if 'Temperature' in X.columns:
        X['Temperature'] = (X['Temperature'] - temp_median) / temp_mad


def _normalize_features_peak_to_peak(X, y, target_class_encoded, peak_columns, 
                                    peak_range, temp_range, normalize_target_class=False):
    """
    Helper function to apply class-based peak-to-peak normalization to features.
    
    Args:
        X (pd.DataFrame): Feature matrix to normalize
        y (pd.Series): Labels
        target_class_encoded (int): Encoded target class value
        peak_columns (list): List of peak column names
        peak_range (np.array): Range (max - min) for peak normalization
        temp_range (float): Range (max - min) for temperature normalization
        normalize_target_class (bool): Whether to normalize target class features
    """
    # Determine which classes to normalize
    if normalize_target_class:
        # Normalize all classes
        normalize_mask = pd.Series([True] * len(y), index=y.index)
    else:
        # Normalize only non-target classes (original behavior)
        normalize_mask = (y != target_class_encoded)
    
    if peak_columns and normalize_mask.sum() > 0:
        # Peak-to-peak normalization: x / range
        X.loc[normalize_mask, peak_columns] = (
            X.loc[normalize_mask, peak_columns] / peak_range
        )
    
    # Normalize temperature based on normalize_target_class setting
    if 'Temperature' in X.columns:
        if normalize_target_class:
            # Normalize temperature for all classes
            X['Temperature'] = X['Temperature'] / temp_range
        else:
            # Normalize temperature only for non-target classes
            X.loc[normalize_mask, 'Temperature'] = X.loc[normalize_mask, 'Temperature'] / temp_range