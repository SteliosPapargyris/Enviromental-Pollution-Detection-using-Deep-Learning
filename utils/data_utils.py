import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.plot_utils import plot_raw_test_mean_feature_per_class
import torch
import os
from typing import List
from utils.config import *
import json
import numpy as np
from pathlib import Path

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


def compute_peak_to_peak_class_4_then_normalize(
    df,
    chip_exclude,
    class_column,
    chip_column,
    columns_to_normalize,
    target_class=4,
    save_stats_json=None
):
    """
    Compute the range (max - min) of the target class (e.g., class 4) per chip,
    normalize other-class rows by dividing by the range, and save stats to JSON.
    
    Peak-to-peak normalization preserves spectral shape while normalizing amplitude.
    Formula: x_normalized = x / (max - min)

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        chip_exclude: Chip to exclude from processing.
        class_column (str): Column name for class labels.
        chip_column (str): Column name for chip IDs.
        columns_to_normalize (list): Names of feature columns to normalize.
        target_class (int): Class to compute normalization stats from.
        save_stats_json (str): Optional JSON path to save normalization statistics.

    Returns:
        tuple: (normalized_df, range_stats)
    """
    df_copy = df.copy()
    ranges_target_rows = []
    stats_per_chip = []

    for chip, chip_group in df_copy.groupby(chip_column):
        if chip == chip_exclude:
            continue

        target_rows = chip_group[chip_group[class_column] == target_class]
        if target_rows.empty:
            continue  # Skip if no class 4 in this chip

        # Compute range (max - min) for peak-to-peak normalization
        min_target = target_rows[columns_to_normalize].min()
        max_target = target_rows[columns_to_normalize].max()
        range_target = max_target - min_target
        
        # Handle cases where range is 0 (avoid division by zero)
        range_target = range_target.replace(0, 1)

        ranges_target_rows.append(range_target.values)

        # Save per-chip stats for JSON
        chip_stats = {
            'chip': int(chip),
            'min_values': dict(zip(columns_to_normalize, min_target.values)),
            'max_values': dict(zip(columns_to_normalize, max_target.values)),
            'range_values': dict(zip(columns_to_normalize, range_target.values))
        }
        stats_per_chip.append(chip_stats)

        # Peak-to-peak normalize other-class rows in same chip: x / range
        other_mask = (df_copy[chip_column] == chip) & (df_copy[class_column] != target_class)
        df_copy.loc[other_mask, columns_to_normalize] = (
            df_copy.loc[other_mask, columns_to_normalize].astype(float) / range_target
        )

    # Compute overall statistics
    range_class_4_overall = np.mean(np.stack(ranges_target_rows), axis=0)

    # Save statistics to JSON if path provided
    if save_stats_json:
        stats_dict = {
            "overall_statistics": {
                "range": range_class_4_overall.tolist(),
                "feature_names": columns_to_normalize,
                "target_class": target_class,
                "excluded_chip": chip_exclude
            },
            "per_chip_statistics": stats_per_chip,
            "metadata": {
                "creation_date": pd.Timestamp.now().isoformat(),
                "total_chips_processed": len(stats_per_chip),
                "feature_count": len(columns_to_normalize),
                "normalization_type": "peak_to_peak",
                "description": "Peak-to-peak normalization using range (max - min) scaling"
            }
        }
        
        # Create directory if needed
        os.makedirs(os.path.dirname(save_stats_json), exist_ok=True)
        
        with open(save_stats_json, 'w') as f:
            json.dump(stats_dict, f, indent=2)
        
        print(f"Peak-to-peak normalization statistics saved to: {save_stats_json}")

    return df_copy, range_class_4_overall


def _apply_class_based_peak_to_peak_normalization(X, y, df, target_class_encoded, normalization_columns, 
                                                 stats_source, stats_path, normalize_target_class):
    """Apply class-based peak-to-peak normalization using target class statistics."""
    
    # Get normalization statistics based on chosen method
    if stats_source == 'json':
        if stats_path is None:
            raise ValueError("stats_path must be provided when stats_source='json'")
        
        # Load statistics from JSON
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        # Check if this is peak-to-peak stats
        if 'range' in stats.get('overall_statistics', {}):
            # Peak-to-peak statistics
            peak_range = np.array(stats['overall_statistics']['range'])
        else:
            raise ValueError("JSON stats file does not contain peak-to-peak statistics. Please use peak-to-peak normalization stats.")
        
        # Handle temperature stats
        if 'Temperature' in X.columns:
            normalization_mask = (df['Chip'] == chip_exclude) & (df['Class'] == target_class_encoded)
            normalization_data = df[normalization_mask]
            
            if normalization_data.empty:
                raise ValueError(f"No data found for target class {target_class} in chip {chip_exclude}")
            
            temp_min = normalization_data['Temperature'].min()
            temp_max = normalization_data['Temperature'].max()
            temp_range = temp_max - temp_min
            temp_range = temp_range if temp_range != 0 else 1
        
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
        peak_range = peak_max - peak_min
        peak_range = np.where(peak_range == 0, 1, peak_range)  # Handle zero range
        
        # Calculate temperature stats separately
        temp_min = normalization_data['Temperature'].min()
        temp_max = normalization_data['Temperature'].max()
        temp_range = temp_max - temp_min
        temp_range = temp_range if temp_range != 0 else 1
        
    else:
        raise ValueError("stats_source must be either 'compute' or 'json'")
    
    # Apply normalization
    peak_columns = [col for col in normalization_columns if col.startswith('Peak')]
    _normalize_features_peak_to_peak(X, y, target_class_encoded, peak_columns, 
                                    peak_range, temp_range, normalize_target_class)


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


def dataset_creation(csv_indices: List[int], baseline_chip: int) -> pd.DataFrame:
    merged_csv_path = os.path.join(base_path, "shuffled_dataset", "merged.csv")

    # Merge specified CSVs
    dfs = [pd.read_csv(os.path.join(base_path, f"{i}.csv")) for i in csv_indices]
    merged_train_df = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)  # Shuffle

    # Load baseline chip
    df_baseline = pd.read_csv(os.path.join(base_path, f"{baseline_chip}.csv"))
    merged_rows = []

    for _, train_row in merged_train_df.iterrows():
        matching_rows = df_baseline[
            (df_baseline['Temperature'] == train_row['Temperature']) &
            (df_baseline['Class'] == train_row['Class'])
        ]
        for _, match_row in matching_rows.iterrows():
            train_series = train_row.add_prefix("train_")
            match_series = match_row.add_prefix("match_")
            merged_row = pd.concat([train_series, match_series])
            merged_rows.append(merged_row)

    merged_df = pd.DataFrame(merged_rows)

    # Remove duplicated columns
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Save to file
    os.makedirs(os.path.dirname(merged_csv_path), exist_ok=True)
    merged_df.to_csv(merged_csv_path, index=False)
    return merged_df

def load_and_preprocess_data_autoencoder(file_path, random_state=42):
    # Load data
    df = pd.read_csv(file_path)

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['train_Class'] = label_encoder.fit_transform(df['train_Class'])
    df['match_Class'] = label_encoder.fit_transform(df['match_Class'])
    # NOTE: Temperature normalization removed - should be handled by apply_normalization.py scripts
    X = df.drop(columns=["train_Chip"])
    X = X.iloc[:, :33]
    y = df.drop(columns=["match_Chip"])
    y = y.iloc[:, 35:-1]

    # 70-20-10 split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.3,  # 30% for temp
        random_state=random_state
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.333,  # 10% of total
        random_state=random_state
    )

    print(f"\n=== Autoencoder Data Split ===")
    total_samples = len(X)
    print(f"Training: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/total_samples:.1%})")

    return (X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)

def load_and_preprocess_data_classifier(file_path, random_state=42):
    # Load data
    df = pd.read_csv(file_path)

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['train_Class'] = label_encoder.fit_transform(df['train_Class'])
    df['match_Class'] = label_encoder.fit_transform(df['match_Class'])
    
    # NOTE: Temperature normalization removed - should be handled by apply_normalization.py scripts

    X = df.drop(columns=["train_Chip"])
    X = X.iloc[:, :33]
    y = df['train_Class']

    # First split: 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        test_size=0.3,  # 30% for temp (validation + test)
        random_state=random_state,
        stratify=y  # Ensure class distribution is maintained
    )
    
    # Second split: 20% validation, 10% test (from the 30% temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.333,  # 10% of total = 1/3 of 30%
        random_state=random_state,
        stratify=y_temp  # Ensure class distribution is maintained
    )
    
    # Print actual split sizes for verification
    total_samples = len(X)
    print(f"\n=== Data Split Verification ===")
    print(f"Total samples: {total_samples}")
    print(f"Training: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/total_samples:.1%})")

    return (X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)

def load_and_preprocess_test_data(file_path, fraction=1, random_seed=42, 
                                stats_source='compute', stats_path=None,
                                apply_normalization=True, normalization_type='class_based', 
                                normalize_target_class=False):
    """
    Load and preprocess test data with optional normalization strategies.
    
    Args:
        file_path (str): Path to the test CSV file
        fraction (float): Fraction of data to use (default: 1.0)
        random_seed (int): Random seed for reproducibility
        stats_source (str): 'compute' to calculate stats, 'json' to load from file
        stats_path (str): Path to JSON stats file (required if stats_source='json')
        apply_normalization (bool): Whether to apply normalization (default: True)
        normalization_type (str): Type of normalization - 'class_based', 'standard', 'minmax', 'none'
        normalize_target_class (bool): Whether to normalize target class features (default: False)
        
    Returns:
        tuple: (X, y, label_encoder) - features, labels, and label encoder
    """
    # Load and visualize raw data
    df = pd.read_csv(file_path)
    
    plot_raw_test_mean_feature_per_class(
        df,
        class_column='Class',
        save_path='out/raw_test_mean_feature_per_class.png',
        title='Raw Test Mean Feature per Class'
    )
    
    # Shuffle and sample data
    df = df.sample(frac=fraction, random_state=random_seed).reset_index(drop=True)
    
    # Encode class labels
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])
    
    # Define columns for processing
    feature_columns = [col for col in df.columns if col not in ['Class', 'Chip']]
    peak_columns = [col for col in feature_columns if col.startswith('Peak')]
    temp_columns = [col for col in feature_columns if 'Temperature' in col]
    normalization_columns = peak_columns + temp_columns  # Include both Peak and Temperature columns
    
    # Separate features and labels
    X = df[feature_columns].copy()
    y = df['Class'].copy()
    
    target_class_encoded = target_class - 1  # Adjust for 0-based indexing
    
    # Apply normalization based on the chosen strategy
    if apply_normalization:
        if normalization_type == 'class_based':
            # Class-based mean/std normalization (using target class statistics)
            # For mean/std scaling, always normalize ALL classes including Class 4
            _apply_class_based_normalization(X, y, df, target_class_encoded, normalization_columns, 
                                           stats_source, stats_path, normalize_target_class=True)
            print(f"Applied class-based mean/std normalization (all classes normalized using Class 4 statistics)")
            
        elif normalization_type == 'standard':
            # Standard z-score normalization
            _apply_standard_normalization(X, normalization_columns)
            print("Applied standard z-score normalization")
            
        elif normalization_type == 'minmax':
            # Min-max normalization
            _apply_minmax_normalization(X, normalization_columns)
            print("Applied min-max normalization")
            
        elif normalization_type == 'class_based_minmax':
            # Class-based min-max normalization (using target class statistics)
            # For min-max scaling, always normalize ALL classes including Class 4
            _apply_class_based_minmax_normalization(X, y, df, target_class_encoded, normalization_columns, 
                                                  stats_source, stats_path, normalize_target_class=True)
            print(f"Applied class-based min-max normalization (all classes normalized using Class 4 statistics)")
            
        elif normalization_type == 'class_based_robust':
            # Class-based robust scaling normalization (using target class statistics)
            # For robust scaling, always normalize ALL classes including Class 4
            _apply_class_based_robust_normalization(X, y, df, target_class_encoded, normalization_columns, 
                                                   stats_source, stats_path, normalize_target_class=True)
            print(f"Applied class-based robust scaling normalization (all classes normalized using Class 4 statistics)")
            
        elif normalization_type == 'class_based_peak_to_peak':
            # Class-based peak-to-peak normalization (using target class statistics)
            _apply_class_based_peak_to_peak_normalization(X, y, df, target_class_encoded, normalization_columns, 
                                                         stats_source, stats_path, normalize_target_class)
            print(f"Applied class-based peak-to-peak normalization (target class normalization: {normalize_target_class})")
            
        elif normalization_type == 'none':
            print("No normalization applied")
            
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type}. "
                           "Choose from: 'class_based', 'standard', 'minmax', 'class_based_minmax', 'class_based_robust', 'class_based_peak_to_peak', 'none'")
    else:
        print("Normalization disabled")
    
    return X, y, label_encoder


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


def tensor_dataset_autoencoder(batch_size: int, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None):
    train_loader, val_loader, test_loader = None, None, None

    if X_train is not None and len(X_train) > 0:
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        X_train = X_train[:, :33]
        y_train = y_train[:, :33]
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)

    if X_val is not None and len(X_val) > 0:
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)
        X_val = X_val[:, :33]
        y_val = y_val[:, :33]
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True, drop_last=True)

    if X_test is not None and len(X_test) > 0:
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
        X_test = X_test[:, :33]
        y_test = y_test[:, :33]
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True, drop_last=False)

    return train_loader, val_loader, test_loader

def tensor_dataset_classifier(batch_size: int, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None):
    train_loader, val_loader, test_loader = None, None, None

    if X_train is not None and len(X_train) > 0:
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        X_train = X_train[:, :33]
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)

    if X_val is not None and len(X_val) > 0:
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.long)
        X_val = X_val[:, :33]
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True, drop_last=True)

    if X_test is not None and len(X_test) > 0:
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.long)
        X_test = X_test[:, :33]
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True, drop_last=False)

    return train_loader, val_loader, test_loader