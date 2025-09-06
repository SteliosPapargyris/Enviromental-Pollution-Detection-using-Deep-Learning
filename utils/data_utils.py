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
from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
    df['train_Temperature'] = (df['train_Temperature'] - df['train_Temperature'].mean()) / df['train_Temperature'].std()
    df.rename(columns={"train_Temperature": "train_Temperature_normalized"}, inplace=True)
    df['match_Temperature'] = (df['match_Temperature'] - df['match_Temperature'].mean()) / df['match_Temperature'].std()
    df.rename(columns={"match_Temperature": "match_Temperature_normalized"}, inplace=True)
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
    
    # Normalize the 'temperature' column using Z-score standardization
    df['train_Temperature'] = (df['train_Temperature'] - df['train_Temperature'].mean()) / df['train_Temperature'].std()
    df.rename(columns={"train_Temperature": "train_Temperature_normalized"}, inplace=True)

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
    peak_columns = [col for col in feature_columns if col != 'Temperature']
    
    # Separate features and labels
    X = df[feature_columns].copy()
    y = df['Class'].copy()
    
    target_class_encoded = target_class - 1  # Adjust for 0-based indexing
    
    # Apply normalization based on the chosen strategy
    if apply_normalization:
        if normalization_type == 'class_based':
            # Original class-based normalization
            _apply_class_based_normalization(X, y, df, target_class_encoded, peak_columns, 
                                           stats_source, stats_path, normalize_target_class)
            print(f"Applied class-based normalization (target class normalization: {normalize_target_class})")
            
        elif normalization_type == 'standard':
            # Standard z-score normalization
            _apply_standard_normalization(X, peak_columns)
            print("Applied standard z-score normalization")
            
        elif normalization_type == 'minmax':
            # Min-max normalization
            _apply_minmax_normalization(X, peak_columns)
            print("Applied min-max normalization")
            
        elif normalization_type == 'none':
            print("No normalization applied")
            
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type}. "
                           "Choose from: 'class_based', 'standard', 'minmax', 'none'")
    else:
        print("Normalization disabled")
    
    return X, y, label_encoder


def _apply_class_based_normalization(X, y, df, target_class_encoded, peak_columns, 
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
        
        peak_mean = normalization_data[peak_columns].mean().values
        peak_std = normalization_data[peak_columns].std().values
        temp_mean = normalization_data['Temperature'].mean()
        temp_std = normalization_data['Temperature'].std()
        
    else:
        raise ValueError("stats_source must be either 'compute' or 'json'")
    
    # Apply normalization
    _normalize_features(X, y, target_class_encoded, peak_columns, 
                       peak_mean, peak_std, temp_mean, temp_std, normalize_target_class)


def _apply_standard_normalization(X, peak_columns):
    """Apply standard z-score normalization to all features."""
    from sklearn.preprocessing import StandardScaler
    
    # Normalize peak features
    if peak_columns:
        scaler_peaks = StandardScaler()
        X[peak_columns] = scaler_peaks.fit_transform(X[peak_columns])
    
    # Normalize temperature
    if 'Temperature' in X.columns:
        scaler_temp = StandardScaler()
        X[['Temperature']] = scaler_temp.fit_transform(X[['Temperature']])


def _apply_minmax_normalization(X, peak_columns):
    """Apply min-max normalization to all features."""
    from sklearn.preprocessing import MinMaxScaler
    
    # Normalize peak features
    if peak_columns:
        scaler_peaks = MinMaxScaler()
        X[peak_columns] = scaler_peaks.fit_transform(X[peak_columns])
    
    # Normalize temperature
    if 'Temperature' in X.columns:
        scaler_temp = MinMaxScaler()
        X[['Temperature']] = scaler_temp.fit_transform(X[['Temperature']])


def _normalize_features(X, y, target_class_encoded, peak_columns, 
                       peak_mean, peak_std, temp_mean, temp_std, normalize_target_class=False):
    """
    Helper function to apply class-based normalization to features.
    
    Args:
        X (pd.DataFrame): Feature matrix to normalize
        y (pd.Series): Labels
        target_class_encoded (int): Encoded target class value
        peak_columns (list): List of peak column names
        peak_mean, peak_std (np.array): Mean and std for peak normalization
        temp_mean, temp_std (float): Mean and std for temperature normalization
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
        # Avoid division by zero
        peak_std_safe = np.where(peak_std == 0, 1, peak_std)
        X.loc[normalize_mask, peak_columns] = (
            X.loc[normalize_mask, peak_columns] - peak_mean
        ) / peak_std_safe
    
    # Normalize temperature for ALL classes (or subset based on normalize_target_class)
    if 'Temperature' in X.columns:
        temp_std_safe = temp_std if temp_std != 0 else 1
        if normalize_target_class:
            # Normalize temperature for all classes
            X['Temperature'] = (X['Temperature'] - temp_mean) / temp_std_safe
        else:
            # Normalize temperature for all classes (original behavior)
            X['Temperature'] = (X['Temperature'] - temp_mean) / temp_std_safe

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