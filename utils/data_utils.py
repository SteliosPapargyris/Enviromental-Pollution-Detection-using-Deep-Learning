import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.plot_utils import plot_raw_test_mean_feature_per_class
import torch
import os
from typing import List
from utils.config import *

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

    # Split the data while maintaining Temperature and Class tracking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.222, random_state=random_state)
    return (X_train, y_train, X_val, y_val, X_test, y_test,label_encoder)


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

    # Split the data while maintaining Temperature and Class tracking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.222, random_state=random_state)
    return (X_train, y_train, X_val, y_val, X_test, y_test,label_encoder)

def load_and_preprocess_test_data(file_path, fraction=1, random_seed=42, 
                                stats_source='compute', stats_path=None):
    """
    Load and preprocess test data with normalization based on target class statistics.
    
    Args:
        file_path (str): Path to the test CSV file
        fraction (float): Fraction of data to use (default: 1.0)
        random_seed (int): Random seed for reproducibility
        stats_source (str): 'compute' to calculate stats, 'json' to load from file
        stats_path (str): Path to JSON stats file (required if stats_source='json')
        
    Returns:
        tuple: (X, y, label_encoder) - features, labels, and label encoder
    """
    import json
    import numpy as np
    
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
    
    # Get normalization statistics based on chosen method
    if stats_source == 'json':
        if stats_path is None:
            raise ValueError("stats_path must be provided when stats_source='json'")
        
        # Load statistics from JSON
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        peak_mean = np.array(stats['mean'])
        peak_std = np.array(stats['std'])
        
        # Handle temperature stats - you may need to adjust this based on your JSON structure
        if 'Temperature' in feature_columns:
            # Option 1: Calculate temperature stats (if not in JSON)
            normalization_mask = (df['Chip'] == chip_exclude) & (df['Class'] == target_class_encoded)
            normalization_data = df[normalization_mask]
            
            if normalization_data.empty:
                raise ValueError(f"No data found for target class {target_class} in chip {chip_exclude}")
            
            temp_mean = normalization_data['Temperature'].mean()
            temp_std = normalization_data['Temperature'].std()
            
            # Option 2: Load from JSON if available (uncomment if you save temp stats)
            # temp_mean = stats.get('temp_mean', temp_mean)
            # temp_std = stats.get('temp_std', temp_std)
        
        print(f"Loaded normalization statistics from: {stats_path}")
        
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
        
        print("Computed normalization statistics from test data")
        
    else:
        raise ValueError("stats_source must be either 'compute' or 'json'")
    
    # Apply normalization
    _normalize_features(X, y, target_class_encoded, peak_columns, 
                       peak_mean, peak_std, temp_mean, temp_std)
    
    return X, y, label_encoder


def _normalize_features(X, y, target_class_encoded, peak_columns, 
                       peak_mean, peak_std, temp_mean, temp_std):
    """
    Helper function to apply normalization to features.
    
    Args:
        X (pd.DataFrame): Feature matrix to normalize
        y (pd.Series): Labels
        target_class_encoded (int): Encoded target class value
        peak_columns (list): List of peak column names
        peak_mean, peak_std (np.array): Mean and std for peak normalization
        temp_mean, temp_std (float): Mean and std for temperature normalization
    """
    # Normalize peak features for non-target classes only
    non_target_mask = (y != target_class_encoded)
    
    if peak_columns and non_target_mask.sum() > 0:
        # Avoid division by zero
        peak_std_safe = np.where(peak_std == 0, 1, peak_std)
        X.loc[non_target_mask, peak_columns] = (
            X.loc[non_target_mask, peak_columns] - peak_mean
        ) / peak_std_safe
    
    # Normalize temperature for ALL classes
    if 'Temperature' in X.columns:
        temp_std_safe = temp_std if temp_std != 0 else 1
        X['Temperature'] = (X['Temperature'] - temp_mean) / temp_std_safe


def get_normalization_statistics(df, chip_id, class_id, feature_columns):
    """
    Extract normalization statistics from specific chip and class.
    
    Args:
        df (pd.DataFrame): Input dataframe
        chip_id (int): Chip identifier
        class_id (int): Class identifier
        feature_columns (list): Columns to calculate statistics for
        
    Returns:
        tuple: (mean_values, std_values) as numpy arrays
    """
    subset = df[(df['Chip'] == chip_id) & (df['Class'] == class_id)]
    
    if subset.empty:
        raise ValueError(f"No data found for chip {chip_id}, class {class_id}")
    
    mean_values = subset[feature_columns].mean().values
    std_values = subset[feature_columns].std().values
    
    return mean_values, std_values


def validate_normalization_data(df, required_columns):
    """
    Validate that the dataframe contains required columns and data.
    
    Args:
        df (pd.DataFrame): Input dataframe
        required_columns (list): List of required column names
        
    Raises:
        ValueError: If validation fails
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if df.empty:
        raise ValueError("Input dataframe is empty")
    
    # Check for sufficient data per class
    class_counts = df['Class'].value_counts()
    if class_counts.min() < 5:  # Arbitrary threshold
        print(f"Warning: Some classes have very few samples: {class_counts.to_dict()}")

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

# TODO fix this
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