import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.plot_utils import plot_raw_test_mean_feature_per_class
import torch
import os
import numpy as np
from typing import List
from utils.config import *
from utils.normalization_techniques import _apply_class_based_normalization, _apply_standard_normalization, _apply_minmax_normalization, _apply_class_based_minmax_normalization, _apply_class_based_robust_normalization

def dataset_creation(csv_indices: List[int], baseline_chip: int = None) -> List[pd.DataFrame]:
    """
    Create separate datasets for each CSV, matching each row with rows from the same chip
    that have the same class and temperature = 25°C.

    Args:
        csv_indices: List of CSV indices to process
        baseline_chip: Not used (kept for compatibility)

    Returns:
        List of DataFrames, one for each input CSV
    """
    result_dfs = []

    for csv_idx in csv_indices:
        # Load individual CSV
        df_current = pd.read_csv(f"data/out/mean_std/chip_{csv_idx}_mean_std.csv")
        # Filter for 25°C temperature from the same chip
        df_current_25c = df_current[df_current['Temperature'] == 25]
        merged_rows = []

        for _, train_row in df_current.iterrows():
            # Match with rows from the same chip that have same class and temp=25°C
            matching_rows = df_current_25c[df_current_25c['Class'] == train_row['Class']]

            for _, match_row in matching_rows.iterrows():
                train_series = train_row.add_prefix("train_")
                match_series = match_row.add_prefix("match_")
                merged_row = pd.concat([train_series, match_series])
                merged_rows.append(merged_row)

        # Create DataFrame for this CSV
        merged_df = pd.DataFrame(merged_rows)

        # Shuffle the data
        merged_df = merged_df.sample(frac=1).reset_index(drop=True)

        # Save to separate file
        output_dir = os.path.join("data/out/", "shuffled_dataset")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{csv_idx}_self_match_25C.csv")
        merged_df.to_csv(output_path, index=False)

        result_dfs.append(merged_df)
        print(f"Created dataset for CSV {csv_idx}: {len(merged_df)} samples saved to {output_path}")

    return result_dfs

def load_and_preprocess_data_autoencoder(file_path, random_state=42, finetune=False):
    # Load data
    df = pd.read_csv(file_path)

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['train_Class'] = label_encoder.fit_transform(df['train_Class'])
    df['match_Class'] = label_encoder.fit_transform(df['match_Class'])

    X = df.drop(columns=["train_Chip"])
    X = X.iloc[:, :33]
    y = df.drop(columns=["match_Chip"])
    y = y.iloc[:, 35:-1]

    if finetune == False:
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
    else:
        # 80-20 split (train-val only, no test)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,  # 20% for validation
            random_state=random_state
        )
        X_test, y_test = pd.DataFrame(), pd.DataFrame()

    print(f"\n=== Autoencoder Data Split ===")
    total_samples = len(X)
    print(f"Training: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/total_samples:.1%})")

    return (X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)

def load_and_preprocess_data_classifier(file_path, random_state=42, finetune=False):
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
    
    if finetune == False:
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
    else:
        # 80-20 split (train-val only, no test)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y,
            test_size=0.2,  # 20% for validation
            random_state=random_state,
            stratify=y  # Ensure class distribution is maintained
        )
        X_test, y_test = pd.DataFrame(), pd.DataFrame()
    
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
                                normalize_target_class=True):
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
        normalize_target_class (bool): Whether to normalize target class features (default: True)
        
    Returns:
        tuple: (X, y, label_encoder) - features, labels, and label encoder
    """
    # Load raw data
    df_full = pd.read_csv(file_path)

    plot_raw_test_mean_feature_per_class(
        df_full,
        class_column='Class',
        save_path='out/raw/raw_test_mean_feature_per_class.png',
        title='Raw Test Mean Feature per Class'
    )

    # Shuffle and sample data AFTER computing normalization stats from full dataset
    df = df_full.sample(frac=fraction, random_state=random_seed).reset_index(drop=True)
    
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
        if normalization_type == 'class_based_mean_std':
            # Class-based mean/std normalization (using target class statistics)
            # For mean/std scaling, always normalize ALL classes including Class 4
            _apply_class_based_normalization(X, y, df_full, target_class_encoded, normalization_columns,
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
            _apply_class_based_minmax_normalization(X, y, df_full, target_class_encoded, normalization_columns,
                                                  stats_source, stats_path, normalize_target_class=True)
            print(f"Applied class-based min-max normalization (all classes normalized using Class 4 statistics)")
            
        elif normalization_type == 'class_based_robust':
            # Class-based robust scaling normalization (using target class statistics)
            # For robust scaling, always normalize ALL classes including Class 4
            _apply_class_based_robust_normalization(X, y, df_full, target_class_encoded, normalization_columns,
                                                   stats_source, stats_path, normalize_target_class=True)
            print(f"Applied class-based robust scaling normalization (all classes normalized using Class 4 statistics)")

        elif normalization_type == 'none':
            print("No normalization applied")
            
        else:
            raise ValueError(f"Unknown normalization_type: {normalization_type}. "
                           "Choose from: 'class_based', 'standard', 'minmax', 'class_based_minmax', 'class_based_robust', 'class_based_peak_to_peak', 'none'")
    else:
        print("Normalization disabled")
    
    return X, y, label_encoder

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

def load_and_preprocess_data_autoencoder_peaks_only(file_path, random_state=42, finetune=False):
    """
    Load and preprocess data for autoencoder training with peaks-only approach.

    Input: 32 peaks (excluding Temperature)
    Target: Same data but only T=25°C rows, normalized via apply_normalization

    Args:
        file_path: Path to the CSV file
        random_state: Random state for splitting
        finetune: Whether to use train/val split only (no test)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
        where X contains 32 peaks and y contains normalized 32 peaks from T=25°C
    """
    from utils.apply_normalization import apply_normalization

    # Load data
    df = pd.read_csv(file_path)

    # Encode Class labels for any class-related operations
    label_encoder = LabelEncoder()
    if "train_Class" in df.columns:
        df["train_Class"] = label_encoder.fit_transform(df["train_Class"])
    elif "Class" in df.columns:
        df["Class"] = label_encoder.fit_transform(df["Class"])

    # Apply normalization to get normalized T=25°C baseline data
    normalized_datasets = apply_normalization([df])
    normalized_df = normalized_datasets[0]

    # Extract peak columns (32 peaks) from original data as input
    if "train_Peak 1" in df.columns:
        # Paired data format
        peak_cols = [col for col in df.columns if col.startswith("train_Peak") and col != "train_Peak Temperature"]
        peak_cols = sorted(peak_cols, key=lambda x: int(x.split()[-1]))[:32]  # First 32 peaks
        X_input = df[peak_cols]

        # Extract normalized peak data as target (T=25°C equivalent)
        normalized_peak_cols = [col for col in normalized_df.columns if col.startswith("train_Peak") and col != "train_Peak Temperature"]
        normalized_peak_cols = sorted(normalized_peak_cols, key=lambda x: int(x.split()[-1]))[:32]  # First 32 peaks
        y_target = normalized_df[normalized_peak_cols]

    else:
        # Original data format (Peak 1, Peak 2, etc.)
        peak_cols = [f"Peak {i}" for i in range(1, 33)]  # Peak 1 to Peak 32
        available_peaks = [col for col in df.columns if col in peak_cols]
        X_input = df[available_peaks]
        y_target = normalized_df[available_peaks]

    print(f"=== Peaks-Only Autoencoder Data Preparation ===")
    print(f"Input features (32 peaks): {X_input.shape[1]} columns")
    print(f"Target features (normalized 32 peaks): {y_target.shape[1]} columns")

    if finetune == False:
        # 70-20-10 split
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_input, y_target,
            test_size=0.3,  # 30% for temp
            random_state=random_state
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=0.333,  # 10% of total
            random_state=random_state
        )
    else:
        # 80-20 split (train-val only, no test)
        X_train, X_val, y_train, y_val = train_test_split(
            X_input, y_target,
            test_size=0.2,  # 20% for validation
            random_state=random_state
        )
        X_test, y_test = pd.DataFrame(), pd.DataFrame()

    print(f"=== Peaks-Only Autoencoder Data Split ===")
    total_samples = len(X_input)
    print(f"Training: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/total_samples:.1%})")

    return (X_train, y_train, X_val, y_val, X_test, y_test, label_encoder)

def tensor_dataset_autoencoder_peaks_only(batch_size: int, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None, indices_train=None, indices_val=None, indices_test=None):
    """
    Create tensor datasets for autoencoder (33 input features: 32 peaks + 1 chip, 32 target features: peaks only)
    """
    train_loader, val_loader, test_loader = None, None, None

    if X_train is not None and len(X_train) > 0:
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        # Use all input features (33: 32 peaks + 1 chip for X, 32 peaks for y)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=False, drop_last=False)

    if X_val is not None and len(X_val) > 0:
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False)

    if X_test is not None and len(X_test) > 0:
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader, test_loader, indices_train, indices_val, indices_test



def load_and_preprocess_data_autoencoder_prenormalized(file_path, random_state=42, finetune=False):
    """
    Load and preprocess normalized data for autoencoder training with peaks-only approach.
    This function assumes normalization has already been applied to the dataset.

    Input: train_Peak columns (32 peaks) + train_Chip
    Target: match_Peak columns (32 peaks)

    Args:
        file_path: Path to the CSV file (should contain normalized data)
        random_state: Random state for splitting
        finetune: Whether to use train/val split only (no test)

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, indices_train, indices_val, indices_test
        where X contains train_Peak columns + train_Chip and y contains match_Peak columns
    """
    # Load pre-normalized data
    df = pd.read_csv(file_path)

    # Encode Class labels for any class-related operations
    label_encoder = LabelEncoder()
    if "train_Class" in df.columns:
        df["train_Class"] = label_encoder.fit_transform(df["train_Class"])
    elif "Class" in df.columns:
        df["Class"] = label_encoder.fit_transform(df["Class"])

    # Extract peak columns (32 peaks) - input: train_Peak + train_Chip, target: match_Peak
    if "train_Peak 1" in df.columns:
        # Paired data format - input: train_Peak + train_Chip, target: match_Peak
        train_peak_cols = [col for col in df.columns if col.startswith("train_Peak") and "Temperature" not in col]
        train_peak_cols = sorted(train_peak_cols, key=lambda x: int(x.split()[-1]))[:32]  # First 32 peaks

        # Add train_Chip to input features
        input_cols = train_peak_cols + ["train_Chip"]
        X_input = df[input_cols]

        match_peak_cols = [col for col in df.columns if col.startswith("match_Peak") and "Temperature" not in col]
        match_peak_cols = sorted(match_peak_cols, key=lambda x: int(x.split()[-1]))[:32]  # First 32 peaks
        y_target = df[match_peak_cols]

    else:
        # Original data format (Peak 1, Peak 2, etc.) - input: Peaks + Chip, target are the same
        peak_cols = [f"Peak {i}" for i in range(1, 33)]  # Peak 1 to Peak 32
        available_peaks = [col for col in df.columns if col in peak_cols]

        # Add Chip to input features if available
        input_cols = available_peaks + (["Chip"] if "Chip" in df.columns else [])
        X_input = df[input_cols]
        y_target = df[available_peaks]

    print(f"=== Normalized Autoencoder Data Preparation ===")
    print(f"Input features (train_Peak + train_Chip): {X_input.shape[1]} columns")
    print(f"Target features (match_Peak): {y_target.shape[1]} columns")

    # Create indices array to track original row positions
    indices = np.arange(len(X_input))

    if finetune == False:
        # 70-20-10 split
        X_train, X_temp, y_train, y_temp, indices_train, indices_temp = train_test_split(
            X_input, y_target, indices,
            test_size=0.3,  # 30% for temp
            random_state=random_state
        )

        X_val, X_test, y_val, y_test, indices_val, indices_test = train_test_split(
            X_temp, y_temp, indices_temp,
            test_size=0.333,  # 10% of total
            random_state=random_state
        )
    else:
        # 80-20 split (train-val only, no test)
        X_train, X_val, y_train, y_val, indices_train, indices_val = train_test_split(
            X_input, y_target, indices,
            test_size=0.2,  # 20% for validation
            random_state=random_state
        )
        X_test, y_test = pd.DataFrame(), pd.DataFrame()
        indices_test = np.array([])

    print(f"=== Normalized Autoencoder Data Split ===")
    total_samples = len(X_input)
    print(f"Training: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
    print(f"Validation: {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
    print(f"Test: {len(X_test)} samples ({len(X_test)/total_samples:.1%})")

    return (X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, indices_train, indices_val, indices_test)

