import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from utils.plot_utils import plot_raw_test_mean_feature_per_class
import torch
import os
from typing import List
from utils.config import *
from utils.normalization_techniques import _apply_class_based_normalization, _apply_standard_normalization, _apply_minmax_normalization, _apply_class_based_minmax_normalization, _apply_class_based_robust_normalization

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
    
    # Save to file
    os.makedirs(os.path.dirname(merged_csv_path), exist_ok=True)
    merged_df.to_csv(merged_csv_path, index=False)
    return merged_df

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