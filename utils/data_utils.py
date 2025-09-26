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

def dataset_creation(csv_indices: List[int], baseline_chip: int = None, norm_method: str = None) -> List[pd.DataFrame]:
    """
    Create separate datasets for each CSV, matching each row with rows from the same chip
    that have the same class and temperature = 25°C.

    Args:
        csv_indices: List of CSV indices to process
        baseline_chip: Not used (kept for compatibility)
        norm_method: Normalization method folder name (e.g., 'mean_std', 'minmax', 'robust')

    Returns:
        List of DataFrames, one for each input CSV
    """
    result_dfs = []

    for csv_idx in csv_indices:
        # Load individual CSV using the specified normalization method
        df_current = pd.read_csv(f"data/out/{norm_method}/{total_num_chips}chips/chip_{csv_idx}_{norm_method}.csv")

        # Filter for 27°C temperature from the same chip - get all rows within tolerance
        df_current_27c = df_current[abs(df_current['Temperature'] - 27.0) <= 0.03]

        # Average within each class to get one representative row per class
        df_current_27c = df_current_27c.groupby('Class').mean().reset_index()
        merged_rows = []

        for _, train_row in df_current.iterrows():
            # Match with rows from the same chip that have same class and temp=25°C
            matching_rows = df_current_27c[df_current_27c['Class'] == train_row['Class']]

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
        output_path = os.path.join(output_dir, f"{csv_idx}_self_match_27C.csv")
        merged_df.to_csv(output_path, index=False)

        result_dfs.append(merged_df)
        print(f"Created dataset for CSV {csv_idx}: {len(merged_df)} samples saved to {output_path}")

    return result_dfs

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

