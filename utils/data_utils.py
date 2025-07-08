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

def load_and_preprocess_test_data(file_path, fraction=1, random_seed=42):
    df = pd.read_csv(file_path)

    # columns_to_normalize = ['Temperature'] + [f'Peak {i}' for i in range(1, 33)]
    columns_to_normalize = [f'Peak {i}' for i in range(1, 33)]

    plot_raw_test_mean_feature_per_class(
    df,
    class_column='Class',
    save_path='out/raw_test_mean_feature_per_class.png',
    title='Raw Test Mean Feature per Class'
)
    # Shuffle dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    df_copy = df.copy()
    # Take only a fraction of the dataset
    df = df.iloc[:int(len(df) * fraction)]

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])

    # Extract features and labels
    # X = df.drop(['Class', 'Chip'], axis=1)
    X = df.drop(['Class', 'Temperature', 'Chip'], axis=1)
    y = df['Class']

    # Get mean and std for class 4 normalization
    chip_5_target_rows = df_copy[(df_copy[chip_column] == chip_exclude) & (df_copy[class_column] == target_class)]
    mean_values= chip_5_target_rows[columns_to_normalize].mean(axis=0).to_numpy().reshape(1, -1)
    std_values = chip_5_target_rows[columns_to_normalize].std(axis=0).to_numpy().reshape(1, -1)
    # mean_values = np.load("/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_4_of_every_chip/class_4_mean_and_std/fts_mzi_dataset/mean_statistics/mean_class_4.npy")
    # std_values = np.load("/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_4_of_every_chip/class_4_mean_and_std/fts_mzi_dataset/std_statistics/std_class_4.npy")
    # Normalize for non-class-4 samples
    exclude_class_4 = (df['Class'] != label_encoder.transform(['4'])[0])
    X[exclude_class_4] = (X[exclude_class_4] - mean_values) / (np.log10(std_values) * 0.1)

    # # Min-Max normalization per row for all samples
    # row_min = X.min(axis=1)
    # row_max = X.max(axis=1)
    # denominator = (row_max - row_min).replace(0, 1)  # Avoid divide-by-zero

    # X_normalized = (X.subtract(row_min, axis=0)
    #             .div(denominator, axis=0))
    
    # # Min-Max normalization per row for samples with Class 4 excluded
    # # Identify Class 4 (to exclude from normalization)
    # class_4_encoded = label_encoder.transform(['4'])[0]
    # mask_not_class4 = (df['Class'] != class_4_encoded)

    # # Min-Max normalization per row for samples not in Class 4
    # row_min = X[mask_not_class4].min(axis=1)
    # row_max = X[mask_not_class4].max(axis=1)
    # denominator = (row_max - row_min).replace(0, 1)

    # X.loc[mask_not_class4] = (
    #     X.loc[mask_not_class4].subtract(row_min, axis=0)
    #                           .div(denominator, axis=0)
    # )

    # # --- Min-Max Normalization ---
    # col_stats_path = f'/Users/steliospapargyris/Documents/MyProjects/data_thesis/per_peak_minmax_excl_chip_class{target_class}/fts_mzi_dataset/{chip_exclude}chips_20percent_noise/col_minmax_stats_excl_chip{chip_exclude}_class{target_class}.csv'
    # stats_df = pd.read_csv(col_stats_path, index_col=0)

    # # Step 1: Add a feature index (assuming order matters and matches X's columns)
    # stats_df['feature_idx'] = stats_df.groupby('chip').cumcount()
    # stats_df = stats_df.reset_index() 

    # # Step 2: Pivot so each row is one feature, each column is a chip's value
    # min_df = stats_df.pivot(index='feature_idx', columns='chip', values='min')
    # max_df = stats_df.pivot(index='feature_idx', columns='chip', values='max')

    # # Step 3: Compute the average across chips (i.e., across columns)
    # avg_min = min_df.mean(axis=1)
    # avg_max = max_df.mean(axis=1)

    # # Prevent divide-by-zero by replacing 0s in the denominator
    # avg_range = (avg_max - avg_min).replace(0, 1)

    # # Step 4: Mask for class ≠ 4 (inference-time known labels)
    # class_mask = y != 3

    # # # --- Z-score Normalization ---
    # stats_df = pd.read_csv(col_stats_path, index_col=0)
    
    # # Step 1: Add a feature index (assuming order matters and matches X's columns)
    # stats_df['feature_idx'] = stats_df.groupby('chip').cumcount()
    # stats_df = stats_df.reset_index() 

    # # Step 2: Pivot so each row is one feature, each column is a chip's value
    # mean_df = stats_df.pivot(index='feature_idx', columns='chip', values='mean')
    # std_df = stats_df.pivot(index='feature_idx', columns='chip', values='std')

    # # Step 3: Compute the average across chips (i.e., across columns)
    # avg_mean = mean_df.mean(axis=1)
    # avg_std = std_df.replace(0, 1).mean(axis=1)

    # # avg_mean = mean_df.iloc[0:33, 0]
    # # avg_std = std_df.iloc[0:33, 0]

    # # Step 4: Mask for class ≠ 4 (inference-time known labels)
    # class_mask = y != 3

    # # Step 5: Apply normalization only to non-class-4 rows
    # X_normalized = X.copy()
    # X_normalized.loc[class_mask] = (
    #     X.loc[class_mask].subtract(avg_mean.values, axis=1).div(avg_std.values, axis=1)
    # )

    # KLEPSIMO --> FROM THE SAME CHIP

    # chip_ids = df['Chip']

    # # Step 1: Create a copy for normalized data
    # X_normalized = X.copy()

    # # Step 2: Identify non-Class-4 samples
    # class_mask = y != 3

    # # Step 3: Normalize each sample using the mean and std of samples from the same chip
    # for chip in chip_ids.unique():
    #     # Get mask for samples of this chip and non-Class-4
    #     chip_mask = (chip_ids == chip) & class_mask

    #     # Compute mean and std from these rows
    #     chip_X = X.loc[chip_mask]
    #     chip_mean = chip_X.mean()
    #     chip_std = chip_X.std().replace(0, 1)  # Avoid division by zero

    #     # Apply normalization
    #     X_normalized.loc[chip_mask] = (chip_X - chip_mean) / chip_std
        
    #  # --- Robust Normalization: (x - median) / IQR ---
    # row_median = X.median(axis=1)
    # row_q75 = X.quantile(0.75, axis=1)
    # row_q25 = X.quantile(0.25, axis=1)
    # row_iqr = (row_q75 - row_q25).replace(0, 1)  # Avoid divide-by-zero

    # X_robust = X.subtract(row_median, axis=0).div(row_iqr, axis=0)

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