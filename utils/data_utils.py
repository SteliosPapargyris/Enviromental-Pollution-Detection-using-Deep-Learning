import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np
import os

def dataset_creation(train_df) -> pd.DataFrame:

    merged_csv_path = "/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_4_of_every_chip/shuffled_dataset/merged.csv"

    # If file exists, skip creation and just return it
    if os.path.exists(merged_csv_path):
        return pd.read_csv(merged_csv_path)
    
    df2_compare = pd.read_csv("/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_4_of_every_chip/2.csv")
    merged_rows = []

    for _, train_row in train_df.iterrows():
        matching_rows = df2_compare[(df2_compare['Temperature'] == train_row['Temperature']) & (df2_compare['Class'] == train_row['Class'])]
        for _, match_row in matching_rows.iterrows():
            # Convert both rows to Series, reset index, and rename overlapping columns
            train_series = train_row.add_prefix("train_")
            match_series = match_row.add_prefix("match_")
            # Combine the two rows horizontally
            merged_row = pd.concat([train_series, match_series])
            merged_rows.append(merged_row)

    # Create a DataFrame from clean Series list
    merged_df = pd.DataFrame(merged_rows)

    # Remove duplicated columns
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    # Save the final merged file
    merged_df.to_csv("/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_4_of_every_chip/shuffled_dataset/merged.csv", index=False)
    return merged_df

def load_and_preprocess_data_autoencoder(file_path, test_size=0.1, random_state=42):
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

    # Split the data while maintaining Temperature and Class tracking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.222, random_state=random_state)
    return (X_train, y_train, X_val, y_val, X_test, y_test,label_encoder)


def load_and_preprocess_data_classifier(file_path, test_size=0.1, random_state=42):
    # Load data
    df = pd.read_csv(file_path)

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['train_Class'] = label_encoder.fit_transform(df['train_Class'])
    df['match_Class'] = label_encoder.fit_transform(df['match_Class'])
    X = df.drop(columns=["train_Chip", "train_Temperature"])
    X = X.iloc[:, :32]
    y = df['train_Class']

    # Split the data while maintaining Temperature and Class tracking
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.222, random_state=random_state)
    return (X_train, y_train, X_val, y_val, X_test, y_test,label_encoder)

def load_and_preprocess_test_data(file_path, fraction=1, random_seed=42):
    df = pd.read_csv(file_path)

    columns_to_normalize = [f'Peak {i}' for i in range(1, 33)]
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    df_copy = df.copy()  # Copy for normalization calculations

    # Take only a fraction of the dataset
    df = df.iloc[:int(len(df) * fraction)]

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])

    # Extract features and labels
    X = df.drop(['Class', 'Temperature', 'Chip'], axis=1)
    y = df['Class']

    # Get mean and std for class 4 normalization
    chip_column = "Chip"
    class_column = "Class"
    target_class = 4
    chip_5_target_rows = df_copy[(df_copy[chip_column] == 5) & (df_copy[class_column] == target_class)]
    mean_values = chip_5_target_rows[columns_to_normalize].mean(axis=0).to_numpy().reshape(1, -1)
    std_values = chip_5_target_rows[columns_to_normalize].std(axis=0).to_numpy().reshape(1, -1)

    # Normalize for non-class-4 samples
    exclude_class_4 = (df['Class'] != label_encoder.transform(['4'])[0])
    X[exclude_class_4] = (X[exclude_class_4] - mean_values) / std_values

    return X, y, label_encoder

#TODO --> do load_and_preprocess_data_classifier
def tensor_dataset_autoencoder(batch_size: int, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None):
    train_loader, val_loader, test_loader = None, None, None

    if X_train is not None and len(X_train) > 0:
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.float32)
        X_train = X_train[:, :32]
        y_train = y_train[:, :32]
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)

    if X_val is not None and len(X_val) > 0:
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.float32)
        X_val = X_val[:, :32]
        y_val = y_val[:, :32]
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True, drop_last=True)

    if X_test is not None and len(X_test) > 0:
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.float32)
        X_test = X_test[:, :32]
        y_test = y_test[:, :32]
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True, drop_last=False)

    return train_loader, val_loader, test_loader

# TODO fix this
def tensor_dataset_classifier(batch_size: int, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None):
    train_loader, val_loader, test_loader = None, None, None

    if X_train is not None and len(X_train) > 0:
        X_train = torch.tensor(X_train.values, dtype=torch.float32)
        y_train = torch.tensor(y_train.values, dtype=torch.long)
        X_train = X_train[:, :32]
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)

    if X_val is not None and len(X_val) > 0:
        X_val = torch.tensor(X_val.values, dtype=torch.float32)
        y_val = torch.tensor(y_val.values, dtype=torch.long)
        X_val = X_val[:, :32]
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True, drop_last=True)

    if X_test is not None and len(X_test) > 0:
        X_test = torch.tensor(X_test.values, dtype=torch.float32)
        y_test = torch.tensor(y_test.values, dtype=torch.long)
        X_test = X_test[:, :32]
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=True, drop_last=False)

    return train_loader, val_loader, test_loader