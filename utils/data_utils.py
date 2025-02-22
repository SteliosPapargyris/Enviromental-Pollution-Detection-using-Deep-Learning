import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch


def load_and_preprocess_data(file_path='data/train.csv', test_size=0.1, random_state=42):
    # Load data
    df = pd.read_csv(file_path)

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])

    # Prepare features and labels for train and test data
    X = df.drop(['Class', 'Temperature', 'Chip'], axis=1).to_numpy()
    y = df['Class'].to_numpy()

    # Reshape data for model input: [batch_size, channels, sequence_length]
    X = X.reshape(-1, 1, 32)

    # Split the training data into training and validation sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.222, random_state=random_state)

    # Initialize placeholders for denoised versions
    X_denoised_train, X_denoised_val, X_denoised_test = None, None, None

    return X_train, y_train, X_val, y_val, X_test, y_test, X_denoised_train, X_denoised_val, X_denoised_test, label_encoder


def create_dataloaders(batch_size: int, X_train=None, y_train=None, X_val=None, y_val=None, X_test=None, y_test=None,
                        X_denoised_train=None, X_denoised_val=None, X_denoised_test=None):

    train_loader, val_loader, test_loader = None, None, None
    denoised_train_loader, denoised_val_loader, denoised_test_loader = None, None, None


    if X_train is not None and len(X_train) > 0:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=False)

    if X_val is not None and len(X_val) > 0:
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=True, drop_last=False)

    if X_test is not None and len(X_test) > 0:
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)

    # Denoised data loaders
    if X_denoised_train is not None:
        X_denoised_train = torch.tensor(X_denoised_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        denoised_train_loader = DataLoader(TensorDataset(X_denoised_train, y_train), batch_size=batch_size,
                                           shuffle=True, drop_last=True)

    if X_denoised_val is not None:
        X_denoised_val = torch.tensor(X_denoised_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.long)
        denoised_val_loader = DataLoader(TensorDataset(X_denoised_val, y_val), batch_size=batch_size, shuffle=True,
                                         drop_last=True)

    if X_denoised_test is not None:
        X_denoised_test = torch.tensor(X_denoised_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.long)
        denoised_test_loader = DataLoader(TensorDataset(X_denoised_test, y_test), batch_size=batch_size,
                                          shuffle=False, drop_last=True)

    return train_loader, val_loader, test_loader, denoised_train_loader, denoised_val_loader, denoised_test_loader