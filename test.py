import torch
from utils.data_utils import create_dataloaders
from utils.train_test_utils import evaluate_encoder_decoder, evaluate_classifier
from utils.plot_utils import plot_conf_matrix, plot_macro_roc_curve
from utils.models import ConvDenoiser, Classifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import matplotlib
import random

matplotlib.use('Agg')  # Use a non-interactive backend
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_denoiser_name = 'denoiser_model'
model_classifier_name = 'classifier_model'
conv_layers = 'enc_dec'
chip_number = 5  # Adjust chip number context if necessary
base_path = "data/mean_and_std_of_class_4_of_every_chip"
test_file_path = f'{base_path}/5.csv'
columns_to_normalize = [f'Peak {i}' for i in range(1, 33)]


def load_and_preprocess_test_data(file_path, fraction=1, random_seed=42):
    """
    Load, shuffle, and preprocess test data using mean normalization for Chip 5.

    Parameters:
    - file_path (str): Path to the test data file.
    - fraction (float): Fraction of the dataset to use (e.g., 1/3 for one-third).
    - random_seed (int): Seed for reproducible shuffling.

    Returns:
    - X (np.ndarray): Preprocessed feature matrix.
    - y (np.ndarray): Target labels.
    - label_encoder (LabelEncoder): Fitted label encoder for class labels.
    """
    # Load test data
    df = pd.read_csv(file_path)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    df_copy = df.copy()

    # Take only a fraction of the dataset
    df = df.iloc[:int(len(df) * fraction)]

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])

    # Separate features and labels
    X = df.drop(['Class', 'Temperature', 'Chip'], axis=1).to_numpy()
    y = df['Class'].to_numpy()

    # Find the finite minimum and maximum values in X
    finite_min = np.nanmin(X[np.isfinite(X)])
    finite_max = np.nanmax(X[np.isfinite(X)])

    # Replace -inf with finite_min and inf with finite_max
    X = np.where(X == -np.inf, finite_min, X)
    X = np.where(X == np.inf, finite_max, X)

    X = X + X * 0.02


    chip_column = "Chip"
    class_column = "Class"
    target_class = 4
    columns_to_normalize = [col for col in df.columns if col.startswith("Peak")]

    # Filter rows for chip 5 and target class
    chip_5_target_rows = df_copy[(df_copy[chip_column] == 5) & (df_copy[class_column] == target_class)]

    # Compute the mean and standard deviation for the specified columns
    mean_values = chip_5_target_rows[columns_to_normalize].mean(axis=0)
    std_values = chip_5_target_rows[columns_to_normalize].std(axis=0)

    # Convert mean and std values to numpy arrays
    mean_values = mean_values.to_numpy().reshape(1, -1)  # Shape (1, num_features)
    std_values = std_values.to_numpy().reshape(1, -1)  # Shape (1, num_features)

    # Identify rows where class is NOT 4
    exclude_class_4 = (df['Class'] != label_encoder.transform(['4'])[0])

    # Normalize using (X - mean) / std for rows where class is NOT 4
    X[exclude_class_4] = (X[exclude_class_4] - mean_values) / std_values

    # Reshape for the model input (assuming a sequence length of 32)
    X = X.reshape(-1, 1, 32)

    return X, y, label_encoder



# Load and preprocess test data
X_test, y_test, label_encoder = load_and_preprocess_test_data(file_path=test_file_path, fraction=1)

# Create only the test DataLoader
_, _, test_loader = create_dataloaders(X_test=X_test, y_test=y_test, batch_size=32)

# Initialize ConvDenoiser model
model_denoiser = ConvDenoiser().to(device)
model_classifier = Classifier().to(device)

denoiser_path = f'pths/denoiser_model_{chip_number - 1}.pth'
classifier_path = f'pths/classifier_model_{chip_number - 1}.pth'
model_denoiser.load_state_dict(torch.load(denoiser_path))
model_classifier.load_state_dict(torch.load(classifier_path))

import torch.nn as nn
# Define loss function, optimizer, and scheduler
criterion = nn.MSELoss()

# Evaluate the model on the test set
avg_test_loss = evaluate_encoder_decoder(
    model_encoder_decoder=model_denoiser,
    data_loader=test_loader,
    criterion=criterion,
    device=device,
    label_encoder=label_encoder,
    model_name='denoiser',
    conv_layers=conv_layers,
    chip_number=chip_number
)

# Evaluate the model on the test set
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    data_loader=test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name='denoiser_and_classifier',
)

# Plot confusion matrix
plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_model', chip_number=chip_number)