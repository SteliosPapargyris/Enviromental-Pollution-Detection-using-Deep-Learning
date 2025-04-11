import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder
from utils.data_utils import create_dataloaders
from utils.train_test_utils import evaluate_encoder_decoder, evaluate_classifier
from utils.plot_utils import plot_conf_matrix
from utils.models import ConvDenoiser, Classifier
import matplotlib

seed = 42
matplotlib.use('Agg')  # Use a non-interactive backend
torch.manual_seed(seed), torch.cuda.manual_seed_all(seed), np.random.seed(seed), random.seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# File paths and settings
chip_number = 5  # Chip number for testing
base_path = "/Users/steliospapargyris/Documents/MyProjects/data_thesis/shuffled_dataset/mean_and_std_of_class_4_of_every_chip"
test_file_path = f'{base_path}/5.csv'
batch_size = 32  # Adjust batch size if needed

# Columns to normalize
columns_to_normalize = [f'Peak {i}' for i in range(1, 33)]


def load_and_preprocess_test_data(file_path, fraction=1, random_seed=42):
    df = pd.read_csv(file_path)

    # Shuffle dataset
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    df_copy = df.copy()  # Copy for normalization calculations

    # Take only a fraction of the dataset
    df = df.iloc[:int(len(df) * fraction)]

    # Encode 'Class' labels
    label_encoder = LabelEncoder()
    df['Class'] = label_encoder.fit_transform(df['Class'])

    # Extract features and labels
    X = df.drop(['Class', 'Temperature', 'Chip'], axis=1).to_numpy()
    y = df['Class'].to_numpy()

    # Handle infinite values
    finite_min = np.nanmin(X[np.isfinite(X)])
    finite_max = np.nanmax(X[np.isfinite(X)])
    X = np.where(X == -np.inf, finite_min, X)
    X = np.where(X == np.inf, finite_max, X)

    X = X + X * 0.02  # Data perturbation (same as in training)

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

    # Reshape input for models (batch_size, channels, features)
    X = X.reshape(-1, 1, 32)

    return X, y, label_encoder

# Load and preprocess test data
X_test, y_test, label_encoder = load_and_preprocess_test_data(file_path=test_file_path, fraction=1)

# Create test DataLoader
_, _, test_loader, *_ = create_dataloaders(X_test=X_test, y_test=y_test, batch_size=batch_size)

denoiser_path = "pths/denoiser_model.pth"
classifier_path = "pths/classifier_train.pth"

# Initialize ConvDenoiser model
model_denoiser = ConvDenoiser().to(device)
model_classifier = Classifier().to(device)

# Set models to evaluation mode
model_denoiser.eval()
model_classifier.eval()

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
)

# Evaluate classifier
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    data_loader=test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name='denoiser_and_classifier_test'
)

plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_test')