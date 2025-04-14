import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from utils.data_utils import tensor_dataset_classifier
from utils.train_test_utils import evaluate_encoder_decoder_for_classifier, evaluate_classifier
from utils.plot_utils import plot_conf_matrix
from utils.models import ConvDenoiser, Classifier
from utils.config import *

# File paths and settings
chip_number = 5  # Chip number for testing
test_file_path = f'{base_path}/5.csv'

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

# Load and preprocess test data
X_test, y_test, label_encoder = load_and_preprocess_test_data(file_path=test_file_path, fraction=1)

# Create test DataLoader
_, _, test_loader = tensor_dataset_classifier(batch_size=batch_size, X_test=X_test, y_test=y_test)

autoencoder_path = "pths/autoencoder_train.pth"
classifier_path = "pths/classifier_train.pth"

# Initialize ConvDenoiser model
model_autoencoder = ConvDenoiser().to(device)
model_classifier = Classifier().to(device)

# Set models to evaluation mode
model_autoencoder.eval()
model_classifier.eval()

model_autoencoder.load_state_dict(torch.load(autoencoder_path))
model_classifier.load_state_dict(torch.load(classifier_path))

# Define loss function, optimizer, and scheduler
criterion = nn.MSELoss()

# Evaluate the model on the test set
X_test_denoised, y_test = evaluate_encoder_decoder_for_classifier(model_encoder_decoder=model_autoencoder, data_loader=test_loader, device=device)
test_dataset = torch.utils.data.TensorDataset(X_test_denoised, y_test)
denoised_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Evaluate classifier
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=denoised_test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name='classifier_test'
)

plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_test')