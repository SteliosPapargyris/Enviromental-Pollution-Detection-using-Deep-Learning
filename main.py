import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import load_and_preprocess_data, create_dataloaders, load_and_preprocess_data_classifier
from utils.train_test_utils import train_encoder_decoder, evaluate_encoder_decoder, train_classifier, \
    evaluate_classifier
from utils.plot_utils import plot_conf_matrix, plot_macro_roc_curve, plot_train_and_val_losses
from utils.models import ConvDenoiser, Classifier
import matplotlib
import numpy as np
import random
import pandas as pd

# Temp Combination: for training of the classifier: combine in one row 4 samples with the same class and same temperature (32x4 → 128 columns)
# 4 encoder decoders (one for each chip) train as it is

matplotlib.use('Agg')  # Use a non-interactive backend
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Hyperparameters
batch_size = 32
learning_rate = 1e-3
num_epochs = 200
num_classes = 4
base_path = "D:\Stelios\Work\Auth_AI\semester_3\Thesis\January\encoder_decoder\code\data\mean_and_std_of_class_4_of_every_chip"

# Initialize dictionaries to store data for each chip
X_denoised_train_dict = {}
X_denoised_val_dict = {}
X_denoised_test_dict = {}

# Loop over chip numbers to train sequentially, loading the pretrained model from the previous chip
for chip_number in range(1, 5):
    print(f"Training on Chip {chip_number}...")

    # Load the shuffled dataset for the current chip
    current_path = f'{base_path}/{chip_number}.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_val, y_val, X_test, y_test, X_denoised_train, X_denoised_val, X_denoised_test, temp_train, temp_val, temp_test, class_train, class_val, class_test, label_encoder = load_and_preprocess_data(
        file_path=current_path
    )

    # Create data loaders for raw data
    train_loader, val_loader, test_loader, _, _, _ = create_dataloaders(batch_size=batch_size, X_train=X_train,
                                                                        y_train=y_train, X_val=X_val, y_val=y_val,
                                                                        X_test=X_test, y_test=y_test)
    conv_layers = 'enc_dec'

    # Initialize ConvDenoiser model
    model_denoiser = ConvDenoiser().to(device)
    model_classifier = Classifier().to(device)

    # Load pretrained model from previous chip if it's not the first chip
    if chip_number > 1:
        denoiser_path = f'pths/denoiser_model_{chip_number - 1}.pth'
        print(f"Loading pretrained models from Chip {chip_number - 1}...")
        model_denoiser.load_state_dict(torch.load(denoiser_path))

    # Define loss function, optimizer, and scheduler
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_denoiser.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

    # Train the model on the current chip
    model_denoiser, training_losses, validation_losses, noise_factor, X_denoised_train, X_denoised_val, X_denoised_test = train_encoder_decoder(
        epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        model_encoder_decoder=model_denoiser,
        device=device,
        model_encoder_decoder_name='denoiser_model',
        chip_number=chip_number
    )

    # Reshape temperature arrays to match denoised data shape
    temp_train = temp_train.reshape(-1, 1, 1)  # Match (batch_size, channels, 1)
    class_train = class_train.reshape(-1, 1, 1)
    temp_val = temp_val.reshape(-1, 1, 1)
    class_val = class_val.reshape(-1, 1, 1)
    temp_test = temp_test.reshape(-1, 1, 1)
    class_test = class_test.reshape(-1, 1, 1)

    # Plot training and validation losses
    plot_train_and_val_losses(training_losses, validation_losses, 'encoder_decoder_model', chip_number=chip_number)

    # Evaluate the model on the test set
    avg_test_loss, X_denoised_test = evaluate_encoder_decoder(
        model_encoder_decoder=model_denoiser,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        label_encoder=label_encoder,
        model_name='denoiser',
        conv_layers=conv_layers,
        chip_number=chip_number
    )

    # Concatenate along the last axis
    X_denoised_train_dict[f"X_denoised_train_{chip_number}"] = np.concatenate(
        (X_denoised_train, temp_train, class_train), axis=2)
    X_denoised_val_dict[f"X_denoised_val_{chip_number}"] = np.concatenate((X_denoised_val, temp_val, class_val), axis=2)

    X_denoised_test_dict[f"X_denoised_test_{chip_number}"] = np.concatenate((X_denoised_test, temp_test, class_test), axis=2)

# Extract train arrays from dictionary
X_denoised_train_1 = X_denoised_train_dict["X_denoised_train_1"]
X_denoised_train_2 = X_denoised_train_dict["X_denoised_train_2"]
X_denoised_train_3 = X_denoised_train_dict["X_denoised_train_3"]
X_denoised_train_4 = X_denoised_train_dict["X_denoised_train_4"]

# Extract validation arrays from dictionary
X_denoised_val_1 = X_denoised_val_dict["X_denoised_val_1"]
X_denoised_val_2 = X_denoised_val_dict["X_denoised_val_2"]
X_denoised_val_3 = X_denoised_val_dict["X_denoised_val_3"]
X_denoised_val_4 = X_denoised_val_dict["X_denoised_val_4"]

# Extract test arrays from dictionary
X_denoised_test_1 = X_denoised_test_dict["X_denoised_test_1"]
X_denoised_test_2 = X_denoised_test_dict["X_denoised_test_2"]
X_denoised_test_3 = X_denoised_test_dict["X_denoised_test_3"]
X_denoised_test_4 = X_denoised_test_dict["X_denoised_test_4"]

# Function to combine arrays while keeping temperature and class only once
def combine_denoised_data(X1, X2, X3, X4):
    # Stack the first 32 columns from all four arrays along axis=2
    X_combined = np.concatenate((
        X1[:, :, :-2],  # Take first 32 columns
        X2[:, :, :-2],
        X3[:, :, :-2],
        X4[:, :, :-2]
    ), axis=2)  # Stack along the last axis

    # Extract the last 2 columns from one of the arrays (since they are identical)
    temperature_class = X1[:, :, -2:]  # Take only once

    # Concatenate everything to form the final array
    return np.concatenate((X_combined, temperature_class), axis=2)

# Combine all datasets
X_denoised_train_all = combine_denoised_data(X_denoised_train_1, X_denoised_train_2, X_denoised_train_3, X_denoised_train_4)
X_denoised_val_all = combine_denoised_data(X_denoised_val_1, X_denoised_val_2, X_denoised_val_3, X_denoised_val_4)
X_denoised_test_all = combine_denoised_data(X_denoised_test_1, X_denoised_test_2, X_denoised_test_3, X_denoised_test_4)

# Final shape checks
print("Final Shapes:")
print(f"Train: {X_denoised_train_all.shape}")  # Expected: (280, 1, 130)
print(f"Validation: {X_denoised_val_all.shape}")  # Expected: (VAL_SIZE, 1, 130)
print(f"Test: {X_denoised_test_all.shape}")  # Expected: (TEST_SIZE, 1, 130)

y_train = X_denoised_train_all[:, :, -1].squeeze()
y_val = X_denoised_val_all[:, :, -1].squeeze()
y_test = X_denoised_test_all[:, :, -1].squeeze()

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit on training labels and transform all sets
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)  # Use transform only (no fit) for validation set
y_test_encoded = label_encoder.transform(y_test)  # Use transform only (no fit) for test set

# Final check
print("Encoded Labels:")
print(f"y_train_encoded: {y_train_encoded[:10]}")  # Print first 10 for verification
print(f"y_val_encoded: {y_val_encoded[:10]}")
print(f"y_test_encoded: {y_test_encoded[:10]}")


X_denoised_train_all = X_denoised_train_all[:, :, :-2]
X_denoised_val_all = X_denoised_val_all[:, :, :-2]
X_denoised_test_all = X_denoised_test_all[:, :, :-2]

# Create data loaders for raw data
_, _, _, denoised_train_loader, denoised_val_loader, denoised_test_loader = create_dataloaders(
    batch_size=batch_size, X_denoised_train=X_denoised_train_all,
    y_train=y_train_encoded, X_denoised_val=X_denoised_val_all, y_val=y_val_encoded, X_denoised_test=X_denoised_test_all, y_test=y_test_encoded)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_classifier = Classifier().to(device)
# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

# Train the model on the current chip
model_classifier, training_losses, validation_losses = train_classifier(
    epochs=num_epochs,
    train_loader=denoised_train_loader,
    val_loader=denoised_val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    model_classifier=model_classifier,
    device=device,
    model_classifier_name='classifier_model',
)

# Plot training and validation losses
plot_train_and_val_losses(training_losses, validation_losses, 'classifier_model', chip_number=5)

model_denoiser = ConvDenoiser().to(device)
denoiser_path = f'pths/denoiser_model_4.pth'
model_denoiser.load_state_dict(torch.load(denoiser_path))

# Evaluate the model on the test set
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_denoiser=model_denoiser,
    model_classifier=model_classifier,
    data_loader=denoised_test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name='denoiser_and_classifier'
)

# Plot confusion matrix
plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_model')
