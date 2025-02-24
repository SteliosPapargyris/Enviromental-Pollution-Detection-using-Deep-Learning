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

# Dictionary to store denoised training data for each chip
X_denoised_train_dict = {}

# Variable to store the last chip's X_denoised_train (only for matching)
prev_X_denoised_train = None

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
        classifier_path = f'pths/classifier_model_{chip_number - 1}.pth'
        print(f"Loading pretrained models from Chip {chip_number - 1}...")
        model_denoiser.load_state_dict(torch.load(denoiser_path))
        model_classifier.load_state_dict(torch.load(classifier_path))

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

    X_denoised_train = np.concatenate((X_denoised_train, temp_train, class_train), axis=2)
    X_denoised_val = np.concatenate((X_denoised_val, temp_val, class_val), axis=2)

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

    X_denoised_test = np.concatenate((X_denoised_test, temp_test, class_test), axis=2)

    # Store X_denoised_train in dictionary with chip_number as the key
    X_denoised_train_dict[f"X_denoised_train_{chip_number}"] = X_denoised_train.copy()

    # Only run matching logic if this is NOT the first chip
    if chip_number > 1 and prev_X_denoised_train is not None:
        # Extract the last two columns for matching
        keys1 = X_denoised_train[:, :, -2:]  # Last two columns from current chip's data
        keys2 = prev_X_denoised_train[:, :, -2:]  # Last two columns from previous chip

        # Find matching row indices
        matching_indices = np.nonzero(np.all(keys1[:, None] == keys2, axis=-1))[1]

        # Ensure indices are valid
        if len(matching_indices) > 0:
            # Filter prev_X_denoised_train to only keep the matching rows (excluding last two columns)
            filtered_prev_X_denoised_train = prev_X_denoised_train[matching_indices, :, :-2]  # Exclude last two columns

            # Concatenate along axis 2 (column-wise) to merge the arrays
            all_X_train_denoised = np.concatenate((X_denoised_train, filtered_prev_X_denoised_train), axis=2)

            print(f"Chip {chip_number}: Shape of new merged array:", all_X_train_denoised.shape)
        else:
            print(f"Chip {chip_number}: No matching rows found. Check your matching logic.")

    # Update prev_X_denoised_train with the current chip's data (for next iteration)
    prev_X_denoised_train = X_denoised_train.copy()

    # file_paths = [
    # "D:\Stelios\Work\Auth_AI\semester_3\Thesis\January\encoder_decoder\code\data\mean_and_std_of_class_4_of_every_chip/1.csv",
    # "D:\Stelios\Work\Auth_AI\semester_3\Thesis\January\encoder_decoder\code\data\mean_and_std_of_class_4_of_every_chip/2.csv",
    # "D:\Stelios\Work\Auth_AI\semester_3\Thesis\January\encoder_decoder\code\data\mean_and_std_of_class_4_of_every_chip/3.csv",
    # "D:\Stelios\Work\Auth_AI\semester_3\Thesis\January\encoder_decoder\code\data\mean_and_std_of_class_4_of_every_chip/4.csv"
    # ]
    #
    # # Load and preprocess raw data
    # X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess_data_classifier(
    #     file_paths=file_paths)
    #
    # # Create data loaders for raw data
    # _, _, _, denoised_train_loader, denoised_val_loader, denoised_test_loader = create_dataloaders(
    #     batch_size=batch_size, X_denoised_train=X_denoised_train,
    #     y_train=y_train, X_denoised_val=X_denoised_val, y_val=y_val, X_denoised_test=X_denoised_test, y_test=y_test)
    #
    # # Define loss function, optimizer, and scheduler
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model_classifier.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)
    #
    # # Train the model on the current chip
    # model_classifier, training_losses, validation_losses = train_classifier(
    #     epochs=num_epochs,
    #     train_loader=denoised_train_loader,
    #     val_loader=denoised_val_loader,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     scheduler=scheduler,
    #     model_classifier=model_classifier,
    #     device=device,
    #     model_classifier_name='classifier_model',
    #     chip_number=chip_number,
    # )
    #
    # # Plot training and validation losses
    # plot_train_and_val_losses(training_losses, validation_losses, 'classifier_model', chip_number=chip_number)
    #
    # # Evaluate the model on the test set
    # acc, prec, rec, f1, conf_mat = evaluate_classifier(
    #     model_denoiser=model_denoiser,
    #     model_classifier=model_classifier,
    #     data_loader=test_loader,
    #     device=device,
    #     label_encoder=label_encoder,
    #     model_name='denoiser_and_classifier',
    #     conv_layers=conv_layers,
    #     chip_number=chip_number
    # )
    #
    # # Plot confusion matrix
    # plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_model', chip_number=chip_number)
