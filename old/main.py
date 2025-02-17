import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import load_and_preprocess_data, create_dataloaders
from utils.train_test_utils import train, evaluate_model
from utils.plot_utils import plot_conf_matrix, plot_macro_roc_curve, plot_train_and_val_losses
from utils.models import ConvDenoiser, Classifier
import matplotlib
import numpy as np
import random

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
base_path = "data/mean_and_std_of_class_4_of_every_chip"

# Loop over chip numbers to train sequentially, loading the pretrained model from the previous chip
for chip_number in range(1, 5):
    print(f"Training on Chip {chip_number}...")

    # Load the shuffled dataset for the current chip
    current_path = f'{base_path}/{chip_number}.csv'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess_data(file_path=current_path)

    train_loader, val_loader, test_loader = create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test)
    conv_layers = 'enc_dec'

    # Initialize ConvDenoiser model
    model_denoiser = ConvDenoiser().to(device)
    model_classifier = Classifier().to(device)

    # Load pretrained model from previous chip if it's not the first chip
    if chip_number > 1:
        denoiser_path = f'pths/denoiser_model_{chip_number -1}.pth'
        classifier_path = f'pths/classifier_model_{chip_number -1}.pth'
        print(f"Loading pretrained models from Chip {chip_number - 1}...")
        model_denoiser.load_state_dict(torch.load(denoiser_path))
        model_classifier.load_state_dict(torch.load(classifier_path))


    # Define loss function, optimizer, and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_denoiser.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

    # Train the model on the current chip
    model_denoiser, model_classifier, training_losses, validation_losses = train(
        epochs=num_epochs,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        model_denoiser=model_denoiser,
        model_classifier=model_classifier,
        device=device,
        model_denoiser_name='denoiser_model',
        model_classifier_name='classifier_model',
        chip_number=chip_number
    )

    # Plot training and validation losses
    plot_train_and_val_losses(training_losses, validation_losses, 'classifier_model', chip_number=chip_number)

    # Evaluate the model on the test set
    acc, prec, rec, f1, conf_mat = evaluate_model(
        model_denoiser=model_denoiser,
        model_classifier=model_classifier,
        data_loader=test_loader,
        device=device,
        label_encoder=label_encoder,
        model_name='denoiser_and_classifier',
        conv_layers=conv_layers,
        chip_number=chip_number
    )

    # Plot confusion matrix
    plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_model', chip_number=chip_number)