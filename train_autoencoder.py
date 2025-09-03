import torch.nn as nn
import torch.optim as optim
from utils.data_utils import dataset_creation, load_and_preprocess_data_autoencoder, tensor_dataset_autoencoder
from utils.train_test_utils import train_encoder_decoder, evaluate_encoder_decoder
from utils.plot_utils import plot_train_and_val_losses, plot_normalized_train_mean_feature_per_class
from utils.models import LinearDenoiser, ConvDenoiser
from utils.config import *

df = dataset_creation(num_chips, baseline_chip=baseline_chip)

# Call the plot function
plot_normalized_train_mean_feature_per_class(
    df,
    class_column='match_Class',
    save_path="out/normalized_train_mean_feature_per_class.png",
    title='Train Mean Peaks per Class after Normalization'
)

# Load the shuffled dataset for the current chip
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess_data_autoencoder(file_path=f"{current_path}/shuffled_dataset/merged.csv")

# Create data loaders for raw data
train_loader, val_loader, test_loader = tensor_dataset_autoencoder(batch_size=batch_size, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

model = LinearDenoiser(input_size=33).to(device)
# model = ConvDenoiser(input_length=33).to(device)

# Define loss function, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=autoencoder_patience, verbose=True)

# Train the model on the current chip
model_denoiser, training_losses, validation_losses = train_encoder_decoder(
    epochs=num_epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    model_encoder_decoder=model,
    device=device,
    model_encoder_decoder_name='autoencoder_train',
    early_stopping_patience=autoencoder_early_stopping)

# Plot training and validation losses
plot_train_and_val_losses(training_losses, validation_losses, 'autoencoder_train')

# Evaluate the model on the test set
avg_test_loss = evaluate_encoder_decoder(
    model_encoder_decoder=model_denoiser,
    test_loader=test_loader,
    criterion=criterion,
    device=device)