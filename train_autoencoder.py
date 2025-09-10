import torch.nn as nn
import torch.optim as optim
from utils.data_utils import dataset_creation, load_and_preprocess_data_autoencoder, tensor_dataset_autoencoder
from utils.train_test_utils import train_encoder_decoder, evaluate_encoder_decoder
from utils.plot_utils import plot_train_and_val_losses, plot_normalized_train_mean_feature_per_class
from utils.models import LinearDenoiser, ConvDenoiser
from utils.config import *

print("=== Autoencoder Training Pipeline Started ===")

# Get normalization info for file naming
norm_config = NORMALIZATION_CONFIG[CURRENT_NORMALIZATION]
norm_name = norm_config['name']
norm_description = norm_config['description']

print(f"Training with normalization: {norm_description}")
print(f"File naming suffix: {norm_name}")

df = dataset_creation(num_chips, baseline_chip=baseline_chip)

# Call the plot function with dynamic naming
plot_normalized_train_mean_feature_per_class(
    df,
    class_column='match_Class',
    save_path=f"out/{norm_name}/{norm_name}_train_mean_feature_per_class.png",
    title=f'Train Mean Peaks per Class after {norm_description}'
)

# Load the shuffled dataset for the current chip
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess_data_autoencoder(file_path=f"{current_path}/shuffled_dataset/merged.csv")

# Create data loaders for raw data
train_loader, val_loader, test_loader = tensor_dataset_autoencoder(batch_size=batch_size, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

model = LinearDenoiser(input_size=33).to(device)

# Define loss function, optimizer, and scheduler
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=autoencoder_patience, verbose=True)

# Train the model on the current chip with dynamic naming
autoencoder_model_name = f'autoencoder_{norm_name}_train'
model_denoiser, training_losses, validation_losses = train_encoder_decoder(
    epochs=num_epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    model_encoder_decoder=model,
    device=device,
    model_encoder_decoder_name=autoencoder_model_name,
    early_stopping_patience=autoencoder_early_stopping)

# Plot training and validation losses with dynamic naming
plot_train_and_val_losses(training_losses, validation_losses, autoencoder_model_name)

# Evaluate the model on the test set
avg_test_loss = evaluate_encoder_decoder(
    model_encoder_decoder=model_denoiser,
    test_loader=test_loader,
    criterion=criterion,
    device=device)

# Print final results with normalization info
print(f"\n=== Autoencoder Training Results ===")
print(f"Normalization Method: {norm_description}")
print(f"Average Test Loss: {avg_test_loss:.6f}")
print(f"\n📁 Generated Files in out/{norm_name}/:")
print(f"   • {norm_name}_train_mean_feature_per_class.png")
print(f"   • train_and_val_loss_{autoencoder_model_name}.png")
print(f"   • pths/{norm_name}/{autoencoder_model_name}.pth")
print("=== Autoencoder Training Completed ===")