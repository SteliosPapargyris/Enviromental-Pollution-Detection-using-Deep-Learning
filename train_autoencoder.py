import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from utils.data_utils import dataset_creation, load_and_preprocess_data_autoencoder_prenormalized, tensor_dataset_autoencoder_peaks_only
from utils.train_test_utils import train_encoder_decoder, evaluate_encoder_decoder
from utils.plot_utils import plot_train_and_val_losses, plot_transferred_data_combined, plot_denoised_data_combined, plot_normalized_data_distribution
from utils.models import LinearDenoiser
from utils.config import *

normalized_datasets = dataset_creation(num_chips, baseline_chip=baseline_chip, norm_method=norm_folder)

# Plot normalized data distribution before autoencoder training
print("Plotting normalized data distribution...")
plot_normalized_data_distribution(normalized_datasets, num_chips, norm_name)

# Storage for combined plots across all chips
all_transferred_train = []
all_transferred_val = []
all_transferred_test = []
all_denoised_train = []
all_denoised_val = []
all_denoised_test = []
all_train_labels = []
all_val_labels = []
all_test_labels = []

for chip_idx, df in enumerate(normalized_datasets):
    chip_id = num_chips[chip_idx]

    temp_file_path = f"{current_path}/temp_chip_{chip_id}.csv"
    df.to_csv(temp_file_path, index=False)

    X_train, y_train, X_val, y_val, X_test, y_test, label_encoder, indices_train, indices_val, indices_test = load_and_preprocess_data_autoencoder_prenormalized(
        file_path=temp_file_path, finetune=False
    )

    # Extract class labels from the original dataframe before creating loaders
    original_df = pd.read_csv(temp_file_path)
    class_col = None
    class_col = "train_Class" if "train_Class" in original_df.columns else None

    train_loader, val_loader, test_loader, indices_train, indices_val, indices_test = tensor_dataset_autoencoder_peaks_only(
        batch_size=batch_size,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        X_test=X_test, y_test=y_test,
        indices_train=indices_train, indices_val=indices_val, indices_test=indices_test
    )

    model = LinearDenoiser(input_size=33, output_size=32).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=autoencoder_patience, verbose=True)

    autoencoder_model_name = f'autoencoder_{norm_name}_chip_{chip_id}'
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
        early_stopping_patience=autoencoder_early_stopping
    )

    plot_train_and_val_losses(training_losses, validation_losses, autoencoder_model_name)

    avg_train_loss, denoised_train_data, train_labels = evaluate_encoder_decoder(
        model_encoder_decoder=model_denoiser,
        test_loader=train_loader,
        criterion=criterion,
        device=device
    )


    avg_val_loss, denoised_val_data, val_labels = evaluate_encoder_decoder(
        model_encoder_decoder=model_denoiser,
        test_loader=val_loader,
        criterion=criterion,
        device=device
    )

    avg_test_loss, denoised_test_data, test_labels = evaluate_encoder_decoder(
        model_encoder_decoder=model_denoiser,
        test_loader=test_loader,
        criterion=criterion,
        device=device
    )

    # Save first autoencoder outputs (before transfer)
    autoencoder_output_path = f"{output_base_dir}/autoencoder_output_chip_{chip_id}_{norm_name}.csv"
    all_denoised_data = torch.cat([denoised_train_data, denoised_val_data, denoised_test_data], dim=0)
    all_indices = np.concatenate([indices_train, indices_val, indices_test])

    denoised_df = pd.DataFrame(all_denoised_data.squeeze().cpu().numpy())
    denoised_df['Chip'] = chip_id
    if class_col is not None:
        # Use indices to map back to original class labels correctly
        denoised_df['Class'] = original_df.iloc[all_indices][class_col].values
    denoised_df.to_csv(autoencoder_output_path, index=False)

    X_baseline_train, y_baseline_train, X_baseline_val, y_baseline_val, X_baseline_test, y_baseline_test, baseline_label_encoder, _, _, _ = load_and_preprocess_data_autoencoder_prenormalized(
        file_path=f"data/out/mean_std/{total_num_chips}chips/chip_{baseline_chip}_mean_std.csv", finetune=False
    )
    X_baseline_train.drop(columns=['Chip'], inplace=True)
    X_baseline_val.drop(columns=['Chip'], inplace=True)
    X_baseline_test.drop(columns=['Chip'], inplace=True)

    transfer_train_dataset = TensorDataset(denoised_train_data.squeeze(), torch.tensor(X_baseline_train.values, dtype=torch.float32))
    transfer_val_dataset = TensorDataset(denoised_val_data.squeeze(), torch.tensor(X_baseline_val.values, dtype=torch.float32))
    transfer_test_dataset = TensorDataset(denoised_test_data.squeeze(), torch.tensor(X_baseline_test.values, dtype=torch.float32))

    transfer_train_loader = DataLoader(transfer_train_dataset, batch_size=batch_size, shuffle=True)
    transfer_val_loader = DataLoader(transfer_val_dataset, batch_size=batch_size, shuffle=False)
    transfer_test_loader = DataLoader(transfer_test_dataset, batch_size=batch_size, shuffle=False)

    transfer_model = LinearDenoiser(input_size=32, output_size=32).to(device)
    transfer_criterion = nn.MSELoss()
    transfer_optimizer = optim.Adam(transfer_model.parameters(), lr=learning_rate)
    transfer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(transfer_optimizer, mode='min', patience=autoencoder_patience, verbose=True)

    transfer_model_name = f'transfer_autoencoder_{norm_name}_chip_{chip_id}_to_baseline_{baseline_chip}'

    trained_transfer_model, transfer_training_losses, transfer_validation_losses = train_encoder_decoder(
        epochs=num_epochs,
        train_loader=transfer_train_loader,
        val_loader=transfer_val_loader,
        optimizer=transfer_optimizer,
        criterion=transfer_criterion,
        scheduler=transfer_scheduler,
        model_encoder_decoder=transfer_model,
        device=device,
        model_encoder_decoder_name=transfer_model_name,
        early_stopping_patience=autoencoder_early_stopping
    )

    plot_train_and_val_losses(transfer_training_losses, transfer_validation_losses, transfer_model_name)

    avg_transfer_train_loss, transferred_train_data, transfer_train_labels = evaluate_encoder_decoder(
        model_encoder_decoder=trained_transfer_model,
        test_loader=transfer_train_loader,
        criterion=transfer_criterion,
        device=device
    )

    avg_transfer_val_loss, transferred_val_data, transfer_val_labels = evaluate_encoder_decoder(
        model_encoder_decoder=trained_transfer_model,
        test_loader=transfer_val_loader,
        criterion=transfer_criterion,
        device=device
    )

    avg_transfer_test_loss, transferred_test_data, transfer_test_labels = evaluate_encoder_decoder(
        model_encoder_decoder=trained_transfer_model,
        test_loader=transfer_test_loader,
        criterion=transfer_criterion,
        device=device
    )

    # Plot the transferred/cleaned data with actual class labels
    if class_col is not None:
        # Get actual class labels for train/val/test splits using the indices
        train_class_labels = torch.tensor(original_df.iloc[indices_train][class_col].values, dtype=torch.long)
        val_class_labels = torch.tensor(original_df.iloc[indices_val][class_col].values, dtype=torch.long)
        test_class_labels = torch.tensor(original_df.iloc[indices_test][class_col].values, dtype=torch.long)

        # Store data for combined plots
        all_transferred_train.append(transferred_train_data)
        all_transferred_val.append(transferred_val_data)
        all_transferred_test.append(transferred_test_data)
        all_denoised_train.append(denoised_train_data)
        all_denoised_val.append(denoised_val_data)
        all_denoised_test.append(denoised_test_data)
        all_train_labels.append(train_class_labels)
        all_val_labels.append(val_class_labels)
        all_test_labels.append(test_class_labels)

    transfer_output_path = f"{output_base_dir}/transfer_autoencoder_output_chip_{chip_id}_to_baseline_{baseline_chip}_{norm_name}.csv"

    all_transferred_data = torch.cat([transferred_train_data, transferred_val_data, transferred_test_data], dim=0)

    transfer_df = pd.DataFrame(all_transferred_data.squeeze().cpu().numpy())
    transfer_df['Chip'] = chip_id

    # Only add Class column if actual class labels exist
    if class_col is not None:
        # Use indices to map back to original class labels correctly
        transfer_df['Class'] = original_df.iloc[all_indices][class_col].values

    transfer_df.to_csv(transfer_output_path, index=False)

transfer_output_files = []
for chip_idx, df in enumerate(normalized_datasets):
    chip_id = num_chips[chip_idx]
    transfer_file = f"{output_base_dir}/transfer_autoencoder_output_chip_{chip_id}_to_baseline_{baseline_chip}_{norm_name}.csv"
    if os.path.exists(transfer_file):
        transfer_output_files.append(transfer_file)

if transfer_output_files:
    all_transfer_outputs = []
    for file_path in transfer_output_files:
        df = pd.read_csv(file_path)
        all_transfer_outputs.append(df)

    merged_transfer_outputs = pd.concat(all_transfer_outputs, ignore_index=True)
    merged_output_path = f"{output_base_dir}/merged_transfer_autoencoder_outputs_{norm_name}_to_baseline_{baseline_chip}.csv"
    merged_transfer_outputs.to_csv(merged_output_path, index=False)

# Merge first autoencoder outputs
autoencoder_output_files = []
for chip_idx, df in enumerate(normalized_datasets):
    chip_id = num_chips[chip_idx]
    autoencoder_file = f"{output_base_dir}/autoencoder_output_chip_{chip_id}_{norm_name}.csv"
    if os.path.exists(autoencoder_file):
        autoencoder_output_files.append(autoencoder_file)

if autoencoder_output_files:
    all_autoencoder_outputs = []
    for file_path in autoencoder_output_files:
        df = pd.read_csv(file_path)
        all_autoencoder_outputs.append(df)

    merged_autoencoder_outputs = pd.concat(all_autoencoder_outputs, ignore_index=True)
    merged_autoencoder_path = f"{output_base_dir}/merged_autoencoder_outputs_{norm_name}.csv"
    merged_autoencoder_outputs.to_csv(merged_autoencoder_path, index=False)

# Create combined plots across all chips
if all_transferred_train:
    print(f"Creating combined transferred plots across all {total_num_chips} chips...")
    plot_transferred_data_combined(all_transferred_train, all_train_labels, 'train')
    plot_transferred_data_combined(all_transferred_val, all_val_labels, 'val')
    plot_transferred_data_combined(all_transferred_test, all_test_labels, 'test')

if all_denoised_train:
    print(f"Creating combined denoised plots across all {total_num_chips} chips...")
    plot_denoised_data_combined(all_denoised_train, all_train_labels, 'train')
    plot_denoised_data_combined(all_denoised_val, all_val_labels, 'val')
    plot_denoised_data_combined(all_denoised_test, all_test_labels, 'test')

