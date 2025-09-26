import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
from utils.config import *

def get_plot_save_path(filename):
    """Get the appropriate save path for plots based on current configuration"""
    os.makedirs(output_base_dir, exist_ok=True)
    return f'{output_base_dir}/{filename}'

def plot_raw_mean_feature_per_class(df, class_column='Class', save_path='raw_mean_feature_per_class.png', title='Raw Mean Feature per Class', log_y=False):
    """
    Plots the mean raw features per class from a DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe containing features and a class column.
        class_column (str): Name of the column containing class labels.
        save_path (str): Path to save the plot.
        title (str): Plot title.
        log_y (bool): Whether to use a logarithmic scale on the y-axis.
    """
    peak_cols = [col for col in df.columns if col.startswith('Peak')]
    mean_per_class = df.groupby(class_column)[peak_cols].mean()

    x = np.arange(1, len(peak_cols) + 1)

    plt.figure(figsize=(12, 6))
    for class_label, row in mean_per_class.iterrows():
        plt.plot(x, row.values, label=f'Class {int(class_label)}')

    plt.title(title)
    plt.xlabel('Peak Index (1–32)')
    plt.ylabel('Raw Value')
    if log_y:
        plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()


def plot_minmax_normalized_mean_feature_per_class(df, class_column='Class', save_path='minmax_normalized_mean_feature_per_class.png', title='Min-Max Normalized Mean Feature per Class'):
    """
    Plots the mean min-max normalized features per class from a DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe containing normalized features and a class column.
        class_column (str): Name of the column containing class labels.
        save_path (str): Path to save the plot.
        title (str): Plot title.
    """
    peak_cols = [col for col in df.columns if col.startswith('Peak')]
    mean_per_class = df.groupby(class_column)[peak_cols].mean()

    x = np.arange(1, len(peak_cols) + 1)

    plt.figure(figsize=(12, 6))
    for class_label, row in mean_per_class.iterrows():
        plt.plot(x, row.values, label=f'Class {int(class_label)}', marker='o', markersize=3)

    plt.title(title)
    plt.xlabel('Peak Index (1–32)')
    plt.ylabel('Min-Max Normalized Value [0, 1]')
    plt.ylim(-0.1, 1.1)  # Set y-axis limits appropriate for min-max scaling
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Min-max normalized plot saved to: {save_path}")


def plot_robust_normalized_mean_feature_per_class(df, class_column='Class', save_path='robust_normalized_mean_feature_per_class.png', title='Robust Scaled Mean Feature per Class'):
    """
    Plots the mean robust scaled features per class from a DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe containing robust normalized features and a class column.
        class_column (str): Name of the column containing class labels.
        save_path (str): Path to save the plot.
        title (str): Plot title.
    """
    peak_cols = [col for col in df.columns if col.startswith('Peak')]
    mean_per_class = df.groupby(class_column)[peak_cols].mean()

    x = np.arange(1, len(peak_cols) + 1)

    plt.figure(figsize=(12, 6))
    for class_label, row in mean_per_class.iterrows():
        plt.plot(x, row.values, label=f'Class {int(class_label)}', marker='o', markersize=3)

    plt.title(title)
    plt.xlabel('Peak Index (1–32)')
    plt.ylabel('Robust Scaled Value (median-centered, MAD-scaled)')
    plt.legend()
    plt.grid(True, alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Robust scaled plot saved to: {save_path}")


def plot_conf_matrix(conf_matrix, label_encoder, model_name, output_dir=None):
    plt.figure()
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cbar=False,
        cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(label_encoder.classes_, rotation=45, ha='right')
    ax.set_yticklabels(label_encoder.classes_, rotation=0)
    plt.title(f'Confusion Matrix_{model_name}')
    plt.tight_layout()

    # Use provided output directory or default to current normalization configuration
    if output_dir is None:
        output_dir = output_base_dir
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/confusion_matrix_{model_name}.jpg")
    plt.show()


def plot_train_and_val_losses(training_losses, validation_losses, model_name):
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss Per Epoch_{model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Use current normalization configuration for output directory
    output_dir = output_base_dir
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/train_and_val_loss_{model_name}.png')
    plt.show()

def plot_normalized_train_mean_feature_per_class(df, class_column='train_Class', save_path=None, title='Normalized Train Mean Feature per Class'):
    """
    Plots the mean normalized train features per class from a DataFrame.

    Args:
        df (pd.DataFrame): Input dataframe with train_Peak columns and class labels.
        class_column (str): Name of the column containing class labels.
        save_path (str): Path to save the plot.
        title (str): Plot title.
    """
    # Select only train_Peak columns
    peak_cols = [col for col in df.columns if col.startswith('train_Peak')]

    # Compute mean for each class
    mean_per_class = df.groupby(class_column)[peak_cols].mean()
    
    # Use default path if none provided
    if save_path is None:
        save_path = get_plot_save_path('normalized_train_mean_feature_per_class.png')
    else:
        # Only create directory if save_path has a directory component
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    # Plot
    plt.figure(figsize=(12, 6))
    for class_label, row in mean_per_class.iterrows():
        plt.plot(row.values, label=f'Class {int(class_label)}')

    plt.title(title)
    plt.xlabel('Peak Index (1–32)')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_raw_test_mean_feature_per_class(df, class_column='Class', save_path='out/raw/raw_test_mean_feature_per_class.png', title='Raw Test Mean Feature per Class'):
    """
    Plots mean raw features per class for the test set.

    Args:
        df (pd.DataFrame): DataFrame with raw peak features and class labels.
        class_column (str): Column name for class labels.
        save_path (str): Path to save the figure.
        title (str): Title of the plot.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Identify peak columns
    peak_cols = [col for col in df.columns if col.startswith('Peak')]

    # Compute mean for each class
    mean_per_class = df.groupby(class_column)[peak_cols].mean()

    # Plot
    plt.figure(figsize=(12, 6))
    for class_label, row in mean_per_class.iterrows():
        plt.plot(row.values, label=f'Class {int(class_label)}')

    plt.title(title)
    plt.xlabel('Peak Index (1–32)')
    plt.ylabel('Raw Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_transferred_data_combined(all_transferred_data, all_class_labels, data_type):
    """
    Plot combined transferred data across all chips - shows mean per class

    Args:
        all_transferred_data (list): List of transferred data tensors from all chips
        all_class_labels (list): List of class label tensors from all chips
        data_type (str): 'train', 'val', or 'test'
    """
    # Combine all data
    combined_data = torch.cat(all_transferred_data, dim=0)
    combined_labels = torch.cat(all_class_labels, dim=0)

    # Convert to numpy and squeeze if needed
    if combined_data.dim() == 3:
        combined_data = combined_data.squeeze(1)

    data_np = combined_data.cpu().numpy()
    labels_np = combined_labels.cpu().numpy()

    # Ensure labels are 1D
    if labels_np.ndim > 1:
        labels_np = labels_np.squeeze()

    # Get unique classes
    unique_classes = np.unique(labels_np)

    plt.figure(figsize=(12, 6))

    # Plot mean for each class across ALL chips
    for class_label in unique_classes:
        class_mask = (labels_np == class_label)
        class_data = data_np[class_mask]
        mean_data = np.mean(class_data, axis=0)

        plt.plot(mean_data, label=f'Class {int(class_label)}', linewidth=1.5, marker='o', markersize=2)

    plt.title(f'Transferred {data_type.title()} Data - Mean Across All Chips')
    plt.xlabel('Feature Index')
    plt.ylabel('Transferred Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_path = get_plot_save_path(f'transferred_{data_type}_all_chips_combined.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined plot: {save_path}")

    # Create zoomed version focusing on lower values (better for distinguishing classes 1-3)
    plt.figure(figsize=(12, 6))

    for class_label in unique_classes:
        class_mask = (labels_np == class_label)
        class_data = data_np[class_mask]
        mean_data = np.mean(class_data, axis=0)

        plt.plot(mean_data, label=f'Class {int(class_label)}', linewidth=1.5, marker='o', markersize=2)

    plt.title(f'Transferred {data_type.title()} Data - Mean Across All Chips (Zoomed)')
    plt.xlabel('Feature Index')
    plt.ylabel('Transferred Value')
    plt.ylim(-0.3, 0.6)  # Focus on lower range to see classes 1-3 better
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save zoomed plot
    save_path_zoomed = get_plot_save_path(f'transferred_{data_type}_all_chips_combined_zoomed.png')
    plt.savefig(save_path_zoomed)
    plt.close()
    print(f"Saved zoomed plot: {save_path_zoomed}")

def plot_denoised_data_combined(all_denoised_data, all_class_labels, data_type):
    """
    Plot combined denoised data from first autoencoder across all chips - shows mean per class

    Args:
        all_denoised_data (list): List of denoised data tensors from all chips
        all_class_labels (list): List of class label tensors from all chips
        data_type (str): 'train', 'val', or 'test'
    """
    # Combine all data
    combined_data = torch.cat(all_denoised_data, dim=0)
    combined_labels = torch.cat(all_class_labels, dim=0)

    # Convert to numpy and squeeze if needed
    if combined_data.dim() == 3:
        combined_data = combined_data.squeeze(1)

    data_np = combined_data.cpu().numpy()
    labels_np = combined_labels.cpu().numpy()

    # Ensure labels are 1D
    if labels_np.ndim > 1:
        labels_np = labels_np.squeeze()

    # Get unique classes
    unique_classes = np.unique(labels_np)

    plt.figure(figsize=(12, 6))

    # Plot mean for each class across ALL chips
    for class_label in unique_classes:
        class_mask = (labels_np == class_label)
        class_data = data_np[class_mask]
        mean_data = np.mean(class_data, axis=0)

        plt.plot(mean_data, label=f'Class {int(class_label)}', linewidth=1.5, marker='o', markersize=2)

    plt.title(f'Denoised {data_type.title()} Data - Mean Across All Chips (First Autoencoder)')
    plt.xlabel('Feature Index')
    plt.ylabel('Denoised Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    save_path = get_plot_save_path(f'denoised_{data_type}_all_chips_combined.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved denoised combined plot: {save_path}")

    # Create zoomed version focusing on lower values
    plt.figure(figsize=(12, 6))

    for class_label in unique_classes:
        class_mask = (labels_np == class_label)
        class_data = data_np[class_mask]
        mean_data = np.mean(class_data, axis=0)

        plt.plot(mean_data, label=f'Class {int(class_label)}', linewidth=1.5, marker='o', markersize=2)

    plt.title(f'Denoised {data_type.title()} Data - Mean Across All Chips (Zoomed)')
    plt.xlabel('Feature Index')
    plt.ylabel('Denoised Value')
    plt.ylim(-0.2, 0.4)  # Adjust range based on typical denoised values
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save zoomed plot
    save_path_zoomed = get_plot_save_path(f'denoised_{data_type}_all_chips_combined_zoomed.png')
    plt.savefig(save_path_zoomed)
    plt.close()
    print(f"Saved denoised zoomed plot: {save_path_zoomed}")

def plot_normalized_data_distribution(normalized_datasets, chip_ids, norm_method):
    """
    Plot normalized data distribution across all chips before autoencoder training

    Args:
        normalized_datasets (list): List of normalized DataFrames
        chip_ids (list): List of chip IDs
        norm_method (str): Normalization method name
    """
    import pandas as pd

    # Combine all normalized datasets
    all_data = []
    for i, df in enumerate(normalized_datasets):
        df_copy = df.copy()
        df_copy['Chip'] = chip_ids[i]
        all_data.append(df_copy)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Get peak columns (assuming they start with 'train_Peak' or 'Peak')
    peak_cols = [col for col in combined_df.columns if 'Peak' in col and not col.endswith('_Class')]

    # Get both train_Peak and match_Peak columns (32 peaks total)
    train_peak_cols = [col for col in peak_cols if col.startswith('train_Peak')]
    match_peak_cols = [col for col in peak_cols if col.startswith('match_Peak')]

    # Sort both types numerically
    train_peak_cols = sorted(train_peak_cols, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
    match_peak_cols = sorted(match_peak_cols, key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

    # Combine train and match peaks (should be 32 total)
    peak_cols = train_peak_cols

    if not peak_cols:
        # Fallback to regular Peak columns if no train/match peaks found
        peak_cols = sorted([col for col in peak_cols if 'Peak' in col],
                          key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)
        peak_cols = peak_cols[:32]

    if not peak_cols:
        print("No peak columns found for plotting")
        return

    if len(peak_cols) < 32:
        print(f"Warning: Only found {len(peak_cols)} peak columns, expected 32")

    # Check if we have class information
    class_col = None
    for col in ['train_Class', 'Class']:
        if col in combined_df.columns:
            class_col = col
            break

    if class_col:
        # Plot mean per class across ALL chips (this is what you want - mean of all chips for 32 peaks)
        mean_per_class = combined_df.groupby(class_col)[peak_cols].mean()
        unique_classes = sorted(combined_df[class_col].unique())

        plt.figure(figsize=(12, 6))

        # Main plot - Mean across ALL chips for each class
        for class_label in unique_classes:
            class_data = mean_per_class.loc[class_label]
            plt.plot(range(1, len(class_data) + 1), class_data.values,
                    label=f'Class {int(class_label)}', linewidth=2, marker='o', markersize=3)

        plt.title(f'Mean Across All Chips - Train Peaks per Class ({norm_method})')
        plt.xlabel(f'Peak Index (1-{len(peak_cols)}) - Train Peaks')
        plt.ylabel('Normalized Value (Mean Across All Chips)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(1, len(peak_cols))

    else:
        # Plot overall distribution without class information
        plt.figure(figsize=(12, 6))

        # Calculate mean across all samples for each peak
        overall_mean = combined_df[peak_cols].mean()
        overall_std = combined_df[peak_cols].std()

        plt.errorbar(range(1, len(overall_mean) + 1), overall_mean.values,
                    yerr=overall_std.values, capsize=3, marker='o', linewidth=2)

        plt.title(f'Normalized Data Distribution - Overall Mean ± Std ({norm_method})')
        plt.xlabel('Peak Index')
        plt.ylabel('Normalized Value')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = get_plot_save_path(f'normalized_data_distribution_{norm_method}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved normalized data distribution plot: {save_path}")

def plot_raw_data_distribution(raw_datasets, chip_ids, output_dir="out/raw"):
    """
    Plot raw data distribution across all chips before normalization

    Args:
        raw_datasets (list): List of raw DataFrames
        chip_ids (list): List of chip IDs
        output_dir (str): Output directory for saving the plot
    """
    import pandas as pd
    import os

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Combine all raw datasets
    all_data = []
    for i, df in enumerate(raw_datasets):
        df_copy = df.copy()
        df_copy['ChipID'] = chip_ids[i]
        all_data.append(df_copy)

    combined_df = pd.concat(all_data, ignore_index=True)

    # Get peak columns (raw data uses "Peak 1", "Peak 2", etc.)
    peak_cols = [col for col in combined_df.columns if col.startswith('Peak ')]

    # Sort peak columns numerically
    peak_cols = sorted(peak_cols, key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0)

    if not peak_cols:
        print("No peak columns found for plotting raw data")
        return

    if len(peak_cols) != 32:
        print(f"Warning: Found {len(peak_cols)} peak columns, expected 32")

    # Check if we have class information
    class_col = 'Class' if 'Class' in combined_df.columns else None

    if class_col:
        # Plot mean per class across ALL chips
        mean_per_class = combined_df.groupby(class_col)[peak_cols].mean()
        unique_classes = sorted(combined_df[class_col].unique())

        plt.figure(figsize=(12, 6))

        # Main plot - Mean across ALL chips for each class
        for class_label in unique_classes:
            class_data = mean_per_class.loc[class_label]
            plt.plot(range(1, len(class_data) + 1), class_data.values,
                    label=f'Class {int(class_label)}', linewidth=2, marker='o', markersize=3)

        plt.title(f'Raw Data Distribution - Mean Across All Chips ({len(chip_ids)} chips)')
        plt.xlabel(f'Peak Index (1-{len(peak_cols)})')
        plt.ylabel('Raw Value (Mean Across All Chips)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xlim(1, len(peak_cols))

    else:
        # Plot overall distribution without class information
        plt.figure(figsize=(12, 6))

        # Calculate mean across all samples for each peak
        overall_mean = combined_df[peak_cols].mean()
        overall_std = combined_df[peak_cols].std()

        plt.errorbar(range(1, len(overall_mean) + 1), overall_mean.values,
                    yerr=overall_std.values, capsize=3, marker='o', linewidth=2)

        plt.title(f'Raw Data Distribution - Overall Mean ± Std ({len(chip_ids)} chips)')
        plt.xlabel('Peak Index')
        plt.ylabel('Raw Value')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    save_path = f"{output_dir}/raw_data_distribution_{len(chip_ids)}chips.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved raw data distribution plot: {save_path}")