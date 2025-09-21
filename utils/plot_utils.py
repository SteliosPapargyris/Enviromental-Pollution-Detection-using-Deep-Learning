import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils.config import *

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
        output_dir = f'out/{norm_name}'
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
    output_dir = f'out/{norm_name}'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/train_and_val_loss_{model_name}.png')
    plt.show()

def plot_normalized_train_mean_feature_per_class(df, class_column='match_Class', save_path='normalized_train_mean_feature_per_class.png', title='Normalized Train Mean Feature per Class'):
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
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
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

def plot_denoised_mean_feature_per_class_before_classifier(X_tensor, y_tensor, save_path='out/mean_feature_per_class.png', title='Mean Denoised Peaks per Class before Classifier'):
    """
    Plots the mean feature vector for each class.

    Args:
        X_tensor (torch.Tensor): Feature tensor of shape (N, D) or (N, 1, D).
        y_tensor (torch.Tensor): Labels tensor of shape (N,).
        save_path (str): Path to save the plot.
        title (str): Title of the plot.
    """

    # Only apply directory restructuring if path doesn't already have a normalization directory
    if not any(f'out/{norm}/' in save_path for norm in ['class_based_mean_std_normalized', 'minmax_normalized', 'normalized', 'raw']):
        # Extract normalization type from save_path and create directory structure
        if 'denoised' in save_path or any(norm in save_path for norm in ['normalized', 'minmax', 'raw']):
            path_parts = save_path.split('/')
            filename = path_parts[-1]
            
            # Create new save path with normalization-specific directory
            save_path = f'out/{norm_name}/{filename}'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Squeeze if necessary
    if X_tensor.dim() == 3 and X_tensor.shape[1] == 1:
        X_tensor = X_tensor.squeeze(1)  # Shape becomes (N, D)

    num_classes = y_tensor.max().item() + 1
    mean_per_class = []

    for class_idx in range(num_classes):
        class_samples = X_tensor[y_tensor == class_idx]
        class_mean = class_samples.mean(dim=0)
        mean_per_class.append(class_mean)

    # Plot
    plt.figure(figsize=(12, 6))
    for i, mean_vector in enumerate(mean_per_class):
        plt.plot(mean_vector.numpy(), label=f"Class {i+1}")

    plt.title(title)
    plt.xlabel('Peak Index (1–32)')
    plt.ylabel('Denoised Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    # Automatically create zoomed version with peaks only (excluding temperature)
    if X_tensor.shape[1] >= 33:  # Has temperature column
        peaks_only_path = save_path.replace('.png', '_peaks_only.png')
        _create_peaks_only_plot(X_tensor, y_tensor, peaks_only_path, f"{title} (Peaks 1-32 only)")

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

def plot_normalized_test_mean_feature_per_class(X_df, y_series, save_path='out/normalized_test_mean_feature_per_class.png', title='Normalized Test Mean Feature per Class'):
    """
    Plots mean normalized test features per class.

    Args:
        X_df (pd.DataFrame): DataFrame with normalized test features (only Peak 1–32).
        y_series (pd.Series): Series with class labels.
        save_path (str): Path to save the figure.
        title (str): Plot title.
    """
    
    # Keep only peak columns
    peak_cols = [col for col in X_df.columns if col.startswith('Peak') and col != 'Temperature']
    X_df = X_df[peak_cols]

    # Combine X and y for grouping
    X_df['Class'] = y_series.values
    mean_per_class = X_df.groupby('Class')[peak_cols].mean()

    # Plot
    plt.figure(figsize=(12, 6))
    for class_label, row in mean_per_class.iterrows():
        plt.plot(row.values, label=f'Class {int(class_label) +1}')

    plt.title(title)
    plt.xlabel('Peak Index (1–32)')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def plot_denoised_test_mean_feature_per_class(X_tensor, y_tensor, save_path='out/denoised_test_mean_feature_per_class.png', title='Denoised Test Mean Feature per Class'):
    """
    Plots mean denoised test features per class from a torch.Tensor.

    Args:
        X_tensor (torch.Tensor): Denoised input tensor of shape (N, 1, 32) or (N, 32).
        y_tensor (torch.Tensor): Class labels tensor of shape (N,).
        save_path (str): Path to save the figure.
        title (str): Plot title.
    """

    if X_tensor.dim() == 3 and X_tensor.shape[1] == 1:
        X_tensor = X_tensor.squeeze(1)  # Shape becomes (N, 32)

    num_classes = int(y_tensor.max().item()) + 1
    mean_per_class = []

    for class_idx in range(num_classes):
        class_samples = X_tensor[y_tensor == class_idx]
        class_mean = class_samples.mean(dim=0)
        mean_per_class.append(class_mean)

    # Plot
    plt.figure(figsize=(12, 6))
    for i, mean_vector in enumerate(mean_per_class):
        plt.plot(mean_vector.numpy(), label=f"Class {i + 1}")

    plt.title(title)
    plt.xlabel("Peak Index (1–32)")
    plt.ylabel("Denoised Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    # Automatically create zoomed version with peaks only (excluding temperature)
    if X_tensor.shape[1] >= 33:  # Has temperature column
        peaks_only_path = save_path.replace('.png', '_peaks_only.png')
        _create_peaks_only_plot(X_tensor, y_tensor, peaks_only_path, f"{title} (Peaks 1-32 only)")

def _create_peaks_only_plot(X_tensor, y_tensor, save_path, title):
    """Helper function to create zoomed plots with peaks only (excluding temperature)"""
    import matplotlib.pyplot as plt
    import os
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Ensure tensor is 2D
    if X_tensor.dim() == 3 and X_tensor.shape[1] == 1:
        X_tensor = X_tensor.squeeze(1)
    
    # Only use peaks 1-32 (exclude temperature at index 32)
    X_peaks = X_tensor[:, :32]  # First 32 features are peaks
    
    num_classes = int(y_tensor.max().item()) + 1
    colors = ["blue", "orange", "green", "red"]
    
    plt.figure(figsize=(12, 6))
    
    for class_idx in range(num_classes):
        class_mask = (y_tensor == class_idx)
        if class_mask.sum() > 0:
            class_data = X_peaks[class_mask]
            mean_features = class_data.mean(dim=0).cpu().numpy()
            
            plt.plot(range(1, 33), mean_features, 
                    color=colors[class_idx % len(colors)], 
                    label=f"Class {class_idx + 1}", 
                    linewidth=2, marker="o", markersize=3)
    
    plt.title(title)
    plt.xlabel("Peak Index (1-32)")
    plt.ylabel("Denoised Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Zoomed peaks plot saved: {save_path}")
    plt.close()  # Close to avoid display

def apply_tsne_and_plot(data, labels, title, save_path, perplexity=30, n_iter=1000):
    """Apply t-SNE to data and create visualization plot with class labels"""
    print(f"Applying t-SNE to {title}...")
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    print(f"Unique classes: {np.unique(labels)}")

    # Ensure data and labels have matching first dimension
    if data.shape[0] != labels.shape[0]:
        print(f"Warning: Data samples ({data.shape[0]}) != Labels ({labels.shape[0]})")
        min_samples = min(data.shape[0], labels.shape[0])
        data = data[:min_samples]
        labels = labels[:min_samples]
        print(f"Truncated to {min_samples} samples")

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_result = tsne.fit_transform(data)
    print(f"t-SNE result shape: {tsne_result.shape}")

    # Create plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    # Define colors for classes
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red

    for i, class_label in enumerate(unique_labels):
        mask = labels == class_label
        points = tsne_result[mask]
        color = colors[i % len(colors)]
        plt.scatter(points[:, 0], points[:, 1], c=color, label=f'Class {class_label}', alpha=0.7, s=50)

    plt.title(f't-SNE Visualization: {title}')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot: {save_path}")

def apply_tsne_and_plot_chips(data, labels, title, save_path, perplexity=30, n_iter=1000):
    """Apply t-SNE to data and create visualization plot with chip labels"""
    print(f"Applying t-SNE to {title} (Chips)...")
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    print(f"Unique chips: {np.unique(labels)}")

    # Ensure data and labels have matching first dimension
    if data.shape[0] != labels.shape[0]:
        print(f"Warning: Data samples ({data.shape[0]}) != Labels ({labels.shape[0]})")
        min_samples = min(data.shape[0], labels.shape[0])
        data = data[:min_samples]
        labels = labels[:min_samples]
        print(f"Truncated to {min_samples} samples")

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_result = tsne.fit_transform(data)
    print(f"t-SNE result shape: {tsne_result.shape}")

    # Create plot
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)

    # Define colors for chips - using different colors than classes
    colors = ['#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#ff9896']  # Pink, Gray, Olive, Cyan, Light Red

    for i, chip_label in enumerate(unique_labels):
        mask = labels == chip_label
        points = tsne_result[mask]
        color = colors[i % len(colors)]
        plt.scatter(points[:, 0], points[:, 1], c=color, label=f'Chip {int(chip_label)}', alpha=0.7, s=50)

    plt.title(f't-SNE Visualization: {title} (by Chip)')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved t-SNE plot: {save_path}")

def get_test_class_labels(X_test, df_full):
    """Extract the actual class labels for test data by matching with original dataframe"""
    # Get the first few features from X_test to match with the original dataframe
    test_features = X_test.iloc[:, :5].values  # Use first 5 features for matching

    # Get the corresponding features from the full dataframe
    df_features = df_full.iloc[:, 1:6].values  # train_Peak 1-5

    # Find matching rows
    test_class_labels = []

    for test_row in test_features:
        # Find the closest match in the full dataframe
        distances = np.sum((df_features - test_row) ** 2, axis=1)
        closest_idx = np.argmin(distances)

        # Get the train_Class for this row
        class_label = df_full.iloc[closest_idx]['train_Class']
        test_class_labels.append(class_label)

    return np.array(test_class_labels)

def get_test_chip_labels(X_test, df_full):
    """Extract the actual chip labels for test data by matching with original dataframe"""
    # Get the first few features from X_test to match with the original dataframe
    test_features = X_test.iloc[:, :5].values  # Use first 5 features for matching

    # Get the corresponding features from the full dataframe
    df_features = df_full.iloc[:, 1:6].values  # train_Peak 1-5

    # Find matching rows
    test_chip_labels = []

    for test_row in test_features:
        # Find the closest match in the full dataframe
        distances = np.sum((df_features - test_row) ** 2, axis=1)
        closest_idx = np.argmin(distances)

        # Get the train_Chip for this row
        chip_label = df_full.iloc[closest_idx]['train_Chip']
        test_chip_labels.append(chip_label)

    return np.array(test_chip_labels)

def extract_representations_with_class_labels(model, data_loader, device, X_test, df_full):
    """Extract decoder outputs, latent representations, and actual class labels from the model"""
    model.eval()
    decoder_outputs = []
    latent_representations = []

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)

            # Get model outputs (decoder output and latent representation)
            decoder_output, latent_repr = model(batch_data)

            # Store results
            decoder_outputs.append(decoder_output.cpu().numpy())
            latent_representations.append(latent_repr.cpu().numpy())

    # Concatenate all batches
    decoder_outputs = np.concatenate(decoder_outputs, axis=0)
    latent_representations = np.concatenate(latent_representations, axis=0)

    # Reshape decoder outputs if needed (remove extra dimensions)
    if decoder_outputs.ndim > 2:
        decoder_outputs = decoder_outputs.reshape(decoder_outputs.shape[0], -1)

    # Get the actual class labels from the original data
    actual_class_labels = get_test_class_labels(X_test, df_full)

    print(f"Final shapes - Decoder: {decoder_outputs.shape}, Latent: {latent_representations.shape}, Class Labels: {actual_class_labels.shape}")
    print(f"Unique classes found: {np.unique(actual_class_labels)}")
    print(f"Class distribution: {np.bincount(actual_class_labels.astype(int))}")

    return decoder_outputs, latent_representations, actual_class_labels

def extract_representations_with_chip_labels(model, data_loader, device, X_test, df_full):
    """Extract decoder outputs, latent representations, and actual chip labels from the model"""
    model.eval()
    decoder_outputs = []
    latent_representations = []

    with torch.no_grad():
        for batch_data, batch_labels in data_loader:
            batch_data = batch_data.to(device)

            # Get model outputs (decoder output and latent representation)
            decoder_output, latent_repr = model(batch_data)

            # Store results
            decoder_outputs.append(decoder_output.cpu().numpy())
            latent_representations.append(latent_repr.cpu().numpy())

    # Concatenate all batches
    decoder_outputs = np.concatenate(decoder_outputs, axis=0)
    latent_representations = np.concatenate(latent_representations, axis=0)

    # Reshape decoder outputs if needed (remove extra dimensions)
    if decoder_outputs.ndim > 2:
        decoder_outputs = decoder_outputs.reshape(decoder_outputs.shape[0], -1)

    # Get the actual chip labels from the original data
    actual_chip_labels = get_test_chip_labels(X_test, df_full)

    print(f"Final shapes - Decoder: {decoder_outputs.shape}, Latent: {latent_representations.shape}, Chip Labels: {actual_chip_labels.shape}")
    print(f"Unique chips found: {np.unique(actual_chip_labels)}")
    print(f"Chip distribution: {np.bincount(actual_chip_labels.astype(int))}")

    return decoder_outputs, latent_representations, actual_chip_labels
