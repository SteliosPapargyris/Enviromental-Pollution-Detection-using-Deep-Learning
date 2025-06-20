import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_conf_matrix(conf_matrix, label_encoder, model_name):
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
    plt.savefig(f"out/confusion_matrix_{model_name}.jpg")
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
    plt.savefig(f'out/train_and_val_loss_{model_name}.png')
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

def plot_denoised_mean_feature_per_class_before_classifier(X_tensor, y_tensor, save_path='out/mean_feature_per_class.png', title='Mean Denoised Peaks per Class before Classifier'):
    """
    Plots the mean feature vector for each class.

    Args:
        X_tensor (torch.Tensor): Feature tensor of shape (N, D) or (N, 1, D).
        y_tensor (torch.Tensor): Labels tensor of shape (N,).
        save_path (str): Path to save the plot.
        title (str): Title of the plot.
    """

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

def plot_raw_test_mean_feature_per_class(df, class_column='Class', save_path='out/raw_test_mean_feature_per_class.png', title='Raw Test Mean Feature per Class'):
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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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