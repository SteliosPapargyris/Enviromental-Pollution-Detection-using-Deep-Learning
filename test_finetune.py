import torch
import torch.nn as nn
import torch.optim as optim
from utils.data_utils import load_and_preprocess_data_classifier, tensor_dataset_classifier, load_and_preprocess_test_data, tensor_dataset_autoencoder, load_and_preprocess_data_autoencoder
from utils.train_test_utils import evaluate_encoder_decoder_for_classifier, evaluate_classifier, train_encoder_decoder, train_classifier
from utils.plot_utils import plot_conf_matrix, plot_normalized_test_mean_feature_per_class, plot_denoised_test_mean_feature_per_class
from utils.models import LinearDenoiser, ConvDenoiser, Classifier
from utils.config import *
import os
import pandas as pd

# Test configuration (can be overridden if needed)
CURRENT_STATS_SOURCE = 'compute'  # 'compute' or 'json'
CURRENT_STATS_PATH = stats_path  # Only used if CURRENT_STATS_SOURCE = 'json'
NORMALIZE_TARGET_CLASS = True

print("=== Test Pipeline Started ===")

# Load and preprocess test data
print("Loading and preprocessing test data...")

merged_csv_path = os.path.join(base_path, "shuffled_dataset", "merged.csv")

# Load and normalize test data using the proper function
X_test, y_test, label_encoder = load_and_preprocess_test_data(
    test_file_path,
    fraction=num_percentage_of_test_df,
    random_seed=seed,
    stats_source=CURRENT_STATS_SOURCE,
    stats_path=CURRENT_STATS_PATH,
    apply_normalization=True,
    normalization_type=CURRENT_NORMALIZATION,
    normalize_target_class=NORMALIZE_TARGET_CLASS
)

# Recreate test_df from processed data for merging
test_df = X_test.copy()
test_df['Class'] = label_encoder.inverse_transform(y_test)  # Convert back to original class labels
test_df['Chip'] = chip_exclude  # Add chip column

# Load already normalized baseline chip (should match the normalized test data)
df_baseline = pd.read_csv(os.path.join(base_path, f"{baseline_chip}.csv"))
merged_rows = []

for _, train_row in test_df.iterrows():
    # Use approximate matching for floating point temperatures
    temp_tolerance = 1e-6
    matching_rows = df_baseline[
        (abs(df_baseline['Temperature'] - train_row['Temperature']) < temp_tolerance) &
        (df_baseline['Class'] == train_row['Class'])
    ]
    for _, match_row in matching_rows.iterrows():
        train_series = train_row.add_prefix("train_")
        match_series = match_row.add_prefix("match_")
        merged_row = pd.concat([train_series, match_series])
        merged_rows.append(merged_row)

merged_df = pd.DataFrame(merged_rows)

print(f"Merged dataset shape: {merged_df.shape}")

# Load the trained autoencoder model based on normalization method
autoencoder_path = f"pths/{norm_name}/autoencoder_{norm_name}_train.pth"
print(f"Loading autoencoder from: {autoencoder_path}")

# Save merged_df temporarily and use it for fine-tuning
temp_merged_path = "temp_merged_for_finetuning.csv"
merged_df.to_csv(temp_merged_path, index=False)

# Prepare data for fine-tuning using the merged_df
X_train_autoencoder, y_train_autoencoder, X_val_autoencoder, y_val_autoencoder, X_test_autoencoder, y_test_autoencoder, label_encoder_ae = load_and_preprocess_data_autoencoder(
    temp_merged_path,
    finetune=True)

# Since finetune=True, test sets are empty, so we just use train data
X_train_combined = X_train_autoencoder
y_train_combined = y_train_autoencoder

print(f"\n=== Modified Autoencoder Data Split ===")
total_samples = len(X_train_autoencoder) + len(X_val_autoencoder) + len(X_test_autoencoder)
print(f"Training (with test samples): {len(X_train_combined)} samples ({len(X_train_combined)/total_samples:.1%})")
print(f"Validation: {len(X_val_autoencoder)} samples ({len(X_val_autoencoder)/total_samples:.1%})")

# Use smaller batch size for small dataset
finetune_batch_size = 8  # Reduced from 32

# Create data loaders with smaller batch size
train_loader_ae, val_loader_ae, _ = tensor_dataset_autoencoder(
    finetune_batch_size, X_train_combined, y_train_combined, X_val_autoencoder, y_val_autoencoder
)

# Load pre-trained autoencoder
autoencoder = LinearDenoiser(input_size=33).to(device)
autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))

print("Starting fine-tuning...")

# Set up optimizer, criterion, and scheduler for fine-tuning
optimizer_ae = optim.Adam(autoencoder.parameters(), lr=learning_rate)
criterion_ae = nn.MSELoss()
scheduler_ae = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ae, mode='min', patience=autoencoder_patience, factor=0.5, verbose=True)

# Fine-tune the autoencoder
fine_tuned_autoencoder, train_losses, val_losses = train_encoder_decoder(
    num_epochs, train_loader_ae, val_loader_ae,
    optimizer_ae, criterion_ae, scheduler_ae, autoencoder, device,
    f"autoencoder_{norm_name}_finetuned", autoencoder_early_stopping
)

print("Fine-tuning completed!")
print(f"Fine-tuned autoencoder saved to: pths/{norm_name}/autoencoder_{norm_name}_finetuned.pth")

# === CLASSIFIER FINE-TUNING ===
print("\n=== Starting Classifier Fine-tuning ===")

# Load the original trained classifier
classifier_path = f"pths/{norm_name}/classifier_{norm_name}_train.pth"
print(f"Loading pre-trained classifier from: {classifier_path}")

model_classifier = Classifier(input_length=33, num_classes=4).to(device)
model_classifier.load_state_dict(torch.load(classifier_path, map_location=device))

# Prepare classifier data from merged_df (need class labels, not autoencoder targets)
X_train_class, y_train_class, X_val_class, y_val_class, X_test_class, y_test_class, label_encoder_class = load_and_preprocess_data_classifier(
    temp_merged_path, finetune=True
)

# Create data loaders with class labels (not autoencoder data)
train_loader_class, val_loader_class, _ = tensor_dataset_classifier(
    finetune_batch_size, X_train_class, y_train_class, X_val_class, y_val_class
)

# Use the fine-tuned autoencoder to denoise data for classifier fine-tuning
fine_tuned_autoencoder.eval()

# Get denoised data from fine-tuned autoencoder (this preserves the class labels)
X_train_denoised, y_train_class_final = evaluate_encoder_decoder_for_classifier(
    model_encoder_decoder=fine_tuned_autoencoder,
    data_loader=train_loader_class,
    device=device
)

X_val_denoised, y_val_class_final = evaluate_encoder_decoder_for_classifier(
    model_encoder_decoder=fine_tuned_autoencoder,
    data_loader=val_loader_class,
    device=device
)

# Create DataLoaders for classifier training with denoised features and class labels
train_dataset_class = torch.utils.data.TensorDataset(X_train_denoised, y_train_class_final)
val_dataset_class = torch.utils.data.TensorDataset(X_val_denoised, y_val_class_final)

denoised_train_loader_class = torch.utils.data.DataLoader(train_dataset_class, batch_size=finetune_batch_size, shuffle=True)
denoised_val_loader_class = torch.utils.data.DataLoader(val_dataset_class, batch_size=finetune_batch_size, shuffle=False)

# Set up optimizer, criterion, and scheduler for classifier fine-tuning
criterion_class = nn.CrossEntropyLoss()
optimizer_class = optim.Adam(model_classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler_class = optim.lr_scheduler.ReduceLROnPlateau(optimizer_class, mode='min', patience=classifier_patience, verbose=True)

# Fine-tune the classifier
classifier_model_name = f'classifier_{norm_name}_finetuned'
fine_tuned_classifier, train_losses_class, val_losses_class = train_classifier(
    epochs=num_epochs,
    train_loader=denoised_train_loader_class,
    val_loader=denoised_val_loader_class,
    optimizer=optimizer_class,
    criterion=criterion_class,
    scheduler=scheduler_class,
    model_classifier=model_classifier,
    device=device,
    model_classifier_name=classifier_model_name,
    early_stopping_patience=classifier_early_stopping
)

print("Classifier fine-tuning completed!")
print(f"Fine-tuned classifier saved to: pths/{norm_name}/classifier_{norm_name}_finetuned.pth")

# === INFERENCE ON REMAINING TEST DATA ===
print("\n=== Starting Inference on Remaining Test Data ===")

# Load the complete test dataset first to properly split and exclude fine-tuning samples
print("Loading complete test dataset to properly exclude fine-tuning samples...")
df_full_test = pd.read_csv(test_file_path)

# First shuffle the full dataset with the same seed, preserving original indices
df_full_test_shuffled = df_full_test.sample(frac=1.0, random_state=seed)

# Calculate how many samples to use for fine-tuning
n_finetuning = int(len(df_full_test_shuffled) * num_percentage_of_test_df)

# Split the shuffled data: first n_finetuning samples for fine-tuning, rest for inference
df_finetuning = df_full_test_shuffled.iloc[:n_finetuning].reset_index(drop=True)
df_remaining = df_full_test_shuffled.iloc[n_finetuning:].reset_index(drop=True)

remaining_fraction = 1 - num_percentage_of_test_df
print(f"Fine-tuning samples: {len(df_finetuning)} ({num_percentage_of_test_df:.1%})")
print(f"Remaining samples: {len(df_remaining)} ({len(df_remaining)/len(df_full_test):.1%})")
print(f"Total samples: {len(df_full_test)}")

# Verify no overlap by checking if any samples are identical
# Compare using a subset of features to check for identical rows
feature_cols = [col for col in df_finetuning.columns if col not in ['Class', 'Chip']]
finetuning_subset = df_finetuning[feature_cols[:5]].round(6)  # Use first 5 features rounded for comparison
remaining_subset = df_remaining[feature_cols[:5]].round(6)

# print(f"Sample overlap check: {'✗ Overlap detected' if overlap_found else '✓ No overlap detected'}")

# Save the remaining dataset temporarily and process it through the normalization pipeline
temp_remaining_path = "temp_remaining_for_inference.csv"
df_remaining.to_csv(temp_remaining_path, index=False)

# Load and preprocess the remaining data (all of it, fraction=1.0)
X_remaining_test, y_remaining_test, label_encoder_remaining = load_and_preprocess_test_data(
    temp_remaining_path,
    fraction=1.0,  # Use all remaining samples
    random_seed=seed,  # Same seed for consistent preprocessing
    stats_source=CURRENT_STATS_SOURCE,
    stats_path=CURRENT_STATS_PATH,
    apply_normalization=True,
    normalization_type=CURRENT_NORMALIZATION,
    normalize_target_class=NORMALIZE_TARGET_CLASS
)

os.makedirs(f'out/{norm_name}_finetuned_test_{str(int(num_percentage_of_test_df * 100))}_percentage', exist_ok=True)
# Dynamic plot paths based on normalization method
plot_normalized_test_mean_feature_per_class(
    X_df=X_remaining_test,
    y_series=y_remaining_test,
    save_path=f'out/{norm_name}_finetuned_test_{str(int(num_percentage_of_test_df * 100))}_percentage/finetuned_test_mean_feature_per_class.png',
    title=f'{norm_description} Test Mean Feature per Class'
)

print(f"Remaining test data loaded: {len(X_remaining_test)} samples, {len(X_remaining_test.columns)} features")

# Create test DataLoader for remaining data
_, _, remaining_test_loader = tensor_dataset_classifier(batch_size=batch_size, X_test=X_remaining_test, y_test=y_remaining_test)

# Set fine-tuned models to evaluation mode
fine_tuned_autoencoder.eval()
fine_tuned_classifier.eval()

# Denoise remaining test data using fine-tuned autoencoder
print("Denoising remaining test data using fine-tuned autoencoder...")
X_remaining_test_denoised, y_remaining_test_final = evaluate_encoder_decoder_for_classifier(
    model_encoder_decoder=fine_tuned_autoencoder,
    data_loader=remaining_test_loader,
    device=device
)

# Create denoised test loader
remaining_test_dataset = torch.utils.data.TensorDataset(X_remaining_test_denoised, y_remaining_test_final)
denoised_remaining_test_loader = torch.utils.data.DataLoader(remaining_test_dataset, batch_size=batch_size, shuffle=False)

# Plot denoised features for remaining test data
plot_denoised_test_mean_feature_per_class(
    X_tensor=X_remaining_test_denoised,
    y_tensor=y_remaining_test_final,
    save_path=f'out/{norm_name}_finetuned_test_{str(int(num_percentage_of_test_df * 100))}_percentage/denoised_finetuned_test_mean_feature_per_class.png',
    title=f'Denoised {norm_description} Fine-tuned Test Mean Feature per Class'
)

# Evaluate fine-tuned classifier on remaining test data
print("Evaluating fine-tuned classifier on remaining test data...")
finetuned_test_model_name = f'classifier_{norm_name}_finetuned_test_{str(int(num_percentage_of_test_df * 100))}_percentage'
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_classifier=fine_tuned_classifier,
    test_loader=denoised_remaining_test_loader,
    device=device,
    label_encoder=label_encoder_remaining,
    model_name=finetuned_test_model_name
)

# Plot confusion matrix
output_dir = f'out/{norm_name}_finetuned_test_{str(int(num_percentage_of_test_df * 100))}_percentage'
plot_conf_matrix(conf_mat, label_encoder_remaining, model_name=finetuned_test_model_name, output_dir=output_dir)

# Print final results
print(f"\n=== Fine-tuned Model Test Results ===")
print(f"Normalization Method: {norm_description}")
print(f"Fine-tuning Data Used: {num_percentage_of_test_df:.1%} of test set")
print(f"Inference Data Used: {remaining_fraction:.1%} of test set")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"\n📁 Generated Files in out/{norm_name}/:")
print(f"   • denoised_{norm_name}_finetuned_test_mean_feature_per_class.png")
print(f"   • denoised_{norm_name}_finetuned_test_mean_feature_per_class_peaks_only.png (zoomed)")
print(f"   • confusion_matrix_{finetuned_test_model_name}.jpg")
print(f"   • {finetuned_test_model_name}.csv")

# Clean up temporary files
import os
if os.path.exists(temp_merged_path):
    os.remove(temp_merged_path)
if os.path.exists("temp_remaining_for_inference.csv"):
    os.remove("temp_remaining_for_inference.csv")

print("=== Fine-tuning and Testing Pipeline Completed ===")


