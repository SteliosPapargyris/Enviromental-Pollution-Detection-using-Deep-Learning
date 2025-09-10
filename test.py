import torch
import torch.nn as nn
from utils.data_utils import tensor_dataset_classifier, load_and_preprocess_test_data
from utils.train_test_utils import evaluate_encoder_decoder_for_classifier, evaluate_classifier
from utils.plot_utils import plot_conf_matrix, plot_normalized_test_mean_feature_per_class, plot_denoised_test_mean_feature_per_class
from utils.models import LinearDenoiser, ConvDenoiser, Classifier
from utils.config import *

print("=== Test Pipeline Started ===")

# Load and preprocess test data
print("Loading and preprocessing test data...")

# Test configuration (can be overridden if needed)
CURRENT_STATS_SOURCE = 'compute'  # 'compute' or 'json'
CURRENT_STATS_PATH = stats_path  # Only used if CURRENT_STATS_SOURCE = 'json'
NORMALIZE_TARGET_CLASS = False

# Load test data with selected normalization
X_test, y_test, label_encoder = load_and_preprocess_test_data(
    file_path=test_file_path, 
    fraction=1, 
    stats_source=CURRENT_STATS_SOURCE, 
    stats_path=CURRENT_STATS_PATH,
    apply_normalization=True,
    normalization_type=CURRENT_NORMALIZATION,
    normalize_target_class=NORMALIZE_TARGET_CLASS
)

# Get normalization info for file naming
norm_config = NORMALIZATION_CONFIG[CURRENT_NORMALIZATION]
norm_name = norm_config['name']
norm_description = norm_config['description']

print(f"Applied normalization: {norm_description}")
print(f"File naming suffix: {norm_name}")

print(f"Test data loaded: {len(X_test)} samples, {len(X_test.columns)} features")

# Dynamic plot paths based on normalization method
plot_normalized_test_mean_feature_per_class(
    X_df=X_test,
    y_series=y_test,
    save_path=f'out/{norm_name}/{norm_name}_test_mean_feature_per_class.png',
    title=f'{norm_description} Test Mean Feature per Class'
)

# Create test DataLoader
print("Creating test data loader...")
_, _, test_loader = tensor_dataset_classifier(batch_size=batch_size, X_test=X_test, y_test=y_test)

# Dynamic model paths based on normalization method
autoencoder_model_name = f'autoencoder_{norm_name}_train'
classifier_model_name = f'classifier_{norm_name}_train'
autoencoder_path = f"pths/{norm_name}/{autoencoder_model_name}.pth"
classifier_path = f"pths/{norm_name}/{classifier_model_name}.pth"

print(f"Loading models from:")
print(f"  Autoencoder: {autoencoder_path}")
print(f"  Classifier: {classifier_path}")

# Initialize LinearDenoiser model
model_autoencoder = LinearDenoiser(input_size=33).to(device)
model_classifier = Classifier(input_length=33, num_classes=4).to(device)

# Set models to evaluation mode
model_autoencoder.eval()
model_classifier.eval()

# Load pretrained weights
model_autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
model_classifier.load_state_dict(torch.load(classifier_path, map_location=device))

print("Models loaded successfully!")

# Denoise test data
print("Denoising test data using pretrained autoencoder...")
X_test_denoised, y_test = evaluate_encoder_decoder_for_classifier(
    model_encoder_decoder=model_autoencoder, 
    data_loader=test_loader, 
    device=device
)
# Create denoised test loader
test_dataset = torch.utils.data.TensorDataset(X_test_denoised, y_test)
denoised_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Plot denoised features
plot_denoised_test_mean_feature_per_class(
    X_tensor=X_test_denoised,
    y_tensor=y_test,
    save_path=f'out/{norm_name}/denoised_{norm_name}_test_mean_feature_per_class.png',
    title=f'Denoised {norm_description} Test Mean Feature per Class'
)

# Evaluate classifier
print("Evaluating classifier on test data...")
test_model_name = f'classifier_{norm_name}_test'
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=denoised_test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name=test_model_name
)

# Plot confusion matrix
plot_conf_matrix(conf_mat, label_encoder, model_name=test_model_name)

# Print final results
print(f"\n=== Final Test Results ===")
print(f"Normalization Method: {norm_description}")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print(f"\n📁 Generated Files in out/{norm_name}/:")
print(f"   • {norm_name}_test_mean_feature_per_class.png")
print(f"   • denoised_{norm_name}_test_mean_feature_per_class.png")
print(f"   • denoised_{norm_name}_test_mean_feature_per_class_peaks_only.png (zoomed)")
print(f"   • confusion_matrix_{test_model_name}.jpg")
print(f"   • {test_model_name}.csv")
print("=== Test Pipeline Completed ===")