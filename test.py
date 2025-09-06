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

# Option 1: Class-based normalization (original behavior) - don't normalize target class
X_test, y_test, label_encoder = load_and_preprocess_test_data(
    file_path=test_file_path, 
    fraction=1, 
    stats_source="compute", 
    stats_path=stats_path,
    apply_normalization=True,
    normalization_type='class_based',
    normalize_target_class=False
)

# Option 2: Class-based normalization - normalize ALL classes including target
# X_test, y_test, label_encoder = load_and_preprocess_test_data(
#     file_path=test_file_path, 
#     fraction=1, 
#     stats_source="compute", 
#     stats_path=stats_path,
#     apply_normalization=True,
#     normalization_type='class_based',
#     normalize_target_class=True
# )

# Option 3: Standard z-score normalization
# X_test, y_test, label_encoder = load_and_preprocess_test_data(
#     file_path=test_file_path, 
#     fraction=1, 
#     stats_source="compute", 
#     stats_path=stats_path,
#     apply_normalization=True,
#     normalization_type='standard',
#     normalize_target_class=False  # This parameter is ignored for standard normalization
# )

# Option 4: Min-max normalization
# X_test, y_test, label_encoder = load_and_preprocess_test_data(
#     file_path=test_file_path, 
#     fraction=1, 
#     stats_source="compute", 
#     stats_path=stats_path,
#     apply_normalization=True,
#     normalization_type='minmax'
# )

# Option 5: No normalization
# X_test, y_test, label_encoder = load_and_preprocess_test_data(
#     file_path=test_file_path, 
#     fraction=1, 
#     stats_source="compute", 
#     stats_path=stats_path,
#     apply_normalization=False,
#     normalization_type='none'
# )

# Option 6: Load stats from JSON file with class-based normalization
# X_test, y_test, label_encoder = load_and_preprocess_test_data(
#     file_path=test_file_path, 
#     fraction=1, 
#     stats_source="json", 
#     stats_path=stats_path,
#     apply_normalization=True,
#     normalization_type='class_based',
#     normalize_target_class=False
# )

print(f"Test data loaded: {len(X_test)} samples, {len(X_test.columns)} features")

plot_normalized_test_mean_feature_per_class(
    X_df=X_test,
    y_series=y_test,
    save_path='out/normalized_test_mean_feature_per_class.png',
    title='Normalized Test Mean Feature per Class'
)

# Create test DataLoader
print("Creating test data loader...")
_, _, test_loader = tensor_dataset_classifier(batch_size=batch_size, X_test=X_test, y_test=y_test)

autoencoder_path = "pths/autoencoder_train.pth"
classifier_path = "pths/classifier_train.pth"

print(f"Loading models from:")
print(f"  Autoencoder: {autoencoder_path}")
print(f"  Classifier: {classifier_path}")

# Initialize LinearDenoiser model
model_autoencoder = LinearDenoiser(input_size=33).to(device)
# model_autoencoder = ConvDenoiser(input_length=33).to(device)
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
    save_path='out/denoised_test_mean_feature_per_class.png',
    title='Denoised Test Mean Feature per Class'
)

# Evaluate classifier
print("Evaluating classifier on test data...")
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=denoised_test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name='classifier_test'
)

# Plot confusion matrix
plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_test')

# Print final results
print(f"\n=== Final Test Results ===")
print(f"Test Accuracy: {acc:.4f}")
print(f"Test Precision: {prec:.4f}")
print(f"Test Recall: {rec:.4f}")
print(f"Test F1-Score: {f1:.4f}")
print("=== Test Pipeline Completed ===")