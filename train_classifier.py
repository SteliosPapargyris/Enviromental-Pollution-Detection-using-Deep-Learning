import torch
from utils.config import *
from utils.models import LinearDenoiser, ConvDenoiser, Classifier
from utils.data_utils import load_and_preprocess_data_classifier, tensor_dataset_classifier
from utils.train_test_utils import train_classifier, evaluate_classifier, evaluate_encoder_decoder_for_classifier
from utils.plot_utils import plot_conf_matrix, plot_train_and_val_losses, plot_denoised_mean_feature_per_class_before_classifier
import torch.nn as nn
import torch.optim as optim

print("=== Classifier Training Pipeline Started ===")

# Get normalization info for file naming
norm_config = NORMALIZATION_CONFIG[CURRENT_NORMALIZATION]
norm_name = norm_config['name']
norm_description = norm_config['description']

print(f"Training classifier with normalization: {norm_description}")
print(f"File naming suffix: {norm_name}")

# Load the shuffled dataset for the current chip
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess_data_classifier(file_path=f"{current_path}/shuffled_dataset/merged.csv")

# Create data loaders for raw data
train_loader, val_loader, test_loader = tensor_dataset_classifier(batch_size=batch_size, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

# Load the appropriate autoencoder model based on normalization method
autoencoder_model_name = f'autoencoder_{norm_name}_train'
autoencoder_path = f"pths/{norm_name}/{autoencoder_model_name}.pth"

print(f"Loading autoencoder model: {autoencoder_path}")
model_autoencoder = LinearDenoiser(input_size=33).to(device)
model_autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
model_autoencoder.eval()

model_classifier = Classifier(input_length=33, num_classes=4).to(device)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_classifier.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=classifier_patience, verbose=True)

X_train_denoised, y_train = evaluate_encoder_decoder_for_classifier(model_encoder_decoder=model_autoencoder, data_loader=train_loader, device=device)
X_val_denoised, y_val = evaluate_encoder_decoder_for_classifier(model_encoder_decoder=model_autoencoder, data_loader=val_loader, device=device)
X_test_denoised, y_test = evaluate_encoder_decoder_for_classifier(model_encoder_decoder=model_autoencoder, data_loader=test_loader, device=device)

# Plot of mean denoised feature per class before classifier training with dynamic naming
plot_denoised_mean_feature_per_class_before_classifier(
    X_tensor=X_train_denoised,
    y_tensor=y_train,
    save_path=f'out/{norm_name}/denoised_{norm_name}_train_mean_feature_per_class.png',
    title=f'Mean Denoised {norm_description} Peaks per Class before Classifier'
)

# Create new DataLoaders for classifier training
train_dataset = torch.utils.data.TensorDataset(X_train_denoised, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val_denoised, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test_denoised, y_test)

denoised_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
denoised_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
denoised_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Train the model on the current chip with dynamic naming
classifier_model_name = f'classifier_{norm_name}_train'
model_classifier, training_losses, validation_losses = train_classifier(
    epochs=num_epochs,
    train_loader=denoised_train_loader,
    val_loader=denoised_val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    model_classifier=model_classifier,
    device=device,
    model_classifier_name=classifier_model_name,
    early_stopping_patience=classifier_early_stopping
)

# Plot training and validation losses with dynamic naming
plot_train_and_val_losses(training_losses, validation_losses, classifier_model_name)

# Evaluate on TRAINING data with dynamic naming
print("Evaluating on training data...")
train_eval_name = f'classifier_{norm_name}_training_eval'
train_acc, train_prec, train_rec, train_f1, train_conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=denoised_train_loader,  # Training data
    device=device,
    label_encoder=label_encoder,
    model_name=train_eval_name
)
plot_conf_matrix(train_conf_mat, label_encoder, model_name=train_eval_name)

# Evaluate on TEST data with dynamic naming
print("Evaluating on test data...")
test_eval_name = f'classifier_{norm_name}_validation_eval'
test_acc, test_prec, test_rec, test_f1, test_conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=denoised_test_loader,   # Test data
    device=device,
    label_encoder=label_encoder,
    model_name=test_eval_name
)
plot_conf_matrix(test_conf_mat, label_encoder, model_name=test_eval_name)

# Compare results with normalization info
print(f"\n=== Classifier Training Results ===")
print(f"Normalization Method: {norm_description}")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Generalization Gap: {train_acc - test_acc:.4f}")
print(f"\n📁 Generated Files in out/{norm_name}/:")
print(f"   • denoised_{norm_name}_train_mean_feature_per_class.png")
print(f"   • denoised_{norm_name}_train_mean_feature_per_class_peaks_only.png (zoomed)")
print(f"   • train_and_val_loss_{classifier_model_name}.png")
print(f"   • pths/{norm_name}/{classifier_model_name}.pth")
print(f"   • confusion_matrix_{train_eval_name}.jpg")
print(f"   • confusion_matrix_{test_eval_name}.jpg")
print(f"   • {train_eval_name}.csv")
print(f"   • {test_eval_name}.csv")
print("=== Classifier Training Completed ===")