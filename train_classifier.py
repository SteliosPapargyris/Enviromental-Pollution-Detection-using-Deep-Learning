import torch
from utils.config import *
from utils.models import ConvDenoiser, Classifier
from utils.data_utils import load_and_preprocess_data_classifier, tensor_dataset_classifier
from utils.train_test_utils import train_classifier, evaluate_classifier
from utils.plot_utils import plot_conf_matrix, plot_train_and_val_losses, plot_denoised_mean_feature_per_class_before_classifier
import torch.nn as nn
import torch.optim as optim

# Load the shuffled dataset for the current chip
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess_data_classifier(file_path=f"{current_path}/shuffled_dataset/merged.csv")

# Create data loaders for raw data
train_loader, val_loader, test_loader = tensor_dataset_classifier(batch_size=batch_size, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

model_classifier = Classifier().to(device)

X_tr = torch.as_tensor(X_train.to_numpy(dtype="float32")).unsqueeze(1)  # shape [1120, 33], float32 for features
y_tr = torch.as_tensor(y_train.to_numpy(dtype="int64"))    # shape [1120],   int64 for labels (e.g., for CrossEntropyLoss)
X_v = torch.as_tensor(X_val.to_numpy(dtype="float32")).unsqueeze(1)      # shape [280, 33], float32 for features
y_v = torch.as_tensor(y_val.to_numpy(dtype="int64"))        # shape [280],   int64 for labels
X_te= torch.as_tensor(X_test.to_numpy(dtype="float32")).unsqueeze(1)  # shape [280, 33], float32 for features
y_te = torch.as_tensor(y_test.to_numpy(dtype="int64"))    # shape [280],   int64 for labels

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience, verbose=True)

# Create new DataLoaders for classifier training
train_dataset = torch.utils.data.TensorDataset(X_tr, y_tr)
val_dataset = torch.utils.data.TensorDataset(X_v, y_v)
test_dataset = torch.utils.data.TensorDataset(X_te, y_te)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Train the model on the current chip
model_classifier, training_losses, validation_losses = train_classifier(
    epochs=num_epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    model_classifier=model_classifier,
    device=device,
    model_classifier_name='classifier_train',
)

# Plot training and validation losses
plot_train_and_val_losses(training_losses, validation_losses, 'classifier_train')

# Evaluate the model on the test set
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name='classifier_train'
)

# Plot confusion matrix
plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_train')