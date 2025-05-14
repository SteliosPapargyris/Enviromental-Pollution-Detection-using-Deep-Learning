import torch
from utils.config import *
from utils.models import ConvDenoiser, Classifier
from utils.data_utils import load_and_preprocess_data_classifier, tensor_dataset_classifier
from utils.train_test_utils import train_classifier, evaluate_classifier, evaluate_encoder_decoder_for_classifier
from utils.plot_utils import plot_conf_matrix, plot_train_and_val_losses
import torch.nn as nn
import torch.optim as optim


# Load the shuffled dataset for the current chip
X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_and_preprocess_data_classifier(file_path=f"{current_path}/shuffled_dataset/merged.csv")

# Create data loaders for raw data
train_loader, val_loader, test_loader = tensor_dataset_classifier(batch_size=batch_size, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

model_autoencoder = ConvDenoiser().to(device)
model_autoencoder.load_state_dict(torch.load("pths/autoencoder_train.pth", map_location=device))
model_autoencoder.eval()
model_classifier = Classifier().to(device)

# Define loss function, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_classifier.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

X_train_denoised, y_train = evaluate_encoder_decoder_for_classifier(model_encoder_decoder=model_autoencoder, data_loader=train_loader, device=device)
X_val_denoised, y_val = evaluate_encoder_decoder_for_classifier(model_encoder_decoder=model_autoencoder, data_loader=val_loader, device=device)
X_test_denoised, y_test = evaluate_encoder_decoder_for_classifier(model_encoder_decoder=model_autoencoder, data_loader=test_loader, device=device)

# Create new DataLoaders for classifier training
train_dataset = torch.utils.data.TensorDataset(X_train_denoised, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val_denoised, y_val)
test_dataset = torch.utils.data.TensorDataset(X_test_denoised, y_test)

denoised_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
denoised_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
denoised_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Train the model on the current chip
model_classifier, training_losses, validation_losses = train_classifier(
    epochs=num_epochs,
    train_loader=denoised_train_loader,
    val_loader=denoised_val_loader,
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
    test_loader=denoised_test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name='classifier_train'
)

# Plot confusion matrix
plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_train')