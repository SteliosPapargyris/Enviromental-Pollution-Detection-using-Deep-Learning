import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader


def train_encoder_decoder(epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_encoder_decoder, device, model_encoder_decoder_name, train_loader_chip1, val_loader_chip1, test_loader_chip1, chip_number=1, noise_factor=0.02):
    early_stopping_counter = 0
    model_encoder_decoder.to(device)
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    denoised_train_list, denoised_val_list = [], []

    for epoch in range(epochs):
        denoised_train_list = []
        denoised_val_list = []

        # Training phase
        model_encoder_decoder.train()
        total_train_loss = 0
        for (inputs, labels), (inputs_chip1, labels_chip1) in zip(train_loader, train_loader_chip1):

            inputs, labels = inputs.to(device), labels.to(device)
            inputs_chip1, labels_chip1 = inputs_chip1.to(device), labels_chip1.to(device)

            noisy_inputs = inputs + noise_factor * torch.randn(*inputs.shape, device=device)

            optimizer.zero_grad()

            # Pass data through denoiser
            denoised_inputs = model_encoder_decoder(noisy_inputs)[0] # Extract only the denoised output not latent space (check the class ConvDenoiser architecture)

            loss = criterion(denoised_inputs, inputs_chip1)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            denoised_train_list.append(denoised_inputs.cpu())

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f'Epoch {epoch} - Training Loss: {avg_train_loss:.6f}')

        # Validation phase
        model_encoder_decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for (inputs, labels), (inputs_chip1, labels_chip1) in zip(val_loader, val_loader_chip1):
                inputs, labels = inputs.to(device), labels.to(device)
                inputs_chip1, labels_chip1 = inputs_chip1.to(device), labels_chip1.to(device)

                # Add noise during validation as well (optional, for consistency)
                noisy_inputs = inputs + noise_factor * torch.randn(*inputs.shape, device=device)

                # Pass through denoiser and classifier
                denoised_inputs = model_encoder_decoder(noisy_inputs)[0]

                loss = criterion(denoised_inputs, inputs_chip1)
                total_val_loss += loss.item()

                denoised_val_list.append(denoised_inputs.cpu())
        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        print(f'Epoch {epoch} - Validation Loss: {avg_val_loss:.6f}')

        # Learning rate adjustment and checkpointing
        if scheduler:
            scheduler.step(avg_val_loss)
            current_lr = scheduler.get_last_lr()[0]
            print(f'Current learning rate: {current_lr}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print('Validation loss decreased, saving model.')
            torch.save(model_encoder_decoder.state_dict(), f'pths/{model_encoder_decoder_name}_{chip_number}.pth')
            early_stopping_counter = 0  # Reset the counter on improvement
        else:
            early_stopping_counter += 1  # Increment counter if no improvement
            print(f'No improvement in validation loss for {early_stopping_counter} consecutive epochs.')

        # Early stopping condition
        if early_stopping_counter >= 6:
            print('Early stopping triggered after 6 epochs with no improvement in validation loss.')
            model_encoder_decoder.load_state_dict(torch.load(f'pths/{model_encoder_decoder_name}_{chip_number}.pth'))
            print('Model restored to best state based on validation loss.')
            break

        print("\n")

    # Convert lists to numpy arrays
    X_denoised_train = torch.cat(denoised_train_list, dim=0).detach().cpu().numpy()
    X_denoised_val = torch.cat(denoised_val_list, dim=0).detach().cpu().numpy()
    X_denoised_test = None  # Test data will be denoised separately

    return model_encoder_decoder, training_losses, validation_losses, noise_factor, X_denoised_train, X_denoised_val, X_denoised_test


def evaluate_encoder_decoder(model_encoder_decoder, data_loader, device, criterion, test_loader_chip1, label_encoder, model_name, conv_layers, chip_number):
    model_encoder_decoder.eval()
    model_encoder_decoder.to(device)
    total_test_loss = 0

    denoised_test_list = []

    with torch.no_grad():
        for (inputs, _), (inputs_chip1, _) in zip(data_loader, test_loader_chip1):
            inputs = inputs.to(device)
            inputs_chip1 = inputs_chip1.to(device)

            noise_factor = 0.02
            noisy_inputs = inputs + noise_factor * torch.randn(*inputs.shape, device=device)

            denoised_inputs = model_encoder_decoder(noisy_inputs)[0]

            # Compute loss
            loss = criterion(denoised_inputs, inputs_chip1)
            total_test_loss += loss.item()

            denoised_test_list.append(denoised_inputs.cpu().detach())

    avg_test_loss = total_test_loss / len(data_loader)
    print(f'Average Test Loss: {avg_test_loss:.6f}')

    # Convert stored denoised test data into a NumPy array
    X_denoised_test = torch.cat(denoised_test_list, dim=0).numpy()

    return avg_test_loss, X_denoised_test


def train_classifier(epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_classifier, device, model_classifier_name):
    early_stopping_counter = 0
    model_classifier.to(device)
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        # Training phase
        model_classifier.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f'Epoch {epoch} - Training Loss: {avg_train_loss:.6f}')

        # Validation phase
        model_classifier.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model_classifier(inputs)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        print(f'Epoch {epoch} - Validation Loss: {avg_val_loss:.6f}')

        # Learning rate adjustment and checkpointing
        if scheduler:
            scheduler.step(avg_val_loss)
            current_lr = scheduler.get_last_lr()[0]
            print(f'Current learning rate: {current_lr}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print('Validation loss decreased, saving model.')
            torch.save(model_classifier.state_dict(), f'pths/{model_classifier_name}.pth')
            early_stopping_counter = 0  # Reset the counter on improvement
        else:
            early_stopping_counter += 1  # Increment counter if no improvement
            print(f'No improvement in validation loss for {early_stopping_counter} consecutive epochs.')

        # Early stopping condition
        if early_stopping_counter >= 6:
            print('Early stopping triggered after 6 epochs with no improvement in validation loss.')
            model_classifier.load_state_dict(torch.load(f'pths/{model_classifier_name}.pth'))
            print('Model restored to best state based on validation loss.')
            break

        print("\n")

    return model_classifier, training_losses, validation_losses


def evaluate_classifier(model_classifier, data_loader, device, label_encoder, model_name):
    model_classifier.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for (X, y) in data_loader:
            X = X.to(device)
            y = y.to(device)

            y_hat_test = model_classifier(X)
            _, predicted = torch.max(y_hat_test.data, 1)
            y_scores.extend(y_hat_test.cpu().numpy())

            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calculating additional metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    conf_mat = confusion_matrix(y_true, y_pred)

    # Generate classification report
    class_report = classification_report(
        y_true, y_pred,
        target_names=[str(class_name) for class_name in label_encoder.classes_],
        output_dict=True
    )
    print(class_report)

    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(class_report).transpose()

    # Create a unique filename using the model name and number of convolutional layers
    csv_filename = f"out\classification_reports/{model_name}.csv"

    # Save the classification report to a CSV file
    report_df.to_csv(csv_filename, index=True)

    print(f"Classification report saved to out/{csv_filename}")

    # Return metrics and ROC curve data
    return acc, prec, rec, f1, conf_mat