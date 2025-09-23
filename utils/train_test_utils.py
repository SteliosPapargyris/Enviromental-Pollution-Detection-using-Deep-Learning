import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd
from utils.config import *
import os 

def train_encoder_decoder(epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_encoder_decoder, device, model_encoder_decoder_name, early_stopping_patience):
    early_stopping_counter = 0
    model_encoder_decoder.to(device)
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        # Training phase
        model_encoder_decoder.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
            optimizer.zero_grad()
            denoised_output, latent_space = model_encoder_decoder(inputs)
            loss = criterion(denoised_output, labels)
            # loss = criterion(latent_space, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f'Epoch {epoch} - Training Loss: {avg_train_loss:.6f}')

        # Validation phase
        model_encoder_decoder.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                inputs = inputs.unsqueeze(1)
                labels = labels.unsqueeze(1)
                denoised_output, latent_space = model_encoder_decoder(inputs)
                loss = criterion(denoised_output, labels)
                # loss = criterion(latent_space, labels)
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
            # Create normalization-specific directory structure
            # Extract norm type from model name (e.g., "autoencoder_minmax_normalized_train" -> "minmax_normalized")
            name_parts = model_encoder_decoder_name.split('_')
            if len(name_parts) >= 3 and name_parts[0] == 'autoencoder':
                norm_type = '_'.join(name_parts[1:-1])  # Everything between 'autoencoder' and 'train'
            else:
                norm_type = 'default'
            model_dir = f'pths/{norm_type}'
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model_encoder_decoder.state_dict(), f'{model_dir}/{model_encoder_decoder_name}.pth')
            early_stopping_counter = 0  # Reset the counter on improvement
        else:
            early_stopping_counter += 1  # Increment counter if no improvement
            print(f'No improvement in validation loss for {early_stopping_counter} consecutive epochs.')

        # Early stopping condition
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {early_stopping_patience} epochs with no improvement in validation loss.')
            # Extract norm type from model name (e.g., "autoencoder_minmax_normalized_train" -> "minmax_normalized")
            name_parts = model_encoder_decoder_name.split('_')
            if len(name_parts) >= 3 and name_parts[0] == 'autoencoder':
                norm_type = '_'.join(name_parts[1:-1])  # Everything between 'autoencoder' and 'train'
            else:
                norm_type = 'default'
            model_dir = f'pths/{norm_type}'
            model_encoder_decoder.load_state_dict(torch.load(f'{model_dir}/{model_encoder_decoder_name}.pth'))
            print('Model restored to best state based on validation loss.')
            break
        print("\n")
    return model_encoder_decoder, training_losses, validation_losses


def evaluate_encoder_decoder(model_encoder_decoder, test_loader, device, criterion):
    model_encoder_decoder.eval()
    model_encoder_decoder.to(device)
    total_test_loss = 0
    denoised_data = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            inputs = inputs.unsqueeze(1)
            labels = labels.unsqueeze(1)
            denoised_output, latent_space = model_encoder_decoder(inputs)
            loss = criterion(denoised_output, labels)
            # loss = criterion(latent_space, labels)
            total_test_loss += loss.item()
            denoised_data.append(denoised_output.cpu())
            all_labels.append(labels.cpu())

    avg_test_loss = total_test_loss / len(test_loader)
    print(f'Final Test Loss: {avg_test_loss:.6f}')

    denoised_data = torch.cat(denoised_data, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return avg_test_loss, denoised_data, all_labels

def train_classifier(epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_classifier, device, model_classifier_name, early_stopping_patience):
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
            # Create normalization-specific directory structure
            # Extract norm type from model name (e.g., "classifier_minmax_normalized_train" -> "minmax_normalized")
            name_parts = model_classifier_name.split('_')
            if len(name_parts) >= 3 and name_parts[0] == 'classifier':
                norm_type = '_'.join(name_parts[1:-1])  # Everything between 'classifier' and 'train'
            else:
                norm_type = 'default'
            model_dir = f'pths/{norm_type}'
            os.makedirs(model_dir, exist_ok=True)
            torch.save(model_classifier.state_dict(), f'{model_dir}/{model_classifier_name}.pth')
            early_stopping_counter = 0  # Reset the counter on improvement
        else:
            early_stopping_counter += 1  # Increment counter if no improvement
            print(f'No improvement in validation loss for {early_stopping_counter} consecutive epochs.')

        # Early stopping condition
        # TODO early_stopping patience to config.py
        if early_stopping_counter >= early_stopping_patience:
            print(f'Early stopping triggered after {early_stopping_patience} epochs with no improvement in validation loss.')
            # Extract norm type from model name (e.g., "classifier_minmax_normalized_train" -> "minmax_normalized")
            name_parts = model_classifier_name.split('_')
            if len(name_parts) >= 3 and name_parts[0] == 'classifier':
                norm_type = '_'.join(name_parts[1:-1])  # Everything between 'classifier' and 'train'
            else:
                norm_type = 'default'
            model_dir = f'pths/{norm_type}'
            model_classifier.load_state_dict(torch.load(f'{model_dir}/{model_classifier_name}.pth'))
            print('Model restored to best state based on validation loss.')
            break

        print("\n")

    return model_classifier, training_losses, validation_losses


def evaluate_classifier(model_classifier, test_loader, device, label_encoder, model_name):
    model_classifier.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for (X, y) in test_loader:
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

    # Use the dynamic output directory from config
    output_dir = output_base_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert the classification report to a DataFrame
    report_df = pd.DataFrame(class_report).transpose()

    # Create a unique filename using the model name - save to normalization-specific folder
    csv_filename = f"{output_dir}/{model_name}.csv"

    # Save the classification report to a CSV file
    report_df.to_csv(csv_filename, index=True)

    print(f"Classification report saved to {csv_filename}")

    # Return metrics and ROC curve data
    return acc, prec, rec, f1, conf_mat