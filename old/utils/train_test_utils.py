import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd


def train(epochs, train_loader, val_loader, optimizer, criterion, scheduler, model_denoiser, model_classifier, device, model_denoiser_name, model_classifier_name, chip_number=1, noise_factor=0.01):
    early_stopping_counter = 0
    model_denoiser.to(device)
    model_classifier.to(device)
    best_val_loss = float('inf')
    training_losses = []
    validation_losses = []

    for epoch in range(epochs):
        # Training phase
        model_denoiser.train()
        model_classifier.train()
        total_train_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # noisy_inputs = noisy_inputs.to(device)
            labels = labels.long()
            # add random noise to the input data
            noisy_inputs = inputs + noise_factor * torch.randn(*inputs.shape, device=device)

            optimizer.zero_grad()
            # Pass data through denoiser
            denoised_outputs = model_denoiser(noisy_inputs)

            # Pass denoised outputs through classifier
            predictions = model_classifier(denoised_outputs)

            # # Ensure label and output shapes match
            # assert outputs.shape == labels.shape, f"Mismatch: Outputs {outputs.shape}, Labels {labels.shape}"

            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        training_losses.append(avg_train_loss)
        print(f'Epoch {epoch} - Training Loss: {avg_train_loss:.6f}')

        # Validation phase
        model_denoiser.eval()
        model_classifier.eval()
        total_val_loss = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                # Add noise during validation as well (optional, for consistency)
                noisy_inputs = inputs + noise_factor * torch.randn(*inputs.shape, device=device)

                # Pass through denoiser and classifier
                denoised_outputs = model_denoiser(noisy_inputs)
                predictions = model_classifier(denoised_outputs)

                loss = criterion(predictions, labels)
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
            torch.save(model_denoiser.state_dict(), f'pths/{model_denoiser_name}_{chip_number}.pth')
            torch.save(model_classifier.state_dict(), f'pths/{model_classifier_name}_{chip_number}.pth')
            early_stopping_counter = 0  # Reset the counter on improvement
        else:
            early_stopping_counter += 1  # Increment counter if no improvement
            print(f'No improvement in validation loss for {early_stopping_counter} consecutive epochs.')

        # Early stopping condition
        if early_stopping_counter >= 6:
            print('Early stopping triggered after 6 epochs with no improvement in validation loss.')
            model_denoiser.load_state_dict(torch.load(f'pths/{model_denoiser_name}_{chip_number}.pth'))
            model_classifier.load_state_dict(torch.load(f'pths/{model_classifier_name}_{chip_number}.pth'))
            print('Model restored to best state based on validation loss.')
            break

        print("\n")

    return model_denoiser, model_classifier, training_losses, validation_losses


def evaluate_model(model_denoiser, model_classifier, data_loader, device, label_encoder, model_name, conv_layers, chip_number):
    model_denoiser.eval()
    model_classifier.eval()
    y_true = []
    y_pred = []
    y_scores = []

    with torch.no_grad():
        for (X, y) in data_loader:
            X = X.to(device)
            y = y.to(device)

            noise_factor = 0.01
            noisy_inputs = X + noise_factor * torch.randn(*X.shape, device=device)

            denoised_X = model_denoiser(noisy_inputs)

            # Pass denoised inputs through the classifier
            y_hat_test = model_classifier(denoised_X)

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
    csv_filename = f"out\classification_reports/{model_name}_{conv_layers}_{chip_number}.csv"

    # Save the classification report to a CSV file
    report_df.to_csv(csv_filename, index=True)

    print(f"Classification report saved to out/{csv_filename}")

    # Return metrics and ROC curve data
    return acc, prec, rec, f1, conf_mat