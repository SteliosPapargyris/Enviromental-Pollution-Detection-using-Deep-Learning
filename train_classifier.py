import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from utils.config import *
from utils.models import Classifier
from utils.train_test_utils import train_classifier, evaluate_classifier
from utils.plot_utils import plot_conf_matrix, plot_train_and_val_losses

data_path = f"out/{norm_name}/merged_autoencoder_outputs_{norm_name}.csv"

dataset = pd.read_csv(data_path)
print(f"Dataset shape: {dataset.shape}")
print(f"Dataset columns: {list(dataset.columns)}")
if 'Class' in dataset.columns:
    print(f"Unique classes: {dataset['Class'].unique()}")
    print(f"Class distribution: {dataset['Class'].value_counts()}")
else:
    print("No 'Class' column found!")

feature_columns = [col for col in dataset.columns if col not in ['Chip', 'Class', 'labels']]
X = dataset[feature_columns].values
y = dataset['Class'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Number of classes after encoding: {len(label_encoder.classes_)}")
print(f"Encoded classes: {label_encoder.classes_}")

X_train, X_temp, y_train, y_temp = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42, stratify=y_temp)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

model_classifier = Classifier(input_length=X_train_tensor.shape[2], num_classes=len(label_encoder.classes_)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_classifier.parameters(), lr=0.0001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, verbose=True)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

classifier_model_name = f'classifier_transfer_{norm_name}_to_baseline_{baseline_chip}'

model_classifier, training_losses, validation_losses = train_classifier(
    epochs=num_epochs,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
    model_classifier=model_classifier,
    device=device,
    model_classifier_name=classifier_model_name,
    early_stopping_patience=classifier_early_stopping
)

plot_train_and_val_losses(training_losses, validation_losses, classifier_model_name)

train_acc, train_prec, train_rec, train_f1, train_conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=train_loader,
    device=device,
    label_encoder=label_encoder,
    model_name=f'classifier_transfer_{norm_name}_training_eval'
)
plot_conf_matrix(train_conf_mat, label_encoder, model_name=f'classifier_transfer_{norm_name}_training_eval')

val_acc, val_prec, val_rec, val_f1, val_conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=val_loader,
    device=device,
    label_encoder=label_encoder,
    model_name=f'classifier_transfer_{norm_name}_validation_eval'
)
plot_conf_matrix(val_conf_mat, label_encoder, model_name=f'classifier_transfer_{norm_name}_validation_eval')

test_acc, test_prec, test_rec, test_f1, test_conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name=f'classifier_transfer_{norm_name}_test_eval'
)
plot_conf_matrix(test_conf_mat, label_encoder, model_name=f'classifier_transfer_{norm_name}_test_eval')