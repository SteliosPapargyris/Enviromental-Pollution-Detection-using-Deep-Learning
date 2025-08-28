import torch
import torch.nn as nn
from utils.data_utils import tensor_dataset_classifier, load_and_preprocess_test_data
from utils.train_test_utils import evaluate_encoder_decoder_for_classifier, evaluate_classifier
from utils.plot_utils import plot_conf_matrix, plot_normalized_test_mean_feature_per_class, plot_denoised_test_mean_feature_per_class
from utils.models import LinearDenoiser, ConvDenoiser, Classifier
from utils.config import *

# Load and preprocess test data
X_test, y_test, label_encoder = load_and_preprocess_test_data(file_path=test_file_path, fraction=1)

plot_normalized_test_mean_feature_per_class(
    X_df=X_test,
    y_series=y_test,
    save_path='out/normalized_test_mean_feature_per_class.png',
    title='Normalized Test Mean Feature per Class'
)

# Create test DataLoader
_, _, test_loader = tensor_dataset_classifier(batch_size=batch_size, X_test=X_test, y_test=y_test)

# autoencoder_path = "pths/autoencoder_train.pth"
classifier_path = "pths/classifier_train.pth"

# Initialize LinearDenoiser model
# model_autoencoder = LinearDenoiser().to(device)
# model_autoencoder = ConvDenoiser().to(device)
model_classifier = Classifier().to(device)

# Set models to evaluation mode
# model_autoencoder.eval()
model_classifier.eval()

# model_autoencoder.load_state_dict(torch.load(autoencoder_path))
model_classifier.load_state_dict(torch.load(classifier_path))

# Define loss function, optimizer, and scheduler
criterion = nn.MSELoss()

# # Evaluate the model on the test set
# X_test_denoised, y_test = evaluate_encoder_decoder_for_classifier(model_encoder_decoder=model_autoencoder, data_loader=test_loader, device=device)
# test_dataset = torch.utils.data.TensorDataset(X_test_denoised, y_test)
# denoised_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# plot_denoised_test_mean_feature_per_class(
#     X_tensor=X_test_denoised,
#     y_tensor=y_test,
#     save_path='out/denoised_test_mean_feature_per_class.png',
#     title='Denoised Test Mean Feature per Class'
# )


X_te= torch.as_tensor(X_test.to_numpy(dtype="float32")).unsqueeze(1)  # shape [280, 33], float32 for features
y_te = torch.as_tensor(y_test.to_numpy(dtype="int64"))    # shape [280],   int64 for labels
test_dataset = torch.utils.data.TensorDataset(X_te, y_te)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Evaluate classifier
acc, prec, rec, f1, conf_mat = evaluate_classifier(
    model_classifier=model_classifier,
    test_loader=test_loader,
    device=device,
    label_encoder=label_encoder,
    model_name='classifier_test'
)

plot_conf_matrix(conf_mat, label_encoder, model_name='classifier_test')