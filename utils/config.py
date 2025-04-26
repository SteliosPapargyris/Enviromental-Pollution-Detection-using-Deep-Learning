import matplotlib
import random
import torch
import numpy as np

# Hyperparameters
seed = 42
batch_size = 32
learning_rate = 1e-3
num_epochs = 500
num_classes = 4
base_path = "/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_4_of_every_chip/shuffled_dataset"
current_path = f"{base_path}"
test_file_path = f'{base_path}/5.csv'
matplotlib.use('Agg')  # Use a non-interactive backend
torch.manual_seed(seed), torch.cuda.manual_seed_all(seed), np.random.seed(seed), random.seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")