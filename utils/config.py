import matplotlib
import random
import torch
import numpy as np

# Hyperparameters
seed = 42
batch_size = 32
patience = 10
early_stopping_max_number = 20
learning_rate = 1e-3
num_epochs = 500
num_classes = 4
chip_exclude = 50
num_chips = list(range(1, chip_exclude))
baseline_chip = 4
chip_column = "Chip"
class_column = "Class"
target_class = 4
num_chip_selection = 50
base_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/per_peak_standardscaler_excl_chip_class{target_class}/fts_mzi_dataset/{num_chip_selection}chips_20percent_noise"
current_path = f"{base_path}"
test_file_path = f'{base_path}/{chip_exclude}.csv'
matplotlib.use('Agg')  # Use a non-interactive backend
torch.manual_seed(seed), torch.cuda.manual_seed_all(seed), np.random.seed(seed), random.seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")