import matplotlib
import random
import torch
import numpy as np

# Hyperparameters
seed = 42
batch_size = 32
patience = 2
early_stopping_max_number = 6
learning_rate = 1e-3
num_epochs = 500
num_classes = 4
chip_exclude = 10
num_chips = list(range(1, chip_exclude))
baseline_chip = 4
chip_column = "Chip"
class_column = "Class"
# normalization_technique = "standardscaler"
target_class = 4
num_chip_selection = 10
# base_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/per_peak_{normalization_technique}_excl_chip_class{target_class}/fts_mzi_dataset/{num_chip_selection}chips_20percent_noise"
# col_stats_path = f'/Users/steliospapargyris/Documents/MyProjects/data_thesis/per_peak_standardscaler_excl_chip_class{target_class}/fts_mzi_dataset/{chip_exclude}chips_20percent_noise/per_peak_standardscaler_exclude_chip_and_class{chip_exclude}_class{target_class}.csv'
base_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_{target_class}_of_every_chip/{num_chip_selection}chips_no_noise_exp(std)"
# col_stats_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_{target_class}_of_every_chip/class_{target_class}_mean_and_std/fts_mzi_dataset/class_{target_class}_per_chip_stats.csv"
current_path = f"{base_path}"
test_file_path = f'{base_path}/{chip_exclude}.csv'
matplotlib.use('Agg')  # Use a non-interactive backend
torch.manual_seed(seed), torch.cuda.manual_seed_all(seed), np.random.seed(seed), random.seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")