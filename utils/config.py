import matplotlib
import random
import torch
import numpy as np

# Hyperparameters
seed = 42
batch_size = 32

# Recommended settings for autoencoder
autoencoder_patience = 7          # Learning rate patience
autoencoder_early_stopping = 15   # Early stopping patience

# For classifier (can keep more aggressive)
classifier_patience = 3
classifier_early_stopping = 6

early_stopping_max_number = 6
learning_rate = 1e-3
num_epochs = 500
num_classes = 4
chip_exclude = 5
num_chips = list(range(1, chip_exclude))
baseline_chip = 4
chip_column = "Chip"
class_column = "Class"
normalization_technique = "standardscaler"
target_class = 4
num_chip_selection = 5
base_path = f"data/out/normalized"
stats_path = f"data/fts_mzi_dataset/normalization_statistics.json"
# base_path = "/Users/steliospapargyris/Documents/MyProjects/data_thesis/no_normalization/5chips_old"
# base_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_{target_class}_of_every_chip/{num_chip_selection}chips_old"
# base_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_{target_class}_of_every_chip/{num_chip_selection}chips_no_noise"
# base_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_{target_class}_of_every_chip/{num_chip_selection}chips_20percent_noise"
# stats_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_{target_class}_of_every_chip/class_{target_class}_mean_and_std/fts_mzi_dataset/statistics/class_{target_class}_normalization_stats.json"
# col_stats_path = f"/Users/steliospapargyris/Documents/MyProjects/data_thesis/mean_and_std_of_class_{target_class}_of_every_chip/class_{target_class}_mean_and_std/fts_mzi_dataset/class_{target_class}_per_chip_stats.csv"
current_path = f"{base_path}"
test_file_path = f'{base_path}/{chip_exclude}.csv'
matplotlib.use('Agg')  # Use a non-interactive backend
torch.manual_seed(seed), torch.cuda.manual_seed_all(seed), np.random.seed(seed), random.seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")