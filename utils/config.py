import matplotlib
import random
import torch
import numpy as np
import warnings
from utils.path_utils import get_paths_for_normalization

CURRENT_NORMALIZATION = 'class_based_robust'  # Change this to use different methods
base_path, stats_path = get_paths_for_normalization(CURRENT_NORMALIZATION)

# Recommended settings for autoencoder
autoencoder_patience = 7          # Learning rate patience
autoencoder_early_stopping = 15   # Early stopping patience

# For classifier (can keep more aggressive)
classifier_patience = 5
classifier_early_stopping = 6

# Hyperparameters
seed = 42
batch_size = 32
learning_rate = 1e-3
num_epochs = 500
total_num_chips = 500
num_chips = list(range(1, total_num_chips+1))
baseline_chip = 4
chip_column = "Chip"
class_column = "Class"
target_class = 4
current_path = f"{base_path}"
matplotlib.use('Agg')  # Use a non-interactive backend
torch.manual_seed(seed), torch.cuda.manual_seed_all(seed), np.random.seed(seed), random.seed(seed)
torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
warnings.filterwarnings("ignore")

# Normalization configuration
NORMALIZATION_CONFIG = {
    'class_based_mean_std': {
        'name': 'class_based_mean_std_normalized',
        'folder': 'mean_std',
        'description': 'Class-4 Mean/Std Normalization'
    },
    'class_based_minmax': {
        'name': 'minmax_normalized',
        'folder': 'minmax',
        'description': 'Class-4 Min-Max Normalization'
    },
    'class_based_robust': {
        'name': 'robust_normalized',
        'folder': 'robust',
        'description': 'Class-4 Robust Scaling Normalization'
    },
    'none': {
        'name': 'raw',
        'folder': 'raw',
        'description': 'No Normalization'
    }
}

# Current normalization configuration - available everywhere
norm_config = NORMALIZATION_CONFIG[CURRENT_NORMALIZATION]
norm_name = norm_config['name']
norm_folder = norm_config['folder']
norm_description = norm_config['description']

# Dynamic output directory structure
output_base_dir = f'out/{total_num_chips}chips/{norm_name}'

def print_current_config():
    """Print current configuration info"""
    print(f"Current config: {total_num_chips} chips, normalization: {norm_description}")
    print(f"Output directory: {output_base_dir}")

# Print configuration on import
print_current_config()