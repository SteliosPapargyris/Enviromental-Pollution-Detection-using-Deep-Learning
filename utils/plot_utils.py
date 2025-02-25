import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle


def plot_conf_matrix(conf_matrix, label_encoder, model_name):
    plt.figure()
    ax = sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cbar=False,
        cmap="Blues")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticklabels(label_encoder.classes_, rotation=45, ha='right')
    ax.set_yticklabels(label_encoder.classes_, rotation=0)
    plt.title(f'Confusion Matrix_{model_name}')
    plt.tight_layout()
    plt.savefig(f"out/confusion_matrix_{model_name}.jpg")
    plt.show()


def plot_macro_roc_curve(fpr, tpr, roc_auc, fpr_macro, tpr_macro, roc_auc_macro, label_encoder, model_name, chip_number):
    num_classes = len(label_encoder.classes_)
    colors = cycle(
        ["aqua", "darkorange", "cornflowerblue", "green", "red", "black", "yellow", "pink", "purple", "brown", "gray",
         "orange"])
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(num_classes), colors):
        label = label_encoder.classes_[i]
        plt.plot(fpr[label], tpr[label], color=color, lw=2,
                 label=f"ROC curve of class {label} has auc {roc_auc[label]:.2f}")
    plt.plot(fpr_macro, tpr_macro, color="navy", lw=4, label=f"Macro ROC curve auc value {roc_auc_macro}")
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC-Macro Average_{model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"out/roc_curve_{model_name}_{chip_number}.jpg")
    plt.show()


def plot_train_and_val_losses(training_losses, validation_losses, model_name, chip_number):
    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title(f'Training and Validation Loss Per Epoch_{model_name}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'out/train_and_val_loss_{model_name}_{chip_number}.png')
    plt.show()
