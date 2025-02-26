import matplotlib.pyplot as plt
import seaborn as sns


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
