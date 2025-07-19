# utils.py

import matplotlib.pyplot as plt

def plot_metrics(train_losses, val_losses, val_accuracies):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.plot(val_accuracies, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training Performance")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("training_plot.png")
    plt.show()

