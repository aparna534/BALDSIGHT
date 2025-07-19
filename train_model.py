import os
import shutil
import torch
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def ensure_val_split(data_dir, val_ratio=0.2):
    """
    Splits training data into training and validation if val folder doesn't exist or is empty.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.exists(val_dir) or all(
        len(os.listdir(os.path.join(val_dir, d))) == 0
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ):
        print("[INFO] Creating validation split...")
        os.makedirs(val_dir, exist_ok=True)
        for cls in os.listdir(train_dir):
            cls_path = os.path.join(train_dir, cls)
            if not os.path.isdir(cls_path):
                continue  # skip .DS_Store etc.
            images = os.listdir(cls_path)
            os.makedirs(os.path.join(val_dir, cls), exist_ok=True)
            split_size = int(len(images) * val_ratio)
            val_imgs = images[:split_size]
            for img in val_imgs:
                shutil.move(
                    os.path.join(cls_path, img),
                    os.path.join(val_dir, cls, img)
                )


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss, correct = 0.0, 0
    total = 0
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss, correct = 0.0, 0
    total = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_preds.extend(preds.cpu())
            all_labels.extend(labels.cpu())

    epoch_loss = val_loss / len(dataloader)
    epoch_acc = correct / total
    print("\nClassification Report:\n", classification_report(all_labels, all_preds))
    return epoch_loss, epoch_acc


def plot_stats(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    plt.show()


def main():
    data_dir = "./datasets"
    ensure_val_split(data_dir)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform)

    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(train_set.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    epochs = 10
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

    torch.save(model.state_dict(), "baldsight_resnet18.pth")
    plot_stats(train_losses, val_losses, train_accs, val_accs)


if __name__ == "__main__":
    main()

