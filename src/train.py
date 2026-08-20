import torch
import torch.nn as nn
import pandas as pd
from torchvision import models

from src.dataloaders import get_dataloaders


def build_model(num_classes=4):
    model = models.resnet50(weights="IMAGENET1K_V2")

    for param in model.parameters():
        param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


def compute_pos_weights(labels_csv="data/severstal/train_split.csv"):
    df = pd.read_csv(labels_csv)
    defect_cols = ["defect_1", "defect_2", "defect_3", "defect_4"]

    pos_weights = []
    for col in defect_cols:
        num_pos = df[col].sum()
        num_neg = len(df) - num_pos
        weight = num_neg / num_pos
        pos_weights.append(weight)
        print(f"{col}: {int(num_pos)} positive, {int(num_neg)} negative, pos_weight={weight:.2f}")

    return torch.tensor(pos_weights, dtype=torch.float32)


def get_loss_and_optimizer(model, pos_weight=None, learning_rate=0.001):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    return criterion, optimizer


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(val_loader.dataset)
    return epoch_loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = get_dataloaders(batch_size=32)

    model = build_model(num_classes=4)
    model = model.to(device)

    pos_weights = compute_pos_weights()
    pos_weights = pos_weights.to(device)

    criterion, optimizer = get_loss_and_optimizer(model, pos_weight=pos_weights, learning_rate=0.001)

    num_epochs = 5
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{num_epochs} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")

    torch.save(model.state_dict(), "models/defect_model_weighted.pth")
    print("\nSaved model -> models/defect_model_weighted.pth")