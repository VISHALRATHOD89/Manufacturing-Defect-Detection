import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

from src.train import build_model
from src.dataloaders import get_dataloaders


def load_trained_model(weights_path="models/defect_model.pth", device="cpu"):
    model = build_model(num_classes=4)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def get_predictions(model, val_loader, device, threshold=0.5):
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs >= threshold).float()

            all_labels.append(labels.numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return np.vstack(all_labels), np.vstack(all_preds), np.vstack(all_probs)


def print_classification_report(labels, preds):
    defect_names = ["defect_1", "defect_2", "defect_3", "defect_4"]

    for i, name in enumerate(defect_names):
        print(f"\n=== {name} ===")
        print(classification_report(labels[:, i], preds[:, i], target_names=["No Defect", "Defect"], zero_division=0))


def find_best_thresholds(labels, probs):
    defect_names = ["defect_1", "defect_2", "defect_3", "defect_4"]
    best_thresholds = {}

    for i, name in enumerate(defect_names):
        precisions, recalls, thresholds = precision_recall_curve(labels[:, i], probs[:, i])

        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores[:-1])

        best_threshold = thresholds[best_idx]
        best_thresholds[name] = best_threshold
        print(f"{name}: best threshold = {best_threshold:.3f} "
              f"(precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}, F1={f1_scores[best_idx]:.3f})")

    return best_thresholds


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, val_loader = get_dataloaders(batch_size=32)
    model = load_trained_model(weights_path="models/defect_model_weighted.pth", device=device)
    labels, preds, probs = get_predictions(model, val_loader, device)
    print_classification_report(labels, preds)

    print("\n" + "="*50)
    print("THRESHOLD TUNING")
    print("="*50)
    best_thresholds = find_best_thresholds(labels, probs)

    print("\n" + "="*50)
    print("CLASSIFICATION REPORT AT TUNED THRESHOLDS")
    print("="*50)

    tuned_preds = np.zeros_like(probs)
    defect_names = ["defect_1", "defect_2", "defect_3", "defect_4"]
    for i, name in enumerate(defect_names):
        tuned_preds[:, i] = (probs[:, i] >= best_thresholds[name]).astype(float)

    print_classification_report(labels, tuned_preds)