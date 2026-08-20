import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.train import build_model
from src.transforms import val_transforms


def load_model_for_gradcam(weights_path="models/defect_model_weighted.pth", device="cpu"):
    model = build_model(num_classes=4)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def generate_gradcam(model, image_path, target_class, device="cpu"):
    original_image = Image.open(image_path).convert("RGB")
    original_resized = original_image.resize((224, 224))
    rgb_img_float = np.array(original_resized).astype(np.float32) / 255.0

    input_tensor = val_transforms(original_image).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    targets = [ClassifierOutputTarget(target_class)]

    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]

    visualization = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)

    return visualization, grayscale_cam


if __name__ == "__main__":
    import os

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model_for_gradcam(device=device)

    examples = [
        ("3a153274d.jpg", 0, "defect_1 (real defect)"),
        ("50308589f.jpg", 0, "defect_1 (clean image, no real defect)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (image_id, target_class, label) in zip(axes, examples):
        image_path = os.path.join("data/severstal/train_images", image_id)
        visualization, heatmap = generate_gradcam(model, image_path, target_class, device=device)

        ax.imshow(visualization)
        ax.set_title(f"{image_id}\n{label}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("models/gradcam_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved -> models/gradcam_comparison.png")

    