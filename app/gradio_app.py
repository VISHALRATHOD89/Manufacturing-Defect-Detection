import gradio as gr
import torch
import numpy as np
from PIL import Image

from src.gradcam import load_model_for_gradcam, generate_gradcam
from src.transforms import val_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model_for_gradcam(device=device)

DEFECT_NAMES = ["Defect 1", "Defect 2", "Defect 3", "Defect 4"]

def predict_defect(image):
    if image is None:
        return None, "No image uploaded"

    pil_image = Image.fromarray(image).convert("RGB")

    input_tensor = val_transforms(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output)[0].cpu().numpy()

    results_text = ""
    for name, prob in zip(DEFECT_NAMES, probs):
        risk = "🔴 HIGH" if prob >= 0.5 else "🟡 LOW" if prob >= 0.2 else "🟢 MINIMAL"
        results_text += f"{name}: {prob:.1%} — {risk}\n"

    top_class_idx = int(np.argmax(probs))

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        pil_image.save(tmp.name)
        visualization, _ = generate_gradcam(model, tmp.name, top_class_idx, device=device)

    return visualization, results_text

demo = gr.Interface(
    fn=predict_defect,
    inputs=gr.Image(label="Upload Steel Sheet Image"),
    outputs=[
        gr.Image(label="Grad-CAM Explanation"),
        gr.Textbox(label="Defect Predictions", lines=5),
    ],
    title="🏭 Steel Surface Defect Detection",
    description=(
        "Upload a steel sheet image to detect surface defects using a ResNet50 model "
        "fine-tuned on Severstal's real production-line data. The Grad-CAM overlay shows "
        "which region most influenced the model's top prediction."
    ),
    examples=[
        "data/severstal/train_images/3a153274d.jpg",
        "data/severstal/train_images/50308589f.jpg",
    ],
)

if __name__ == "__main__":
    demo.launch()