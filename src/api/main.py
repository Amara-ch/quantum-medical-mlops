"""
FastAPI app to serve the Brain Tumor detection model.
Upload an MRI image -> get a prediction (tumor / no tumor) back.
"""

import io

import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image

from src.data.data_loader import get_transforms
from src.models.hybrid_model import HybridModel

# ----- Setup -----
app = FastAPI(title="Brain Tumor Detection API", version="1.0")

# Class names: index 0 = "no" (healthy), index 1 = "yes" (tumor)
CLASS_NAMES = ["no_tumor", "tumor"]

# Device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model ONCE when the app starts
model = HybridModel(n_classes=2)
model.load_state_dict(torch.load("brain_tumor_model.pth", map_location=device))
model.to(device)
model.eval()  # set to evaluation mode (no training)

# Image preprocessing (same as during training)
transform = get_transforms()


@app.get("/")
def home():
    """Simple health-check endpoint."""
    return {
        "message": "Brain Tumor Detection API is running! 🧠⚛️",
        "usage": "POST an MRI image to /predict",
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Takes an uploaded MRI image and returns a tumor prediction.
    """
    # 1) Read the uploaded image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    # 2) Preprocess it (resize, normalize, etc.)
    input_tensor = transform(image).unsqueeze(0).to(device)  # add batch dimension

    # 3) Run the model (no gradients needed for prediction)
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        predicted_idx = int(probabilities.argmax())
        confidence = float(probabilities[predicted_idx])

    # 4) Return the result
    return {
        "filename": file.filename,
        "prediction": CLASS_NAMES[predicted_idx],
        "confidence": round(confidence, 4),
        "probabilities": {
            "no_tumor": round(float(probabilities[0]), 4),
            "tumor": round(float(probabilities[1]), 4),
        },
    }