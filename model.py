import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "best_densenet121_pneumonia.pth"
)

IMAGE_SIZE = 224

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    model = models.densenet121(weights=None)

    num_features = model.classifier.in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_features, 1)
    )

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

print("=" * 70)
print("LOADING PNEUMONIA DETECTION MODEL")
print("=" * 70)

print(f"Model path : {MODEL_PATH}")
print(f"Device     : {DEVICE}")

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model file not found:\n{MODEL_PATH}"
    )


model = create_model()

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)


# ============================================================
# HANDLE CHECKPOINT FORMAT
# ============================================================

if isinstance(checkpoint, dict):

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

    elif "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    else:

        state_dict = checkpoint

else:

    state_dict = checkpoint


# ============================================================
# LOAD WEIGHTS
# ============================================================

model.load_state_dict(state_dict)

model.to(DEVICE)

model.eval()


print("Model loaded successfully.")
print("=" * 70)


# ============================================================
# IMAGE TRANSFORMATION
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# PREDICTION
# ============================================================

def predict(image: Image.Image):

    image = image.convert("RGB")

    input_tensor = transform(image)

    input_tensor = (
        input_tensor
        .unsqueeze(0)
        .to(DEVICE)
    )


    with torch.no_grad():

        output = model(input_tensor)

        probability = torch.sigmoid(
            output
        ).item()


    pneumonia_probability = probability

    normal_probability = (
        1.0 - probability
    )


    # Deployment threshold
    threshold = 0.5


    if pneumonia_probability >= threshold:

        prediction = "PNEUMONIA"

    else:

        prediction = "NORMAL"


    return {

        "prediction": prediction,

        "pneumonia_probability":
            pneumonia_probability,

        "normal_probability":
            normal_probability,

        "threshold":
            threshold
    }