import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import sys
from collections import OrderedDict
import os

# -------- Settings -------- #
NUM_CLASSES = 5  # Update if you changed this
MODEL_PATH = "saved_models/baldsight_best_20250717_164421.pth"
CLASS_NAMES = ['stage1', 'stage2', 'stage3', 'stage4', 'stage5']

# -------- Preprocessing -------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -------- Model Definition -------- #
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Load checkpoint with "base_model." prefix stripped
checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
new_state_dict = OrderedDict()
for k, v in checkpoint.items():
    new_k = k.replace("base_model.", "")
    new_state_dict[new_k] = v

model.load_state_dict(new_state_dict)
model.eval()

# -------- Predict Function -------- #
def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0)  # Add batch dim

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        predicted_stage = CLASS_NAMES[predicted.item()]

    print(f"Predicted Balding Stage: {predicted_stage}")

    # Show image
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_stage}")
    plt.axis('off')
    plt.show()

# -------- CLI Entry Point -------- #
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        predict_image(sys.argv[1])

