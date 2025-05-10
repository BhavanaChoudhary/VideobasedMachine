import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

# Load pre-trained ResNet model for image classification
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# ImageNet class labels
# Download from https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
imagenet_path = os.path.join(project_root, "imagenet_classes.txt")
with open(imagenet_path) as f:
    imagenet_classes = [line.strip() for line in f.readlines()]

# Define image preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def identify_machine(image: Image.Image) -> str:
    """
    Use pre-trained ResNet model to identify the machine from the image.
    """
    # Convert image to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

    with torch.no_grad():
        output = resnet_model(input_batch)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    predicted_label = imagenet_classes[top_catid]
    return f"{predicted_label} (confidence: {top_prob.item():.2f})"

import cv2
import numpy as np

def assess_condition(image: Image.Image) -> str:
    """
    Simple heuristic to assess condition based on image sharpness (variance of Laplacian).
    """
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 100:
        return "Poor condition (blurry or low detail)"
    elif variance < 300:
        return "Fair condition (some blur)"
    else:
        return "Good condition (sharp image)"

def detect_rust(image: Image.Image):
    """
    Simple heuristic to detect rust by identifying reddish-brown color regions.
    Returns a tuple of (rust_message, rust_mask)
    """
    # Convert PIL Image to OpenCV format
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    # Define rust color range in HSV
    lower_rust = np.array([5, 50, 50])
    upper_rust = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower_rust, upper_rust)
    rust_ratio = np.sum(mask > 0) / (mask.shape[0] * mask.shape[1])

    if rust_ratio > 0.05:
        message = f"Significant rust detected ({rust_ratio*100:.1f}%)"
    elif rust_ratio > 0.01:
        message = f"Minor rust detected ({rust_ratio*100:.1f}%)"
    else:
        message = "No significant rust detected"

    return message, mask

def suggest_repurpose(condition: str) -> str:
    """
    Suggest repurposing options based on condition with multiple sustainability points.
    """
    points = []
    if "end of life" in condition.lower():
        points.append("Repurpose as industrial art or scrap metal recycling")
        points.append("Use parts for educational or training purposes")
        points.append("Donate to community workshops or makerspaces")
    else:
        points.append("Continue regular maintenance and operation")
        points.append("Implement energy-efficient upgrades")
        points.append("Schedule periodic inspections to extend lifespan")
    return "Sustainability Repurpose Suggestions:\n- " + "\n- ".join(points)
