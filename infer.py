import os
import torch
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
import numpy as np
from PIL import Image
import argparse

# Color map for segmentation masks
COLOR_MAP = {
    0: (127, 127, 127),  # Black Background
    1: (210, 140, 140),  # Abdominal Wall
    2: (255, 114, 114),  # Liver
    3: (231, 70, 156),  # Gastrointestinal Tract
    4: (186, 183, 75),  # Fat
    5: (170, 255, 0),  # Grasper
    6: (255, 85, 0),  # Connective Tissue
    7: (255, 0, 0),  # Blood
    8: (255, 255, 0),  # Cystic Duct
    9: (169, 255, 184),  # L-hook Electrocautery
    10: (255, 160, 165), # Gallbladder
    11: (0, 50, 128), # Hepatic Vein
    12: (111, 74, 0),    # Liver Ligament
}

# Load the trained model
def load_model(checkpoint_path, device):
    model = models.deeplabv3_resnet101(weights=models.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = torch.nn.Conv2d(256, len(COLOR_MAP), kernel_size=1)  # Adjust output channels
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)  # Explicitly set weights_only=True
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode
    print(f"Loaded model from {checkpoint_path}")

    return model

# Preprocess the input image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Store original size (width, height)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), original_size  # Add batch dimension

# Convert model output to a color-mapped mask
def get_colored_mask(output, original_size):
    output = torch.argmax(output, dim=1).squeeze().cpu().numpy()  # Get class indices
    height, width = output.shape
    mask = np.zeros((height, width, 3), dtype=np.uint8)

    for cls, color in COLOR_MAP.items():
        mask[output == cls] = color  # Assign corresponding color

    mask = Image.fromarray(mask)  # Convert to PIL Image
    mask = mask.resize(original_size, Image.NEAREST)  # Resize to original image size

    return mask

# Perform inference
def perform_inference(model, image_path, output_path, device):
    image_tensor, original_size = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)["out"]  # Forward pass
        mask = get_colored_mask(output, original_size)  # Convert to color mask

    mask.save(output_path)
    print(f"Saved segmented mask to {output_path}")

# Command-line interface
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Perform segmentation using a trained model.")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("checkpoint_path", type=str, help="Path to the trained .pth checkpoint")
    parser.add_argument("output_path", type=str, help="Path to save the output mask")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint_path, device)
    perform_inference(model, args.image_path, args.output_path, device)
