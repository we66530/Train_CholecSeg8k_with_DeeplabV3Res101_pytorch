import torch
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
import numpy as np
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import os

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

def load_model(checkpoint_path, device):
    model = models.deeplabv3_resnet101(weights=models.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = torch.nn.Conv2d(256, len(COLOR_MAP), kernel_size=1)
    model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0), original_size

def get_colored_mask(output, original_size):
    output = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    mask = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        mask[output == cls] = color
    mask = Image.fromarray(mask).resize(original_size, Image.NEAREST)
    return mask

def perform_inference(model, image_path, device):
    image_tensor, original_size = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)["out"]
        mask = get_colored_mask(output, original_size)
    return mask

def select_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if image_path:
        img = Image.open(image_path).resize((300, 300))
        img = ImageTk.PhotoImage(img)
        img_label.config(image=img)
        img_label.image = img
        process_button.config(state=tk.NORMAL)

def select_model():
    global model_path, model
    model_path = filedialog.askopenfilename(filetypes=[("Model Files", "*.pth")])
    if model_path:
        model = load_model(model_path, device)
        print("Model Loaded Successfully")

def process_image():
    if model and image_path:
        mask = perform_inference(model, image_path, device)
        mask.save("output.png")
        mask_cv = cv2.imread("output.png")
        cv2.imshow("Segmentation Result", mask_cv)
        cv2.waitKey(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
image_path = ""
model_path = ""

root = tk.Tk()
root.title("Segmentation Tool")
root.geometry("500x400")

Label(root, text="Select Image").pack()
img_label = Label(root)
img_label.pack()
Button(root, text="Browse Image", command=select_image).pack()

Label(root, text="Select Model (.pth)").pack()
Button(root, text="Browse Model", command=select_model).pack()

process_button = Button(root, text="Process Image", state=tk.DISABLED, command=process_image)
process_button.pack()

root.mainloop()
