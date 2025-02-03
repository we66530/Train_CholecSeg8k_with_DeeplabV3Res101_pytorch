import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models.segmentation as models
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.v2 as v2  # New torchvision v2 transforms

# Pretrained checkpoint path
CHECKPOINT_PATH = "x"

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

# Function to convert RGB mask to class indices
def mask_to_class(mask):
    mask = np.array(mask)
    output = np.zeros(mask.shape[:2], dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        output[np.all(mask == color, axis=-1)] = cls
    return output

# Custom Dataset Class
class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        mask = mask_to_class(mask)  # Convert RGB mask to class indices
        mask = torch.tensor(mask, dtype=torch.long)  # Ensure correct dtype

        if self.transform:
            image = self.transform(image)

        return image, mask

# GPU-based Augmentations
def gpu_augment(image, mask):
    if torch.rand(1).item() > 0.5:
        image = torch.flip(image, dims=[2])  # Horizontal flip
        mask = torch.flip(mask, dims=[1])

    if torch.rand(1).item() > 0.5:
        angle = torch.randint(-15, 15, (1,)).item()
        image = v2.functional.rotate(image, angle)
        mask = v2.functional.rotate(mask.unsqueeze(0).float(), angle).squeeze(0).long()

    if torch.rand(1).item() > 0.5:
        image = v2.functional.adjust_brightness(image, 0.8 + torch.rand(1).item() * 0.4)
        image = v2.functional.adjust_contrast(image, 0.8 + torch.rand(1).item() * 0.4)

    return image, mask

# IoU Calculation
def calculate_iou(pred, target, num_classes=13):
    ious = []
    for cls in range(num_classes):
        intersection = torch.logical_and(pred == cls, target == cls).sum().float()
        union = torch.logical_or(pred == cls, target == cls).sum().float()
        iou = intersection / (union + 1e-6)
        
        # Only add the IoU value if it's non-zero (i.e., the class is present in the image)
        if iou > 0:
            ious.append(iou)

    # Return the average IoU over all non-zero IoU classes
    if len(ious) > 0:
        return torch.tensor(ious).mean()
    else:
        return torch.tensor(0.0)  # Return 0 if no valid class IoU is found


# Dice Loss Calculation
def dice_loss(pred, target, num_classes=13, smooth=1e-6):
    total_loss = 0.0
    valid_classes = 0
    
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()

        intersection = torch.sum(pred_cls * target_cls)
        union = torch.sum(pred_cls) + torch.sum(target_cls)

        # Only compute Dice loss for classes that appear in both prediction and target
        if union > 0:  # Ensure class exists in both pred and target
            dice = 1 - (2. * intersection + smooth) / (union + smooth)
            total_loss += dice
            valid_classes += 1

    # Return average Dice loss across all valid classes
    if valid_classes > 0:
        return total_loss / valid_classes
    else:
        return torch.tensor(0.0)  # Return 0 if no valid class is found


# Save Checkpoints
def save_checkpoint(model, optimizer, epoch, loss, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}_loss_{loss:.4f}.pth")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def main():
    # Data Paths
    train_img_dir = r"D:\CholecSeg8k\train_img"
    train_mask_dir = r"D:\CholecSeg8k\mask"

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create Dataset & DataLoader
    train_dataset = SegmentationDataset(train_img_dir, train_mask_dir, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    model = models.deeplabv3_resnet101(weights=models.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[4] = nn.Conv2d(256, len(COLOR_MAP), kernel_size=1)  # Adjust output channels
    model.to(device)

    # Loss & Optimizer
    criterion_ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Load Pretrained Checkpoint
    start_epoch = 0
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Resuming training from checkpoint: Epoch {start_epoch}")

    # Training Loop
    num_epochs = 50
    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch", ncols=100)

        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)

            # Apply GPU Augmentations
            images, masks = gpu_augment(images, masks)

            optimizer.zero_grad()
            outputs = model(images)["out"]
            outputs_softmax = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs_softmax, dim=1)

            # Compute Dice Loss and Cross-Entropy Loss
            dice = dice_loss(preds, masks.float())
            ce_loss = criterion_ce(outputs, masks)

            # Total Loss = Dice Loss + 0.2 * Cross-Entropy Loss
            total_loss = dice + ce_loss

            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()
            running_iou += calculate_iou(preds, masks).item()

            avg_loss = running_loss / (pbar.n + 1)
            avg_iou = running_iou / (pbar.n + 1)

            # Estimated time remaining
            elapsed_time = time.time() - start_time
            estimated_total_time = (elapsed_time / (pbar.n + 1)) * len(pbar)
            remaining_time = estimated_total_time - elapsed_time

            pbar.set_postfix(loss=f"{avg_loss:.4f}", IoU=f"{avg_iou:.4f}", eta=f"{remaining_time:.1f}s")

        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}, Avg IoU: {avg_iou:.4f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 1 == 0:
            save_checkpoint(model, optimizer, epoch, avg_loss)
            print(f"Model saved at epoch {epoch+1}")

    print("Training finished!")

if __name__ == '__main__':
    main()
