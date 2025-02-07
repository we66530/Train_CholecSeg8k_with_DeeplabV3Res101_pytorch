# Train_CholecSeg8k_with_DeeplabV3Res101_pytorch
## 🚀 Update
02/07: Upload the pre-trained model ```epoch_39_loss_0.0630.pth``` with average IOU score = 0.9289
## Auto-segmentation in laparoscopic cholecystectomy images
This project utilizes the DeepLabV3_Res101 model, provided by the official PyTorch library, to train on the CholecSeg8k dataset, which consists of 8,080 image-mask pairs.
* The original color mapping provided by CholecSeg8k dataset is wrong, the correct color mapping is like following:
```COLOR_MAP = {
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
}```

![Model will choose the picture with conversation fit your input](./demo.jpg)
