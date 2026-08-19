import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

# 5 Finger Millet Leaf Condition Classes
CLASSES = [
    "Healthy Leaf",
    "Leaf Blast (Pyricularia oryzae)",
    "Cercospora Leaf Spot",
    "Helminthosporium Blight",
    "Finger Millet Smut"
]

CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(CLASSES)}

def get_transforms(img_size=224):
    """
    Returns train and validation/test torchvision data augmentation pipelines.
    """
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def generate_synthetic_leaf(class_idx, size=(224, 224)):
    """
    Generates a synthetic finger millet leaf image with characteristic disease symptoms
    for testing, demonstration, and pipeline verification.
    """
    # Create green leaf background
    img = Image.new("RGB", size, (34, 139, 34))
    draw = ImageDraw.Draw(img)
    
    # Draw leaf vein structure
    draw.line([(size[0]//2, 0), (size[0]//2, size[1])], fill=(50, 205, 50), width=4)
    for y in range(20, size[1], 30):
        draw.line([(size[0]//2, y), (size[0]//2 - 60, y - 20)], fill=(46, 139, 87), width=2)
        draw.line([(size[0]//2, y), (size[0]//2 + 60, y - 20)], fill=(46, 139, 87), width=2)
        
    # Inject disease specific patterns
    if class_idx == 1:  # Leaf Blast (Spindle shaped brown/grey lesions)
        for cx, cy in [(80, 70), (140, 130), (100, 180)]:
            draw.ellipse([cx-25, cy-10, cx+25, cy+10], fill=(139, 69, 19), outline=(80, 40, 10))
            draw.ellipse([cx-10, cy-4, cx+10, cy+4], fill=(211, 211, 211))
            
    elif class_idx == 2:  # Cercospora Leaf Spot (Circular brown spots)
        for cx, cy in [(60, 50), (160, 90), (90, 140), (130, 190)]:
            draw.ellipse([cx-12, cy-12, cx+12, cy+12], fill=(160, 82, 45), outline=(100, 50, 25))
            draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(245, 245, 220))
            
    elif class_idx == 3:  # Helminthosporium Blight (Dark streaks)
        for y in [40, 110, 170]:
            draw.rectangle([50, y, 170, y+15], fill=(101, 67, 33), outline=(50, 30, 15))
            
    elif class_idx == 4:  # Smut (Black seed galls)
        for cx, cy in [(112, 80), (112, 120), (112, 160)]:
            draw.ellipse([cx-20, cy-20, cx+20, cy+20], fill=(20, 20, 20), outline=(0, 0, 0))
            
    return img

class FingerMilletDataset(Dataset):
    """
    PyTorch Dataset for Finger Millet Disease Detection.
    Loads real images from directory or generates synthetic datasets for testing.
    """
    def __init__(self, root_dir=None, transform=None, num_synthetic_per_class=50):
        self.transform = transform
        self.samples = []
        
        if root_dir and os.path.exists(root_dir):
            for cls_name in CLASSES:
                cls_dir = os.path.join(root_dir, cls_name)
                if os.path.exists(cls_dir):
                    for fname in os.listdir(cls_dir):
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.samples.append((os.path.join(cls_dir, fname), CLASS_TO_IDX[cls_name]))
                            
        # If no local images found, generate in-memory synthetic samples for pipeline validation
        if len(self.samples) == 0:
            print(f"[FingerMilletDataset] Generating {num_synthetic_per_class * len(CLASSES)} synthetic samples for validation...")
            for cls_idx in range(len(CLASSES)):
                for _ in range(num_synthetic_per_class):
                    self.samples.append((cls_idx, cls_idx)) # (class_idx, label)
                    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        item, label = self.samples[idx]
        
        if isinstance(item, str):
            image = Image.open(item).convert('RGB')
        else:
            # Generate synthetic image on the fly
            image = generate_synthetic_leaf(label)
            
        if self.transform:
            image = self.transform(image)
            
        return image, label

if __name__ == "__main__":
    _, val_tf = get_transforms()
    ds = FingerMilletDataset(transform=val_tf, num_synthetic_per_class=10)
    print(f"Dataset initialized with {len(ds)} items across {len(CLASSES)} classes.")
    img, lbl = ds[0]
    print(f"Sample tensor shape: {img.shape}, Label: {lbl} ({CLASSES[lbl]})")
