import os
import torch
from torch.utils.data import Dataset
from PIL import Image



# Image dataset loader
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(".jpg") or f.endswith(".jpeg")]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(1.0)  # Label 1 for real images

