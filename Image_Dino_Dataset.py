import os
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from Dino_extractor import DINOExtractor

class ImageDinoFeatureDataset(Dataset):
    def __init__(self, root_dir, target_layer: int, total_layers: int = 25, image_size = 224):
        self.target_layer = target_layer
        self.cond_layer = target_layer + 1 if target_layer < total_layers - 1 else None
    
        self.image_folder = ImageFolder(root=root_dir, transform=self._basic_transform(image_size))
        self.dino = DINOExtractor(device=torch.device("cuda"))

    def _basic_transform(self, image_size):
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        image, label = self.image_folder[idx]
        with torch.no_grad():
            features = self.dino(image.unsqueeze(0))

        x_target = features[self.target_layer].squeeze(0)

        if self.cond_layer is not None:
            x_cond = features[self.cond_layer].squeeze(0)
        else:
            x_cond = torch.zeros_like(x_target)
        
        return{
            "x_target": x_target,
            "x_cond": x_cond,
            "label": label,
            "id": str(idx)
        }