from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np 
import torch
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, image_paths, image_folder, labels, transform=None):
        self.paths = image_paths
        self.image_folder = image_folder
        self.labels = labels
        self.transform = transform
    def __getitem__(self, idx):
        im_name = self.paths.iloc[idx]
        image = np.array(Image.open(self.image_folder+im_name))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
    def __len__(self):
        return len(self.paths)