from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np 
import torch
from PIL import Image, ImageSequence
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):
    def __init__(self, image_paths, image_folder, labels, transform=None):
        self.paths = image_paths
        self.image_folder = image_folder
        self.labels = labels
        self.transform = transform
    def __getitem__(self, idx):
        im_name = self.paths.iloc[idx]
        im = Image.open(self.image_folder+im_name)
        if(im_name.split('.')[-1]=='gif'):
            image = ImageSequence.Iterator(im)[0].convert('RGB')
        else:
            image = im.convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return torch.FloatTensor(image), torch.FloatTensor(label)
    def __len__(self):
        return len(self.paths)