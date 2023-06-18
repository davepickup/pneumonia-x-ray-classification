"""Image preprocessing functions"""

import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class XrayDataset(Dataset):

    label_map = {"normal": 0, "pneumonia": 1}

    def __init__(self, root_dir, transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.files = [
            os.path.normcase(os.path.join(dp, f))
            for dp, _, filenames in os.walk(root_dir)
            for f in filenames
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.files[idx]
        image = Image.open(img_path).convert("RGB")
        label = XrayDataset.label_map[img_path.split(os.sep)[-2].lower()]

        if self.transform:
            image = self.transform(image)

        return image, label
