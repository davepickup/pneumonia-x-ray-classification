"""Image preprocessing functions"""

import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler


class XrayDataset(Dataset):
    """Pneumonia Xray Image Dataset"""

    label_map = {"normal": 0, "pneumonia": 1}

    def __init__(self, root_dir, transform=None):
        """
        Parameters
        ----------
            root_dir : str
                Directory with all the images.
            transform : Optional[Callable]
            Optional transform to be applied on samples.
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


def create_weighted_sampler(dataset):
    """Function to calculate class imbalance in dataset

    Parameters
    ----------
    dataset : torch.util.data.Dataset
        Input Pytorch dataset object

    Returns
    ---------
    sampler : torch.util.data.WeightedRandomSampler
        Sampler object to pass to Dataloader object"""
    targets = [
        XrayDataset.label_map[file.split(os.sep)[-2].lower()] for file in dataset.files
    ]
    class_counts = np.bincount(targets)
    class_weights = 1.0 / class_counts
    weights = [class_weights[label] for label in targets]
    sampler = WeightedRandomSampler(weights, len(weights))
    return sampler
