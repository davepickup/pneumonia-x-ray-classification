"""CNN pneumonia detector model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PneumoniaClassifier(nn.Module):
    """
    Classifier to predict if Chest X-ray image indicates
    presence of Pneumonia or not.
    """

    def __init__(self, image_size=256):
        """
        Parameters
        ----------
        image_size : int
            Image size to use for training. All images will be converted to square images of
            this dimension.
        """

        super(PneumoniaClassifier, self).__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)

        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * (image_size // 8) * (image_size // 8), 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
