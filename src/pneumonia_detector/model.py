"""CNN pneumonia detector model"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PneumoniaClassifier(nn.Module):
    def __init__(self):
        super(PneumoniaClassifier, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # an affine operation: y = Wx + b
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(65536, 512)  # 5*5 from image dimension
        self.fc2 = nn.Linear(512, 2)
        # self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        # print(x.shape)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc2(x)
        return x
