# -------------------------------------------------------
# GazeNet module
# A small CNN to extract the features of cropped frames
# They have a size of 150x150 px
# -------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
import torch


class GazeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, bias=True)
        self.BatcNorm1 = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, bias=True)
        self.BatcNorm2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, bias=True)
        self.BatcNorm3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(36992, 2048)

        # self.BatcNorm4 = nn.InstanceNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BatcNorm1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.BatcNorm2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.BatcNorm3(x)
        x = F.relu(x)
        x = self.pool3(x)

        x = self.flatten(x)
        x = self.linear1(x)
        # x = self.BatcNorm4(x)
        x = F.relu(x)

        return x


# gaze_feature = GazeNet()

# x = torch.rand(1, 3, 150, 150)
# print(gaze_feature(x).shape)
# y = gaze_feature(x)
# print(y)
