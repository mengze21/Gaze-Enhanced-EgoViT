# -------------------------------------------------------
# GazeNet module
# A small CNN to extract the features of cropped frames
# They have a size of 35x35 px
# Use seed 42 to ensure reproducibility
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
        self.linear1 = nn.Linear(512, 2048)

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


class GazeNet_const_weight(nn.Module):
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
        self.linear1 = nn.Linear(512, 2048)

        # self.BatcNorm4 = nn.InstanceNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


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


# test
# def set_seed(seed):
#     """设置固定的随机种子以确保可重复性。"""
#     torch.manual_seed(seed)  # 为CPU设置种子
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)  # 为所有GPU设置种子
#         torch.cuda.manual_seed_all(seed)  # 如果使用多GPU，为所有GPU设置种子
#     torch.backends.cudnn.deterministic = True  # 保证每次返回的卷积算法是确定的，如果不设置的话，可能会因为选择的算法不同而导致每次网络前馈时结果略有差异
#     torch.backends.cudnn.benchmark = False  # 当网络结构不变时，关闭优化，提高可复现性

# set_seed(42)  # 你可以选择任何你喜欢的整数作为种子
# gaze_feature = GazeNet()

# x = torch.rand(8, 3, 35, 35)
# # print(f"x is {x}")
# # print(gaze_feature(x).shape)
# y = gaze_feature(x)
# print(y.shape)
