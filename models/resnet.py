import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, layers, hiddenSize):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.layers = layers
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, self.hiddenSize, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hiddenSize),
            nn.ReLU(),
        )
        self.backBone = nn.ModuleList(
            [ResBlock(self.hiddenSize) for i in range(layers)]
        )

    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        return x
