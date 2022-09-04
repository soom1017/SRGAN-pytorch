import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.prelu = nn.PReLU(num_parameters=64)

    def forward(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.prelu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = torch.add(x, input)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixelShuffler = nn.PixelShuffle(2)
        self.prelu = nn.PReLU(num_parameters=64)

    def forward(self, x):
        x = self.conv(x)
        x = self.pixelShuffler(x)
        x = self.prelu(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=4)
        self.prelu = nn.PReLU(num_parameters=64)
        self.residual_layer = self.make_layer(ResidualBlock, 5)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.upsample_layer = self.make_layer(UpSampleBlock, 2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=9, stride=1, padding=4)
        
    
    def make_layer(self, block, num_block):
        layers = []
        for _ in range(num_block):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu(x)

        res = x
        x = self.residual_layer(x)
        x = self.conv2(x)
        x = self.bn(x)
        out = res + x

        out = self.upsample_layer(out)
        out = self.conv3(out)
        return out

class BaseBlock_D(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.leakyRelu(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.leakyRelu = nn.LeakyReLU(0.2, inplace=True)
        self.blocks = nn.Sequential(
            BaseBlock_D(64, 64, 2),
            BaseBlock_D(64, 128, 1),
            BaseBlock_D(128, 128, 2),
            BaseBlock_D(128, 256, 1),
            BaseBlock_D(256, 256, 2),
            BaseBlock_D(256, 512, 1),
            BaseBlock_D(512, 512, 2)
        )

        # SRGAN implementation from scratch: https://www.youtube.com/watch?v=7FO9qDOhRCc
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

        # Not using sigmoid, instead use BCEWithLogits (Sigmoid + BCE)
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = self.conv(x)
        x = self.leakyRelu(x)
        x = self.blocks(x)

        x = self.classifier(x)
        return x
    