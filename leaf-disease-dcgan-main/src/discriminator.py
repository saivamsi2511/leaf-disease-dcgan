import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, channels=3, features_d=64):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            # Input: (batch, 3, 64, 64)

            # (batch, 64, 32, 32)
            nn.Conv2d(channels, features_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # (batch, 128, 16, 16)
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (batch, 256, 8, 8)
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (batch, 512, 4, 4)
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (batch, 1, 1, 1)
            nn.Conv2d(features_d * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)
