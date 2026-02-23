import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=100, channels=3, features_g=64):
        super(Generator, self).__init__()

        self.net = nn.Sequential(
            # Input: Z -> (batch, 100, 1, 1)

            # (batch, 512, 4, 4)
            nn.ConvTranspose2d(latent_dim, features_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features_g * 8),
            nn.ReLU(True),

            # (batch, 256, 8, 8)
            nn.ConvTranspose2d(features_g * 8, features_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 4),
            nn.ReLU(True),

            # (batch, 128, 16, 16)
            nn.ConvTranspose2d(features_g * 4, features_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g * 2),
            nn.ReLU(True),

            # (batch, 64, 32, 32)
            nn.ConvTranspose2d(features_g * 2, features_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_g),
            nn.ReLU(True),

            # (batch, 3, 64, 64)
            nn.ConvTranspose2d(features_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)
