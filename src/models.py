"""Autoencoder architectures recovered from the original TCC notebook."""

import torch
import torch.nn as nn


class ConvolutionalAutoencoder(nn.Module):
    """Five-stage convolutional autoencoder: 512x512 -> 16x16 bottleneck."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class ConvolutionalVAE(nn.Module):
    """Convolutional variational autoencoder recovered from the notebook."""

    def __init__(self, latent_dim=256):
        super().__init__()
        self.flattened_size = 256 * 16 * 16
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_size, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, self.flattened_size)
        self.decoder_conv = nn.Sequential(
            nn.Unflatten(1, (256, 16, 16)),
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, 2, 1, 1), nn.Tanh(),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder_conv(x).reshape(x.shape[0], -1)
        mu, log_var = self.fc_mu(h), self.fc_log_var(h)
        z = self.reparameterize(mu, log_var)
        return self.decoder_conv(self.decoder_fc(z)), mu, log_var


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class ResidualAutoencoder(nn.Module):
    """Residual autoencoder used in the recovered experiments."""

    def __init__(self):
        super().__init__()
        self.initial_conv = nn.Conv2d(1, 16, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.res_block1 = ResidualBlock(16, 32, 2)
        self.res_block2 = ResidualBlock(32, 64, 2)
        self.res_block3 = ResidualBlock(64, 128, 2)
        self.res_block4 = ResidualBlock(128, 256, 2)

        self.res_block5 = ResidualBlock(256, 256)
        self.upsample1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.res_block6 = ResidualBlock(128, 128)
        self.upsample2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.res_block7 = ResidualBlock(64, 64)
        self.upsample3 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.res_block8 = ResidualBlock(32, 32)
        self.upsample4 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.final_conv = nn.ConvTranspose2d(16, 1, 2, 2)
        self.tanh = nn.Tanh()

    def encode(self, x):
        x = self.relu(self.bn1(self.initial_conv(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return self.res_block4(x)

    def decode(self, x):
        x = self.upsample1(self.res_block5(x))
        x = self.upsample2(self.res_block6(x))
        x = self.upsample3(self.res_block7(x))
        x = self.upsample4(self.res_block8(x))
        return self.tanh(self.final_conv(x))

    def forward(self, x):
        return self.decode(self.encode(x))
