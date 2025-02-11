#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:10:13 2025

@author: andrey
"""


import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm






class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm2d(in_channels, eps=1e-5),
            nn.ReLU(inplace=True),
            spectral_norm(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)),
            nn.InstanceNorm2d(in_channels, eps=1e-5),
        )

    def forward(self, x):
        return x + self.block(x)  # Skip connection

class Generator(nn.Module):
    def __init__(self, noise_dim, nx_in=60, ny_in=70, nx_out=136, ny_out=204, hidden_dim=128, num_channels=3, kernel_size=4,dropout = 0.0):
        super(Generator, self).__init__()
        self.nx_in = nx_in  
        self.ny_in = ny_in  
        self.nx_out = nx_out  
        self.ny_out = ny_out  
        self.hidden_dim = hidden_dim  
        self.num_channels = num_channels  
        self.kernel_size = kernel_size  

        # Fully connected layer to expand noise
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim, self.hidden_dim * self.nx_in * self.ny_in),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        # Transposed Convolutional Upsampling with Residual Blocks
        self.conv_block = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(self.hidden_dim),
            nn.BatchNorm2d(self.hidden_dim),

            spectral_norm(nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim // 2, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(self.hidden_dim // 2),
            nn.BatchNorm2d(self.hidden_dim // 2),

            spectral_norm(nn.ConvTranspose2d(self.hidden_dim // 2, self.hidden_dim // 4, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            nn.Dropout(dropout),
            ResidualBlock(self.hidden_dim // 4),
            nn.BatchNorm2d(self.hidden_dim // 4),

            spectral_norm(nn.ConvTranspose2d(self.hidden_dim // 4, self.num_channels, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1)),
            nn.Tanh()
        )

        # Final resizing to ensure exact (nx_out, ny_out)
        self.upsample = nn.Upsample(size=(nx_out, ny_out), mode="bilinear", align_corners=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten noise
        x = self.fc1(x)
        x = x.view(-1, self.hidden_dim, self.nx_in, self.ny_in) 
        x = self.conv_block(x)
        x = self.upsample(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, num_channels=3, image_size=(136, 204)):
        super(Discriminator, self).__init__()

        self.conv_layers = nn.Sequential(
            spectral_norm(nn.Conv2d(num_channels, 64, 4, stride=2, padding=1)),  
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(64, 128, 4, stride=2, padding=1)),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(128, 256, 4, stride=2, padding=1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(256, 512, 4, stride=2, padding=1)),  
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(512, 1, 4, stride=2, padding=1)),  # Final conv layer
        )

        # Compute the correct linear layer input size dynamically
        self.feature_dim = self._get_conv_output(image_size)
        self.fc = nn.Linear(self.feature_dim, 1)  # Ensure output is (batch_size, 1)

    def _get_conv_output(self, image_size):
        """Pass a dummy tensor through conv layers to compute the output size."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *image_size)  # (batch, channels, H, W)
            output = self.conv_layers(dummy_input)
            return output.view(1, -1).size(1)  # Flatten and get feature size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)  # Flatten to (batch_size, num_features)
        x = self.fc(x)  # Map to (batch_size, 1)
        return x

