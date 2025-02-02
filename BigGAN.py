#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 18:49:34 2025

@author: andreyvlasenko
"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image


# Function to plot generated images for colored 136x204 images
# Function to plot generated images for colored 136x204 images
def plot_generated_images(generator, epoch, device, examples=10):
    # Generate noise of size (examples, noise_dim)
    noise = torch.randn(examples, noise_dim).to(device)  # Adjusted to match noise_dim
    generated_images = generator(noise).cpu().detach()

    # Rescale to [0, 1]
    generated_images = (generated_images + 1) / 2  

    # Convert to numpy and adjust dimensions for plotting
    generated_images = generated_images.permute(0, 2, 3, 1).numpy()

    # Plot the generated images
    plt.figure(figsize=(10, 2))
    for i in range(examples):
        plt.subplot(2, examples // 2, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.tight_layout()

    # Save and display the images
    plt.savefig(f"gan3_generated_image_epoch_{epoch}.png")
    plt.show()



import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, nx_in=34, ny_in=51, nx_out=136, ny_out=204, hidden_dim=128, num_channels=3, kernel_size=4):
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
            nn.ReLU()
        )

        # Convolutional upsampling layers
        self.conv_block = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim // 2, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim // 2),

            nn.ConvTranspose2d(self.hidden_dim // 2, self.hidden_dim // 4, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.hidden_dim // 4),

            nn.ConvTranspose2d(self.hidden_dim // 4, self.num_channels, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        )

        # Final resizing to ensure exact (nx_out, ny_out)
        self.upsample = nn.Upsample(size=(nx_out, ny_out), mode="bilinear", align_corners=True)

#    def forward(self, x):
#        x = self.fc1(x)
#        x = x.view(-1, self.hidden_dim, self.nx_in, self.ny_in)  
#        x = self.conv_block(x)

        # Ensure exact output size
#        x = self.upsample(x)

#        batch_size, channels, height, width = x.shape
#        print(f"Output shape: {x.shape}")  # Debugging output

#        return x.view(batch_size, self.num_channels, height, width)  

    def forward(self, x):
        print(f"Input noise shape: {x.shape}")  # Debug print
        x = x.view(x.shape[0], -1)  # Flatten noise to (batch_size, noise_dim)
        print(f"Shape after flattening: {x.shape}")  # Debug print

        x = self.fc1(x)
        print(f"Shape after fc1: {x.shape}")  # Debug print

        x = x.view(-1, self.hidden_dim, self.nx_in, self.ny_in) 
        print(f"Shape after reshaping: {x.shape}")  # Debug print

        x = self.conv_block(x)
        x = self.upsample(x)
        print(f"Final output shape: {x.shape}")  # Debug print
        return x





# Define Discriminator for 136x204 colored images
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(136 * 204 * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


# Image dataset loader
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(".jpg") or f.endswith(".jpeg")]
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Convert to RGB
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(1.0)  # Label 1 for real images



# Label Smoothing for Real Labels
def smooth_labels(labels, smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing


# Train the GAN
def train_gan(generator, discriminator, g_optimizer, d_optimizer, criterion, dataloader, device, noise_dim, epochs=100, path="."):
    generator.train()
    discriminator.train()

    for epoch in range(epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for real_images, _ in progress_bar:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Add Gaussian noise to real images
                noisy_real_images = real_images + 0.05 * torch.randn_like(real_images)

                # Generate fake images
                noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
                fake_images = generator(noise)

                # Train Discriminator
                real_labels = smooth_labels(torch.ones(batch_size, 1).to(device))
                fake_labels = torch.zeros(batch_size, 1).to(device)

                d_loss_real = criterion(discriminator(noisy_real_images), real_labels)
                d_loss_fake = criterion(discriminator(fake_images.detach()), fake_labels)
                d_loss = d_loss_real + d_loss_fake

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

                # Train Generator (multiple times per epoch)
                for _ in range(2):  # Train the generator more frequently
                    noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
                    fake_images = generator(noise)
                    g_loss = criterion(discriminator(fake_images), real_labels)

                    g_optimizer.zero_grad()
                    g_loss.backward()
                    g_optimizer.step()

                # Update progress bar with losses
                epoch_loss_d += d_loss.item()
                epoch_loss_g += g_loss.item()
                progress_bar.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        # Log epoch-level progress
        print(f"Epoch {epoch+1}/{epochs} - D Loss: {epoch_loss_d/len(dataloader):.8f}, G Loss: {epoch_loss_g/len(dataloader):.4f}")

        # Save generated images and models
        if (epoch + 1) % 10 == 0:
            print(" ==== saving images ==== ")
            plot_generated_images(generator, epoch + 1, device)
            torch.save(generator.state_dict(), os.path.join(path, f"generator_epoch_{epoch+1}.pth"))
            torch.save(discriminator.state_dict(), os.path.join(path, f"discriminator_epoch_{epoch+1}.pth"))



def pretrain_generator(generator, dataloader, optimizer, criterion, device, noise_dim, epochs=20):
    generator.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        with tqdm(dataloader, desc=f"Pretraining Generator Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for real_images, _ in progress_bar:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Generate random noise
                noise = torch.randn(batch_size, noise_dim).to(device)

                # Generate fake images
                fake_images = generator(noise)

                # Pretraining loss (Smooth L1 between fake and real images)
                loss = criterion(fake_images, real_images)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                progress_bar.set_postfix(Loss=loss.item())

        print(f"Pretraining Generator Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.6f}")





      
# Main Function Update
#if __name__ == "__main__":
# Hyperparameters
image_size = (136, 204)  # Updated image size
noise_dim = 136 
batch_size = 64
learning_rate = 0.0002
pretrain_epochs = 3
gan_epochs = 30000
path = "./gan_model/"
path_to_mages = "/gpfs/work/vlasenko/07/NN/Images_clean_small/"
os.makedirs(path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transforms
transform = transforms.Compose([
    transforms.Resize(image_size),  # Fix here
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])


# Load custom dataset
dataset = ImageDataset(root_dir =path_to_mages , transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

      
os.makedirs(path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Initialize models
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)


# Optimizers and Loss
g_pretrain_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
pretrain_criterion = nn.SmoothL1Loss()
gan_criterion = nn.BCELoss()  # Loss for GAN training




# Pretrain the generator
print("Pretraining the generator...")
pretrain_generator(generator, dataloader, g_pretrain_optimizer, pretrain_criterion, device, noise_dim, pretrain_epochs)

# Train the GAN
print("Starting GAN training...")
train_gan(generator, discriminator, g_optimizer, d_optimizer, gan_criterion, dataloader, device, noise_dim, gan_epochs, path=path)



