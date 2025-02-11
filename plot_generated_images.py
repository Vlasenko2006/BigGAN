#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:06:13 2025

@author: andrey
"""

import os
import torch
import matplotlib.pyplot as plt




def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

# Function to plot generated images for colored 136x204 images
def plot_generated_images(noise_dim, generator, epoch, device = "cpu", examples=10, path_to_results="./results"):
    check_and_create_directory(path_to_results)
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
    plt.savefig(f"{path_to_results}/gan3_generated_image_epoch_{epoch}.png")
    plt.show()
