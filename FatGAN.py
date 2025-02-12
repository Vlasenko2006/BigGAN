#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 18:04:59 2025

@author: andrey

This is a generative adversarial neural network. It creates a meaningful images from noise.
It saves the intermediate results and checkpoints while training. You specify the names of the
folders where you want to see the result/checkpoints and it creates it for you (if you did not
create them in advance). 



=======================================  DISCLAMER ===========================================

The code provided here under general MIT licence 3.0 in a state *as is*. No warranties that it
runs safely and any possible damages to your equipment. You run it under your own risk!

!====================================END OF DISCLAMER ========================================

"""


import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from checkpoint_io import checkpoint_io
from my_trainer import train_gan, pretrain_generator
from model import Generator, Discriminator
from ImageDataset import ImageDataset




# Size of generated image
nx_im = 136
ny_im = 204
image_size = (nx_im, ny_im) 


# Note! if images differ a lott (color/featuers) 
# the noise_dim should be of the same order or 
# one order less than the number of images 
# in the training set
noise_dim = int(nx_im * 5) + 5  


# Model hyperparameters

nx = 60
ny = 70
first_layer_hidden_dim = 128 
num_channels = 3 # num_channels is the number of colours in your image (apparently, it is 3 or 1)
kernel_size = 4 # kernel_size in convolution layers
num_repeats=3 # number of repeated conv blocks.


# Training settings
batch_size = 10
dropout = 0.2
learning_rate = 0.0001 * 0.2 
pretrain_epochs = 8
gan_epochs = 300000
smoothing = 0.04  # rate of underconfidence for discriminator
accumulation_steps= 8 * 2


# IO parameters
path = "specify a folder where you want to save checkpoints"
path_to_saved_checkpoints = " path to the saved chcekpoints, if you do not start from scratch"
path_to_results = "specify a folder where you want to save results"
path_to_images = "../../../cats/"
start_epoch = 0 # set to 0 if you start training from scratch, or to the epoch if you continue it from the saved checkpoint 
start_from_scratch = True
save_after_nepochs = 50


os.makedirs(path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize(image_size),  # Fix here
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])

# Load custom dataset
dataset = ImageDataset(root_dir=path_to_images, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(noise_dim, 
                      nx_in = nx, 
                      ny_in = ny,
                      nx_out=nx_im,
                      ny_out=ny_im, 
                      hidden_dim = first_layer_hidden_dim,
                      num_channels=num_channels, 
                      kernel_size = kernel_size,
                      dropout = dropout,
                      num_repeats=num_repeats
                      ).to(device)


discriminator = Discriminator(image_size = image_size,
                              num_channels = num_channels
                              ).to(device)

# Optimizers and Loss
g_pretrain_optimizer = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
pretrain_criterion = nn.SmoothL1Loss()
#gan_criterion = nn.BCELoss()  # Loss for GAN training

# Pretrain the generator
print("Pretraining the generator...")
if start_from_scratch:
    start_epoch = 0
    print("Pretraining the generator...")
    pretrain_generator(generator, 
                       dataloader,
                       g_pretrain_optimizer,
                       pretrain_criterion,
                       device, 
                       noise_dim,
                       pretrain_epochs)
else:
    checkpoint_io(path_to_saved_checkpoints,
                  generator, 
                  discriminator,
                  g_optimizer, 
                  d_optimizer, 
                  device="cpu", 
                  mode = "load", 
                  epoch = start_epoch
                  )
# Train the GAN
print("Starting GAN training...")
train_gan(generator,
          discriminator,
          g_optimizer, 
          d_optimizer,
         # gan_criterion,
          dataloader,
          device, 
          noise_dim,
          gan_epochs,
          path=path, 
          smoothing= smoothing,
          path_to_results=path_to_results, 
          accumulation_steps=accumulation_steps,
          save_after_nepochs= save_after_nepochs,
          start_epoch = start_epoch
          )
