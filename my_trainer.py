#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 11:14:06 2025

@author: andrey
"""

import torch
from tqdm import tqdm
import gc
import torch.cuda.amp as amp  # Import for mixed precision
from checkpoint_io import checkpoint_io
from plot_generated_images import plot_generated_images


def train_gan(generator,
              discriminator,
              g_optimizer, 
              d_optimizer, 
              dataloader, 
              device, 
              noise_dim, 
              epochs=100,
              path=".",
              smoothing=0.1,
              path_to_results="results/", 
              accumulation_steps=4,
              save_after_nepochs= 100,
              start_epoch = 0):
    
    generator.train()
    discriminator.train()

    scaler = amp.GradScaler()  # Initialize mixed precision scaler
    criterion = torch.nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss instead of BCELoss

    for epoch in range(start_epoch,epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0

        # Free GPU memory before each epoch
        gc.collect()
        torch.cuda.empty_cache()

        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for i, (real_images, _) in enumerate(progress_bar):
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Add Gaussian noise to real images
                noisy_real_images = real_images + 0.05 * torch.randn_like(real_images)

                # Generate fake images
                noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

                with amp.autocast():  # Enable mixed precision
                    fake_images = generator(noise)

                    # Train Discriminator
                    real_labels = torch.ones(batch_size, 1).to(device) * (1 - smoothing)  # Label smoothing
                    fake_labels = torch.zeros(batch_size, 1).to(device)

                    d_real_logits = discriminator(noisy_real_images)
                    d_fake_logits = discriminator(fake_images.detach())

                    d_loss_real = criterion(d_real_logits, real_labels)
                    d_loss_fake = criterion(d_fake_logits, fake_labels)
                    d_loss = (d_loss_real + d_loss_fake) / accumulation_steps  # Scale loss for accumulation

                # Accumulate gradients
                scaler.scale(d_loss).backward()

                if (i + 1) % accumulation_steps == 0:  # Update weights every `accumulation_steps`
                    scaler.step(d_optimizer)
                    scaler.update()
                    d_optimizer.zero_grad()

                # Train Generator (multiple times per epoch)
                for _ in range(2):  # Train the generator more frequently
                    noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

                    with amp.autocast():
                        fake_images = generator(noise)
                        g_loss = criterion(discriminator(fake_images), real_labels) / accumulation_steps  # Scale loss

                    scaler.scale(g_loss).backward()

                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(g_optimizer)
                        scaler.update()
                        g_optimizer.zero_grad()

                # Update progress bar with losses
                epoch_loss_d += d_loss.item()
                epoch_loss_g += g_loss.item()
                progress_bar.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())

        # Log epoch-level progress
        print(f"Epoch {epoch+1}/{epochs} - D Loss: {epoch_loss_d/len(dataloader):.8f}, G Loss: {epoch_loss_g/len(dataloader):.4f}")

        # Save generated images and models
        if (epoch + 1) % save_after_nepochs == 0:
            print(" ==== saving images ==== ")
            plot_generated_images(noise_dim, generator, epoch + 1, device = device, path_to_results=path_to_results)
            checkpoint_io(path,
                          generator, 
                          discriminator,
                          g_optimizer, 
                          d_optimizer, 
                          device="cpu", 
                          mode = "save", 
                          epoch = epoch)

def pretrain_generator(generator, dataloader, optimizer, criterion, device, noise_dim, pretrain_epochs=20):
    generator.train()
    scaler = amp.GradScaler()  # Initialize mixed precision scaler

    for epoch in range(pretrain_epochs):
        epoch_loss = 0.0

        # Free GPU memory before each epoch
        gc.collect()
        torch.cuda.empty_cache()

        with tqdm(dataloader, desc=f"Pretraining Generator Epoch {epoch+1}/{pretrain_epochs}", unit="batch") as progress_bar:
            for real_images, _ in progress_bar:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)

                # Generate random noise
                noise = torch.randn(batch_size, noise_dim).to(device)

                with amp.autocast():  # Enable mixed precision
                    # Generate fake images
                    fake_images = generator(noise)

                    # Pretraining loss (Smooth L1 between fake and real images)
                    loss = criterion(fake_images, real_images)

                # Scale loss and backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                progress_bar.set_postfix(Loss=loss.item())

        print(f"Pretraining Generator Epoch {epoch+1}/{pretrain_epochs} - Loss: {epoch_loss/len(dataloader):.6f}")

