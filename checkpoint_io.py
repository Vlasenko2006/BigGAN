#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 17:46:41 2025

@author: andrey

"""


import os
import torch



def checkpoint_io(path, generator, discriminator, g_optimizer, d_optimizer, device="cpu", mode = "load", epoch = "0"):
    checkpoint_path = path + "checkpoint_epoch_" + str(epoch) + ".pth"
 #   checkpoint_path = os.path.join(path, f"checkpoint_epoch_{load_epoch}.pth")
    if mode == "load":
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found at {checkpoint_path}")
            exit()
    
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
        generator.load_state_dict(checkpoint['generator_state_dict'])
        discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['optimizer_G_state_dict'])
        d_optimizer.load_state_dict(checkpoint['optimizer_D_state_dict'])
        
        epoch = checkpoint['epoch']
        
        print(f"Checkpoint loaded from {checkpoint_path}, resuming from epoch {epoch}")
    elif mode == "save":
        print("Check path before save")
        if(os.path.exists(path)):
            print("saving the model")
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': g_optimizer.state_dict(),
                'optimizer_D_state_dict': d_optimizer.state_dict(),
            }, os.path.join(path, f"checkpoint_epoch_{epoch+1}.pth"))

    
