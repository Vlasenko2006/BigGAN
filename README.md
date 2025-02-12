# About 

The FAsT Generative Adversarial neural Network (FATGAN) has a similar background and installation procedure as [ShortGAN neural network](https://github.com/Vlasenko2006/ShortGAN). We omit these details here. It has advanced to [ShortGAN neural network](https://github.com/Vlasenko2006/ShortGAN) Discriminator and Generator with `Conv2DTranspose` and `spectral normalization` layers, overpassed by residual layers. The network has a user-specified depth and can generate colored images of any size.

**Advantages:** Trains faster than other GANs, with less computational resources. For instance, the demonstrative images (see below) of 136 x 206 pixels size FATGAN were generated on a single GPU with 20 GB memory.

**Generated Images**

![Training](https://github.com/Vlasenko2006/FatGAN/blob/main/Sample%20of%20generated%20images.jpg)


# GAN Model for Coloured Image Generation  

This repository contains a **Generative Adversarial Network (GAN)** implementation designed to generate **136x204 color images** using **PyTorch**. It leverages **spectral normalization**, **residual blocks**, and **label smoothing** to enhance stability and image quality.  

-
