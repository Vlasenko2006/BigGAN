# About 

This neural network has similar background and installation procedure as [ShortGAN neural network](https://github.com/Vlasenko2006/ShortGAN), so we will not detail the installation process and its structure. The main difference from the first scheme is that this neural network has a more advanced generator with `Conv2DTranspose` and `spectral normalization` layers, and it can generate colored images. The demonstrative images have 136 x 206 pixels size.

**Images generted in the beginning of the training (100 epochs):**

![Training](https://github.com/Vlasenko2006/FatGAN/blob/main/Sample%20of%20generated%20images.jpg)



# GAN Model for Coloured Image Generation  

This repository contains a **Generative Adversarial Network (GAN)** implementation designed to generate **136x204 color images** using **PyTorch**. It leverages **spectral normalization**, **residual blocks**, and **label smoothing** to enhance stability and image quality.  

---

## Features  
- **Generator** with **transposed convolutions** and **residual blocks** for refined image synthesis.  
- **Discriminator** using **spectral normalization** for stable training.  
- **Mixed Precision Training** (`torch.cuda.amp`) for improved efficiency.  
- **Label Smoothing** for enhanced generalization.  
- **Pretraining Phase** to stabilize the generator before adversarial training.  
- **Progressive Image Saving** every `100` epochs.  

---

## Model Architecture  

### Generator  
- Transforms **random noise** into a `136x204` color image.  
- Uses **fully connected layers** followed by **transposed convolutions**.  
- Includes **residual blocks** to refine feature learning.  

### Discriminator  
- Processes input images using **convolutional layers**.  
- Classifies images as **real or fake** using a **fully connected layer**.  
- Uses **LeakyReLU activations** for improved gradient flow.  

---

## Dataset & Preprocessing  
- Images are loaded from a **custom dataset** directory.  
- Preprocessed using **`torchvision.transforms`**:  
  - Resized to your `taget.shape`.  
  - Normalized to `[-1, 1]` for stable training.  

---

## 🔧 Training Setup  

### **Pretraining Generator**  
Before GAN training, the generator is **pretrained** using a **Smooth L1 loss**:  
```python
pretrain_generator(generator, dataloader, g_pretrain_optimizer, pretrain_criterion, device, noise_dim, pretrain_epochs)
