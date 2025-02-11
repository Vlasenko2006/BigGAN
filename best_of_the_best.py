import os
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.utils.spectral_norm as spectral_norm

def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def plot_generated_images(generator, discriminator, epoch, device, examples=10, path_to_results="./results", sample=1, batch_size=100):
    check_and_create_directory(path_to_results)

    num_samples = 1000  # Total number of images to generate
    scores = []
    images = []

    # Generate images in batches
    for i in range(0, num_samples, batch_size):
        batch_size_actual = min(batch_size, num_samples - i)  # Handle last batch size
        noise = torch.randn(batch_size_actual, noise_dim).to(device)
        with torch.no_grad():
            batch_images = generator(noise).cpu()  # Move to CPU to avoid GPU OOM
            batch_scores = discriminator(batch_images.to(device)).cpu().squeeze()  # Compute scores

        images.append(batch_images)
        scores.append(batch_scores)

    # Concatenate results
    images = torch.cat(images, dim=0)
    scores = torch.cat(scores, dim=0)

    # Select top `examples` images based on discriminator scores
    top_indices = torch.argsort(scores, descending=True)[:examples]
    best_images = images[top_indices]

    # Rescale to [0, 1] and convert to numpy
    best_images = (best_images + 1) / 2  
    best_images = best_images.permute(0, 2, 3, 1).numpy()

    # Plot images
    plt.figure(figsize=(10, 2))
    for i in range(examples):
        plt.subplot(2, examples // 2, i + 1)
        plt.imshow(best_images[i])
        plt.axis('off')
    plt.tight_layout()

    # Save and display
    plt.savefig(f"{path_to_results}/gan3_best_generated_image_epoch_{epoch}_sample{sample}.png")
    plt.show()



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
    def __init__(self, noise_dim, nx_in=60, ny_in=70, nx_out=136, ny_out=204, hidden_dim=128, num_channels=3, kernel_size=4):
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

        # Transposed Convolutional Upsampling with Residual Blocks
        self.conv_block = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            ResidualBlock(self.hidden_dim),
            nn.BatchNorm2d(self.hidden_dim),

            spectral_norm(nn.ConvTranspose2d(self.hidden_dim, self.hidden_dim // 2, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            ResidualBlock(self.hidden_dim // 2),
            nn.BatchNorm2d(self.hidden_dim // 2),

            spectral_norm(nn.ConvTranspose2d(self.hidden_dim // 2, self.hidden_dim // 4, kernel_size=self.kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
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






# Hyperparameters
image_size = (136, 204)  # Updated image size
noise_dim = int(136 * 5) + 5
batch_size = 16
learning_rate = 0.0001 * 0.2
pretrain_epochs = 7
gan_epochs = 300000
smoothing = 0.04  #.45
accumulation_steps=8 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize(image_size),  # Fix here
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])

# Load custom dataset

# Initialize models
generator = Generator(noise_dim).to(device)
discriminator = Discriminator(image_size=image_size).to(device)

start_from_scratch = True
path_to_results = "./results_best"
path_to_images = "/gpfs/work/vlasenko/07/NN/Images_clean_small2/"
epoch = 8500
saved_models = 'gan_model5/'
nsamples= 100


#=========


checkpoint_path_g = saved_models + "generator_epoch_" + str(epoch) + ".pth"
checkpoint_path_d = saved_models + "discriminator_epoch_" + str(epoch) + ".pth"


checkpoint_g = torch.load(checkpoint_path_g, map_location=torch.device('cpu'))
checkpoint_d = torch.load(checkpoint_path_d, map_location=torch.device('cpu'))
generator.load_state_dict(checkpoint_g)
discriminator.load_state_dict(checkpoint_d) 
checkpoint_g = torch.load(checkpoint_path_g, map_location=torch.device('cpu'))
checkpoint_d = torch.load(checkpoint_path_d, map_location=torch.device('cpu'))
generator.load_state_dict(checkpoint_g)
discriminator.load_state_dict(checkpoint_d)


import gc

for i in range(nsamples):
    print(f"Processing: {i} out of {nsamples}")
    
    plot_generated_images(generator,
                          discriminator,
                          epoch,
                          device,
                          examples=10, 
                          path_to_results=path_to_results,
                          sample=i)
    
    torch.cuda.empty_cache()  # Clears GPU memory
    gc.collect()  # Frees CPU memory

    
