import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import gc
import torch.cuda.amp as amp
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def check_and_create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created.")
    else:
        print(f"Directory '{directory_path}' already exists.")

def plot_generated_images(generator, epoch, device, examples=10, path_to_results="./results"):
    check_and_create_directory(path_to_results)
    noise = torch.randn(examples, noise_dim).to(device)
    generated_images = generator(noise).cpu().detach()
    generated_images = (generated_images + 1) / 2
    generated_images = generated_images.permute(0, 2, 3, 1).numpy()

    plt.figure(figsize=(10, 2))
    for i in range(examples):
        plt.subplot(2, examples // 2, i + 1)
        plt.imshow(generated_images[i])
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"{path_to_results}/gan3_generated_image_epoch_{epoch}.png")
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
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, noise_dim, nx_in=60, ny_in=70, nx_out=136, ny_out=204, hidden_dim=128, num_channels=3, kernel_size=4):
        super(Generator, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim * nx_in * ny_in),
            nn.ReLU()
        )
        self.conv_block = nn.Sequential(
            spectral_norm(nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            ResidualBlock(hidden_dim),
            nn.BatchNorm2d(hidden_dim),
            spectral_norm(nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            ResidualBlock(hidden_dim // 2),
            nn.BatchNorm2d(hidden_dim // 2),
            spectral_norm(nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)),
            nn.ReLU(),
            ResidualBlock(hidden_dim // 4),
            nn.BatchNorm2d(hidden_dim // 4),
            spectral_norm(nn.ConvTranspose2d(hidden_dim // 4, num_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1)),
            nn.Tanh()
        )
        self.upsample = nn.Upsample(size=(nx_out, ny_out), mode="bilinear", align_corners=True)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
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
            spectral_norm(nn.Conv2d(512, 1, 4, stride=2, padding=1)),
        )
        self.feature_dim = self._get_conv_output(image_size)
        self.fc = nn.Linear(self.feature_dim, 1)

    def _get_conv_output(self, image_size):
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, *image_size)
            output = self.conv_layers(dummy_input)
            return output.view(1, -1).size(1)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, f) for f in sorted(os.listdir(root_dir)) if f.endswith(".jpg") or f.endswith(".jpeg")]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(1.0)

def smooth_labels(labels, smoothing=0.1):
    return labels * (1 - smoothing) + 0.5 * smoothing

def train_gan(generator, discriminator, g_optimizer, d_optimizer, dataloader, device, noise_dim, epochs=100, path=".", smoothing=0.1, path_to_results="results/", accumulation_steps=4):
    generator.train()
    discriminator.train()
    scaler = amp.GradScaler()
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        epoch_loss_g = 0.0
        epoch_loss_d = 0.0
        gc.collect()
        torch.cuda.empty_cache()
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for i, (real_images, _) in enumerate(progress_bar):
                batch_size = real_images.size(0)
                real_images = real_images.to(device)
                noisy_real_images = real_images + 0.05 * torch.randn_like(real_images)
                noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

                with amp.autocast():
                    fake_images = generator(noise)
                    real_labels = torch.ones(batch_size, 1).to(device) * (1 - smoothing)
                    fake_labels = torch.zeros(batch_size, 1).to(device)
                    d_real_logits = discriminator(noisy_real_images)
                    d_fake_logits = discriminator(fake_images.detach())
                    d_loss_real = criterion(d_real_logits, real_labels)
                    d_loss_fake = criterion(d_fake_logits, fake_labels)
                    d_loss = (d_loss_real + d_loss_fake) / accumulation_steps
                scaler.scale(d_loss).backward()
                if (i + 1) % accumulation_steps == 0:
                    scaler.step(d_optimizer)
                    scaler.update()
                    d_optimizer.zero_grad()

                for _ in range(2):
                    noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
                    with amp.autocast():
                        fake_images = generator(noise)
                        g_loss = criterion(discriminator(fake_images), real_labels) / accumulation_steps
                    scaler.scale(g_loss).backward()
                    if (i + 1) % accumulation_steps == 0:
                        scaler.step(g_optimizer)
                        scaler.update()
                        g_optimizer.zero_grad()
                epoch_loss_d += d_loss.item()
                epoch_loss_g += g_loss.item()
                progress_bar.set_postfix(D_loss=d_loss.item(), G_loss=g_loss.item())
        print(f"Epoch {epoch+1}/{epochs} - D Loss: {epoch_loss_d/len(dataloader):.8f}, G Loss: {epoch_loss_g/len(dataloader):.4f}")
        if (epoch + 1) % 10 == 0:
            print(" ==== saving images ==== ")
            plot_generated_images(generator, epoch + 1, device, path_to_results=path_to_results)
            torch.save({
                'epoch': epoch + 1,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_G_state_dict': g_optimizer.state_dict(),
                'optimizer_D_state_dict': d_optimizer.state_dict(),
            }, os.path.join(path, f"checkpoint_epoch_{epoch+1}.pth"))

def pretrain_generator(generator, dataloader, optimizer, criterion, device, noise_dim, epochs=20):
    generator.train()
    scaler = amp.GradScaler()
    for epoch in range(epochs):
        epoch_loss = 0.0
        gc.collect()
        torch.cuda.empty_cache()
        with tqdm(dataloader, desc=f"Pretraining Generator Epoch {epoch+1}/{epochs}", unit="batch") as progress_bar:
            for real_images, _ in progress_bar:
                batch_size = real_images.size(0)
                real_images = real_images.to(device)
                noise = torch.randn(batch_size, noise_dim).to(device)
                with amp.autocast():
                    fake_images = generator(noise)
                    loss = criterion(fake_images, real_images)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                progress_bar.set_postfix(Loss=loss.item())
        print(f"Pretraining Generator Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(dataloader):.6f}")


# Model setup
image_size = (136, 204)
noise_dim = int(136 * 5) + 5
batch_size = 16
learning_rate = 0.0001 * 0.2 * 0.3
pretrain_epochs = 7
gan_epochs = 300000
smoothing = 0.04
accumulation_steps = 8

# IO setup

start_from_scratch = False
path = ""
path_to_results = ""
path_to_images = ""
load_epoch = 3800
saved_models = ''



#=========


checkpoint_path_g = saved_models + "generator_epoch_-" + str(load_epoch) + ".pth"
checkpoint_path_d = saved_models + "discriminator_epoch_-" + str(load_epoch) + ".pth"

os.makedirs(path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  
])

dataset = ImageDataset(root_dir=path_to_images, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
generator = Generator(noise_dim).to(device)
discriminator = Discriminator(image_size=image_size).to(device)

g_pretrain_optimizer = optim.Adam(generator.parameters(), lr=1e-5, betas=(0.5, 0.999))
g_optimizer = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
pretrain_criterion = nn.SmoothL1Loss()

if start_from_scratch:
    print("Pretraining the generator...")
    pretrain_generator(generator, dataloader, g_pretrain_optimizer, pretrain_criterion, device, noise_dim, pretrain_epochs)
else:
    checkpoint_g = torch.load(checkpoint_path_g, map_location=torch.device('cpu'))
    checkpoint_d = torch.load(checkpoint_path_d, map_location=torch.device('cpu'))
    generator.load_state_dict(checkpoint_g)
    discriminator.load_state_dict(checkpoint_d)

print("Starting GAN training...")
train_gan(generator, discriminator, g_optimizer, d_optimizer, dataloader, device, noise_dim, gan_epochs, path=path, smoothing=smoothing, path_to_results=path_to_results, accumulation_steps=accumulation_steps)
