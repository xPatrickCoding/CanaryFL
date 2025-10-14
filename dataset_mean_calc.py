from medmnist import PathMNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

# 1. Load Dataset
dataset = PathMNIST(split='train', download=True, transform=transforms.ToTensor())

# 2. Loading dataset in dataloader
loader = DataLoader(dataset, batch_size=128, shuffle=False)

# 3. Akkumulation
mean = 0.
std = 0.
n_samples = 0.

for images, _ in loader:
    # images: [B, C, H, W]
    batch_samples = images.size(0)
    images = images.view(batch_samples, images.size(1), -1)  # [B, C, H*W]
    mean += images.mean(2).sum(0)
    std += images.std(2).sum(0)
    n_samples += batch_samples

mean /= n_samples
std /= n_samples

print("Mean:", mean)
print("Std:", std)


