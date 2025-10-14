from medmnist import PathMNIST
from torch.utils.data import Dataset
import torch 

class WrappedPathMNIST(PathMNIST):
    def __init__(self, split, transform=None, download=False):
        super().__init__(split=split, transform=transform, download=download)
        self.targets = self.labels



    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = int(target[0])
        return img, target
