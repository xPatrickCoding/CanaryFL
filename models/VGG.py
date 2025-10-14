import torch.nn as nn
from torchvision import models

class VGG(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, version="vgg11"):
        super(VGG, self).__init__()

        vgg_versions = {
            "vgg11": models.vgg11,
            "vgg13": models.vgg13,
            "vgg16": models.vgg16,
            "vgg19": models.vgg19
        }

        if version not in vgg_versions:
            raise ValueError(f"Unsupported VGG version: {version}")

        # load without pretrained weights
        self.model = vgg_versions[version](weights=None)

        #input layer
        if in_channels != 3:
            self.model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)

        # output layer
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
