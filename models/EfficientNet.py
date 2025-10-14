import torch.nn as nn
from torchvision import models

class EfficientNetWrapper(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, version="efficientnet_b0"):
        super().__init__()

        efficientnet_versions = {
            "efficientnet_b0": models.efficientnet_b0,
            "efficientnet_b1": models.efficientnet_b1,
            "efficientnet_b2": models.efficientnet_b2,
        }

        if version not in efficientnet_versions:
            raise ValueError(f"Unsupported EfficientNet version: {version}")

        self.model = efficientnet_versions[version](weights=None)

        #Input Layer 
        if in_channels != 3:
            conv0 = self.model.features[0][0]
            self.model.features[0][0] = nn.Conv2d(
                in_channels,
                conv0.out_channels,
                kernel_size=conv0.kernel_size,
                stride=conv0.stride,
                padding=conv0.padding,
                bias=False
            )

        # Output Layer 
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
