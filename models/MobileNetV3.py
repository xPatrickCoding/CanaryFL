import torch.nn as nn
from torchvision import models

class MobileNetV3(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, version='small'):
        super(MobileNetV3, self).__init__()

        if version == 'small':
            self.model = models.mobilenet_v3_small(weights=None)  
        elif version == 'large':
            self.model = models.mobilenet_v3_large(weights=None)
        else:
            raise ValueError("MobileNetV3 version must be 'small' or 'large'")

        
        if in_channels != 3:
            self.model.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)

        
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
