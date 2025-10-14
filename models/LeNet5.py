import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet5(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels=1, dropout_p=0.5):
        super(LeNet5, self).__init__()
        
        # Convolutional Layer 1
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)  # Input: 1x28x28, Output: 6x24x24
        self.pool = nn.MaxPool2d(2, 2)  # MaxPool instead of AvgPool
        
        # Convolutional Layer 2
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Input: 6x12x12, Output: 16x8x8
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.dropout1 = nn.Dropout(p=dropout_p)   # Dropout after fc1
        
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=dropout_p)   # Dropout after fc2
        
        self.fc3 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 6x24x24 -> 6x12x12
        x = self.pool(F.relu(self.conv2(x)))  # 16x8x8 -> 16x4x4
        
        x = x.view(-1, 16 * 4 * 4)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)          # Dropout after fc1
        
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)          # Dropout after fc2
        
        x = self.fc3(x)
        return x
