import torch
import torch.nn as nn
import torch.nn.functional as F

class EllipseNet(nn.Module):  
    def __init__(self):
        super(EllipseNet, self).__init__()

        # צמצום נוסף: 12→8, 24→16, 48→32, 96→64
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # input: (1,64,64)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # חצי גודל

        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Calculate the actual flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 64, 64)
            dummy_output = self._forward_features(dummy_input)
            self.flattened_size = dummy_output.view(-1).size(0)

        # צמצום FC נוסף: 192→128
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, 48)
        self.output = nn.Linear(48, 6)  # [isEllipse, x, y, major, minor, angle]
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)

    def _forward_features(self, x):
        """Forward pass through convolutional layers only"""
        x = F.relu(self.conv1(x))      # (64,64) → (64,64)
        x = self.pool(F.relu(self.conv2(x)))  # → (32,32)

        x = F.relu(self.conv3(x))      # → (32,32)
        x = self.pool(F.relu(self.conv4(x)))  # → (16,16) → (8,8)

        x = self.pool(x)               # → (4,4)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        
        # Flatten the features
        x = x.reshape(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))
        x = self.dropout(x)  # Apply dropout
        return self.output(x)          # shape: (batch_size, 6)
