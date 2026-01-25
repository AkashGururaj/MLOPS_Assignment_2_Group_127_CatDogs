import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),  # (3,224,224) -> (32,224,224)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> (32,112,112)
            
            nn.Conv2d(32, 64, 3, padding=1), # -> (64,112,112)
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> (64,56,56)
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                 # -> (128,28,28)
            
            nn.Flatten(),
            nn.Linear(128*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)
