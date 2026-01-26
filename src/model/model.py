import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple CNN for binary classification (Cat vs Dog).
    Input images: 3x224x224
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.model = nn.Sequential(
            # Conv layers
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # for binary classification
        )

    def forward(self, x):
        return self.model(x)
