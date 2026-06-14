import torch.nn as nn

class MarioNet(nn.Module):
    """
    Takes a stack of grayscale frames (so it can see motion) and outputs
    one Q-value per possible action.
    """

    def __init__(self, input_channels, num_actions):
        super().__init__()

        # Convolutional layers: scan the image for patterns
        # (edges, shapes, enemies, pipes, etc.)
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Fully connected layers: turn detected features into action values
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),  # 3136 = output size of conv layers for 84x84 input
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x