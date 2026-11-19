"""LeNet-style classifier with a controlled activation-function ablation."""

import torch
from torch import nn
from torch.nn import functional as F


class LeNet(nn.Module):
    """The reference layer sizes, with either sigmoid or ReLU activations."""

    def __init__(self, activation="sigmoid"):
        super().__init__()
        if activation not in {"sigmoid", "relu"}:
            raise ValueError("activation must be sigmoid or relu")
        self.activation_name = activation
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, 10)

    def activate(self, x):
        return torch.sigmoid(x) if self.activation_name == "sigmoid" else F.relu(x)

    def features(self, x):
        """Expose the post-activation feature maps and average-pooled maps."""
        conv1 = self.activate(self.conv1(x))
        pool1 = F.avg_pool2d(conv1, kernel_size=2, stride=2)
        conv2 = self.activate(self.conv2(pool1))
        pool2 = F.avg_pool2d(conv2, kernel_size=2, stride=2)
        return {"conv1": conv1, "pool1": pool1, "conv2": conv2, "pool2": pool2}

    def forward(self, x):
        x = self.features(x)["pool2"]
        x = torch.flatten(x, start_dim=1)
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        return self.classifier(x)


def integer_translate(images, dx, dy, max_shift=2):
    """Translate each image using zero padding, with no wrap-around.

    Positive dx moves the ink right and positive dy moves it down. dx and dy
    may be integers or a tensor containing one displacement per image.
    """
    count, channels, height, width = images.shape
    dx = torch.as_tensor(dx, device=images.device).expand(count)
    dy = torch.as_tensor(dy, device=images.device).expand(count)
    if torch.any(dx.abs() > max_shift) or torch.any(dy.abs() > max_shift):
        raise ValueError("Translation exceeds the configured padding")
    padded = F.pad(images, (max_shift, max_shift, max_shift, max_shift))
    source_y = torch.arange(height, device=images.device)[None, :, None] + max_shift - dy[:, None, None]
    source_x = torch.arange(width, device=images.device)[None, None, :] + max_shift - dx[:, None, None]
    index = (source_y * padded.shape[-1] + source_x).reshape(count, 1, -1)
    index = index.expand(count, channels, -1)
    return padded.flatten(start_dim=2).gather(2, index).reshape_as(images)
