import torch.nn as nn
import torch.nn.functional as F
import torch


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # Conv2D layer (input size, output size, kernel size, stride, padding).
        # Added a padding of 1 to each layer to avoid too much downsampling so that the
        # image size does not become 0x0.
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 5, padding = 1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size = 5, padding = 1)

        # Fully connected layers (input, output)
        self.fc1 = nn.Linear(128*14*14, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.avg_pool2d(x, 2)

        # Changing the shape of tensors to be one 1, 128*14*14.
        # Fully connected layers take in 1D tensor as input. In this case, tensors were of size 128 x 14 x 14.
        x = x.view(-1, 128*14*14)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

        return output
