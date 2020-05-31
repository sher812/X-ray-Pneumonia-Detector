import torch.nn as nn
import torch.nn.functional as F
import torch


# Add dropouts when we know what/how to use them...
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        # conv2D layer 3 channels in, 64 channels out, kernel size of 11 and padding of 2
        # initialise all convolution layers in the Alex-Net Architecture

        # start with a padding of two to avoid too much downsampling so image doesn't become 0x0
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 11, padding = 2)
        self.conv2 = nn.Conv2d(64, 66, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv2d(66, 68, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(68, 70, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(70, 72, kernel_size=3, padding=1)

        # fully connected layers (inputs, outputs)
        self.fc1 = nn.Linear(72*7*7, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        # output of 2 characteristics: NORMAL or PNEUMONIA
        self.fc3 = nn.Linear(1000, 2)

    # Defining how the Alex-Net is going to run.
    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = F.relu(x)

        x = self.conv5(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # reshape the tensor with 256*7*7 rows
        x = x.view(-1, 72*7*7)
        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)

        x = self.fc3(x)
        # calculate and return log probabilities for each sample in batch
        output = F.log_softmax(x, dim=1)

        return output