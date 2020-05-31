import torch.nn as nn
import torch.nn.functional as F
import torch


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()

        # Conv2D layer (input size, output size, kernel size, stride, padding).
        # Added a padding of 1 to each layer to avoid too much downsampling so that the
        # image size does not become 0x0.
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv8 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv9 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv10 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv11 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv13 = nn.Conv2d(64, 64, 3, 1, 1)

        # Dropout layers (probability)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.dropout3 = nn.Dropout2d(0.5)

        # fully connected layers (input size, output size)
        self.fc1 = nn.Linear(64 * 2 * 2, 4096)
        self.fc2 = nn.Linear(4096, 1000)
        self.fc3 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = F.relu(x)
        x = self.conv7(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = F.relu(x)
        x = self.conv10(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv11(x)
        x = F.relu(x)
        x = self.conv12(x)
        x = F.relu(x)
        x = self.conv13(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        x = self.dropout1(x)

        # Changing the shape of tensors to be one 1 x 64.
        # Fully connected layers take in 1D tensor as input. In this case, tensors were of size 64 x 1 x 1.
        x = x.view(-1, 64 * 2 * 2)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)

        return output
