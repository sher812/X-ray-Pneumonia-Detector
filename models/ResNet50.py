import torch.nn as nn
import torch.nn.functional as F
import torch


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        # convolutional layers with parameters in_channels, out_channels, kernel_size, stride
        self.conv1 = nn.Conv2d(3, 58, 7, 1)

        # convolutional block, Conv with 1 input channels and 3 out put channels as parameters
        self.Conv1 = Conv(58, 68, 66, 68)
        self.Conv2 = Conv(68, 68, 66, 68)
        self.Conv3 = Conv(68, 68, 66, 68)
        self.Conv4 = Conv(68, 68, 66, 68)

        # identity block, Identity with 1 input channels and 3 out put channels as parameters
        self.Identity1 = Identity(68, 64, 64)
        self.Identity2 = Identity(68, 64, 64)
        self.Identity3 = Identity(68, 64, 64)
        self.Identity4 = Identity(68, 64, 64)
        self.Identity5 = Identity(68, 64, 64)
        self.Identity6 = Identity(68, 64, 64)
        self.Identity7 = Identity(68, 64, 64)
        self.Identity8 = Identity(68, 64, 64)
        self.Identity9 = Identity(68, 64, 64)
        self.Identity10 = Identity(68, 64, 64)
        self.Identity11 = Identity(68, 64, 64)
        self.Identity12 = Identity(68, 64, 64)

        # apply batch normalization
        # go through average pooling in second branch
        self.BatchNorm1 = nn.BatchNorm2d(58)
        self.AvgPool2D = nn.AvgPool2d(50)

        # adding fully connected layer
        self.fc1 = nn.Linear(68, 2)

        # adding dropout layer at the end
        self.dropout1 = nn.Dropout2d(0.25)

    def forward(self, x):
        x = self.conv1(x)
        x = self.BatchNorm1(x)
        x = F.relu(x)

        x = F.max_pool2d(x, 3)

        x = self.Conv1(x)

        x = self.Identity1(x)
        x = self.Identity2(x)

        x = self.Conv2(x)

        x = self.Identity3(x)
        x = self.Identity4(x)
        x = self.Identity5(x)

        x = self.Conv3(x)

        x = self.Identity6(x)
        x = self.Identity7(x)
        x = self.Identity8(x)
        x = self.Identity9(x)
        x = self.Identity10(x)

        x = self.Conv4(x)

        x = self.Identity11(x)
        x = self.Identity12(x)

        x = self.AvgPool2D(x)

        # reshape tensor into 68 rows
        x = x.view(-1, 68)
        x = self.fc1(x)
        x = self.dropout1(x)

        # calculating and returning the log probability of each sample in batch
        output = F.log_softmax(x, dim=1)
        return output


class Conv(nn.Module):
    def __init__(self, input1, output1, output2, output3):
        super(Conv, self).__init__()
        self.input1 = input1
        self.output1 = output1
        self.output2 = output2
        self.output3 = output3

        # convolutional layers in first branch of the conv block
        self.conv1a = nn.Conv2d(self.input1, self.output1, 1, 1, 2)
        self.conv2a = nn.Conv2d(self.output1, self.output2, 3, 1, 2)

        # Added a padding of 1 in conv3a so that x1 and x2 in the forward function are of equal sizes
        self.conv3a = nn.Conv2d(self.output2, self.output3, 1, 1, 2)

        # convolutional layer in second branch of the conv block
        self.conv1b = nn.Conv2d(self.input1, self.output3, 1, 1, 5)

        self.BatchNorm1 = nn.BatchNorm2d(self.output1)
        self.BatchNorm2 = nn.BatchNorm2d(self.output2)
        self.BatchNorm3 = nn.BatchNorm2d(self.output3)

        # applying batch normalization
        self.dropout1 = nn.Dropout2d(0.15)

    def forward(self, x):
        x1 = self.conv1a(x)
        x1 = self.BatchNorm1(x1)
        x1 = F.relu(x1)

        x1 = self.conv2a(x1)
        x1 = self.BatchNorm2(x1)
        x1 = F.relu(x1)
        x1 = self.dropout1(x1)

        x1 = self.conv3a(x1)
        x1 = self.BatchNorm3(x1)

        x2 = self.conv1b(x)
        x2 = self.BatchNorm3(x2)

        # adds the two branchs output matrix together
        output = torch.add(x1, x2)
        output = F.relu(output)

        return output


class Identity(nn.Module):
    def __init__(self, input1, output1, output2):
        super(Identity, self).__init__()
        self.input1 = input1
        self.output1 = output1
        self.output2 = output2

        # convolutional layers of the first branch
        self.conv1a = nn.Conv2d(self.input1, self.output1, 1, 1)
        self.conv2a = nn.Conv2d(self.output1, self.output2, 3, 1, 1)
        self.conv3a = nn.Conv2d(self.output2, self.input1, 1, 1)

        self.BatchNorm1 = nn.BatchNorm2d(self.output1)
        self.BatchNorm2 = nn.BatchNorm2d(self.output2)
        self.BatchNorm3 = nn.BatchNorm2d(self.input1)

    def forward(self, x):
        x1 = self.conv1a(x)
        x1 = self.BatchNorm1(x1)
        x1 = F.relu(x1)

        x1 = self.conv2a(x1)
        x1 = self.BatchNorm2(x1)
        x1 = F.relu(x1)

        x1 = self.conv3a(x1)
        x1 = self.BatchNorm3(x1)

        # adds the output of the two branches together
        output = torch.add(x, x1)
        output = F.relu(output)
        return output
