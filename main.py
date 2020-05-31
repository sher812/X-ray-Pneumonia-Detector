from __future__ import print_function
from sklearn import metrics
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from models.AlexNet import AlexNet
from models.vgg16 import vgg16
from models.ResNet50 import ResNet50
from models.LeNet5 import LeNet5

L1 = []  # stores training loss in a list
L2 = []  # stores testing loss in a list
A1 = []  # stores training accuracy in a list

y_true = []  # list of true values for metrics functions
y_pred = []  # list of predicted values for metrics functions


# functions to show an image
def imsave(img):
    npimg = img.numpy()
    npimg = (np.transpose(npimg, (1, 2, 0)) * 255).astype(np.uint8)
    im = Image.fromarray(npimg)
    im.save("./results/your_file.jpeg")

# trains the model and prints running time loss
def train_cnn(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()

    # temp_loss will add up the loss in each iteration of the for loop and loss_track
    # would keep a track of the number of iterations. temp_loss would be divided by
    # loss_track to return average loss. This would be used for ploting the training loss.
    temp_loss = 0
    loss_track = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        output = model(data)
        loss = F.nll_loss(output, target)
        temp_loss = temp_loss + loss
        loss_track = loss_track + 1
        loss.backward();
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    # appending average loss into L1
    L1.append(temp_loss / loss_track)

# tests and outputs the loss and accuracy values for each epoch.
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    # appending testing loss after 1 epoch into L2
    L2.append(test_loss)

    # appending testing accuracy after 1 epoch into A1
    A1.append(float(correct / len(test_loader.dataset)))

    # prints average testing loss and accracy after 1 epoch
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

# this function prints all metrics and plots all results for our trained model
def Evaluation(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = torch.squeeze(data)
            data, target = data.to(device), target.to(device)

            # appending target into y_true as a list
            y_true.append(target.tolist())
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            # changing the size of temp_pred to 78 x 1. temp_pred would then be appended into y_pred as a list
            # temp_pred has to be the same size as target to input in metrics functions
            temp_pred = pred
            temp_pred = temp_pred.view(78)
            y_pred.append(temp_pred.tolist())
            correct += pred.eq(target.view_as(pred)).sum().item()

    # flattening y_true and y_pred. This code would combine all the list type elements in y_true and y_pred
    # into one large list. This is necessary as we want to compare all results in metrics functions.
    y_true_flattened = [y for x in y_true for y in x]
    y_pred_flattened = [y for x in y_pred for y in x]
    print('------------Confusion Matrix------------')
    print(metrics.confusion_matrix(y_true_flattened, y_pred_flattened, labels=[0, 1]))
    print('------------Classification Report------------')
    print(metrics.classification_report(y_true_flattened, y_pred_flattened, labels=[0, 1]))

    # plots training loss
    plt.plot(L1)
    plt.ylabel('Loss(training)')
    plt.show()

    # plots testing accuracy and testing loss
    plt.plot(A1, label='Accuracy(testing)')
    plt.plot(L2, label='Loss(testing)')
    plt.legend()
    plt.show()


def main():
    epoches = 4
    gamma = 0.7
    log_interval = 10
    torch.manual_seed(1)
    save_model = True

    # Check whether you can use Cuda
    use_cuda = torch.cuda.is_available()
    # Use Cuda if you can
    device = torch.device("cuda" if use_cuda else "cpu")

    ######################3   Torchvision    ###########################3
    # Use data predefined loader
    # Pre-processing by using the transform.Compose
    # divide into batches

    # used in trainset and testset.
    # changes the image sizes to 64 x 64, converts them into a tensor and
    # normalizes them
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # takes in the training dataset and transforms them as shown above
    trainset = torchvision.datasets.ImageFolder(root='./data/chest_xray/train', transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=26, shuffle=True, num_workers=0)

    # takes in the testing dataset and transforms them as shown above
    testset = torchvision.datasets.ImageFolder(root='./data/chest_xray/test', transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=78, shuffle=False, num_workers=0)

    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # img = torchvision.utils.make_grid(images)
    # imsave(img)

    # #####################    Build your network and run   ############################

    # select the model to work with
    model = AlexNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)

    for epoch in range(1, epoches + 1):
        train_cnn(log_interval, model, device, train_loader, optimizer, epoch)

        test(model, device, test_loader)
        scheduler.step()

    # runs the evaluation function to print the metrics data and plots
    Evaluation(model, device, test_loader)

    # save results to results folder
    if save_model:
        torch.save(model.state_dict(), "./results/AlexNet.pt")


if __name__ == '__main__':
    main()