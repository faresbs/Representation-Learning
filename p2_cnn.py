import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import torchvision
import torchvision.transforms

# Define image transformations & Initialize datasets
mnist_transforms = torchvision.transforms.ToTensor()
mnist_train = torchvision.datasets.MNIST(root='./data', train=True, transform=mnist_transforms, download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, transform=mnist_transforms, download=True)

# Create multi-threaded DataLoaders
trainloader = torch.utils.data.DataLoader(mnist_train, batch_size=64, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(mnist_test, batch_size=64, shuffle=True, num_workers=2)

# 52650 parameters for the mlp
class Classifier(nn.Module):
    """CNN Classifier"""
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 2
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 3
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),

            # Layer 4
            nn.Conv2d(in_channels=48, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
        # Logistic Regression
        self.clf = nn.Linear(96, 10)

    def forward(self, x):
        return self.clf(self.conv(x).squeeze())


cuda_available = torch.cuda.is_available()

clf = Classifier()
total_parameters = sum([p.data.numpy().size for p in clf.parameters()])
print('total number of parameters in the network = ' + str(total_parameters))

if cuda_available:
    clf = clf.cuda()
optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4)
# criterion includes the softmax
criterion = nn.CrossEntropyLoss()

n_epochs = 10
loss_history = []
train_acc_history = []
valid_acc_history = []
for epoch in range(n_epochs):
    losses = []
    total = 0
    correct = 0
    # Train
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = clf(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    train_acc_history.append(100.*correct/total)
    loss_history.append(np.mean(losses))
    print('Epoch : %d Loss : %.3f ' % (epoch, np.mean(losses)))
    print('Epoch : %d Train Acc : %.3f' % (epoch, 100.*correct/total))

    # Evaluate
    clf.eval()
    total = 0
    correct = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if cuda_available:
            inputs, targets = inputs.cuda(), targets.cuda()

        outputs = clf(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    valid_acc_history.append(100.*correct/total)
    print('Epoch : %d Validation Acc : %.3f' % (epoch, 100.*correct/total))
    print('--------------------------------------------------------------')
    clf.train()

plt.figure()
plt.plot(np.arange(n_epochs),train_acc_history, label='Training accuracy')
plt.plot(np.arange(n_epochs),valid_acc_history,  'r', label='Validation accuracy')
# plt.title("Gradient checking for W2")
# plt.xlabel('N')
# plt.ylabel('Max. difference')
plt.legend()
plt.show()
