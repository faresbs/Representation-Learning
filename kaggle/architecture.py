import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models



class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

#residual blocks with skip connections
class ResLinear(nn.Module):

    def __init__(self, in_features, out_features, activation=nn.ReLU()):
        super(ResLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        self.linear = nn.Linear(in_features, out_features)
        if in_features != out_features:
            self.project_linear = nn.Linear(in_features, out_features)
        
    def forward(self, x):
        inner = self.activation(self.linear(x))
        if self.in_features != self.out_features:
            skip = self.project_linear(x)
        else:
            skip = x
        return inner + skip



class network(nn.Module):

    def __init__(self, num_classes=1000):
        super(network, self).__init__()
        
        self.features = nn.Sequential(

            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),
            
            # Layer 2
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),

            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),
            
            # Layer 4
            #nn.Conv2d(in_channels=64, out_channels=124, kernel_size=(3, 3), padding=1, stride=1),
            #nn.ReLU(),
            #nn.Conv2d(in_channels=124, out_channels=124, kernel_size=(3, 3), padding=1, stride=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),

            # Layer 5
            #nn.Conv2d(in_channels=124, out_channels=124, kernel_size=(3, 3), padding=1, stride=1),
            #nn.ReLU(),
            #nn.Conv2d(in_channels=124, out_channels=124, kernel_size=(3, 3), padding=1, stride=1),
            #nn.ReLU(),
            #nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),
            
        )
        
        #features = list(models.vgg16().features)
        #self.features = nn.ModuleList(features)

        self.classifier = nn.Sequential(
            Flatten(),
            ResLinear(64 * 8 * 8, 124),
            #nn.ReLU(True),
            ResLinear(124, 84),
            #nn.ReLU(True),
            ResLinear(84, num_classes))


    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x