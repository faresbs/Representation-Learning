import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torchvision import models


class vgg(nn.Module):

    def __init__(self, num_classes=1000):
        super(vgg, self).__init__()
        
        self.features = nn.Sequential(

            # Layer 1
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=12, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),
            
            # Layer 2
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),

            # Layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),
            
            # Layer 4
            nn.Conv2d(in_channels=64, out_channels=126, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=126, out_channels=126, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),

            # Layer 5
            nn.Conv2d(in_channels=126, out_channels=224, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=224, out_channels=224, kernel_size=(3, 3), padding=1, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2, dilation=1, padding=0),

            # Layer 6
            #nn.Conv2d(in_channels=126, out_channels=126, kernel_size=(1, 1), padding=0, stride=1),
            #nn.ReLU(),
            #nn.Conv2d(in_channels=126, out_channels=64, kernel_size=(1, 1), padding=0, stride=1),
            #nn.ReLU(),
            
        )
        
        #features = list(models.vgg16().features)
        #self.features = nn.ModuleList(features)

        self.classifier = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, num_classes))


    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x