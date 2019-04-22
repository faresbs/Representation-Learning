import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torch.utils.data import dataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch
import matplotlib.pyplot as plt

data_path='Assignment3/'

####################### HYPERPARAMETERS DEFINITION
seed=1111
n_epochs=700
batch_size=32
lr=0.0001
img_size=32
channels=3
latent_dim=100
b1=0.5
b2=0.999
sample_interval=1000
critic=10
lambda_gp = 10 #loss weight for gradient penalty
img_shape = (channels, img_size, img_size)

cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

#######################

####################### Configure data loader and split dataset
transform = image_transform = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((.5, .5, .5),(.5, .5, .5))])
    
svhn_dataset = datasets.SVHN(data_path,
                        split='train',
                        download=True,
                        transform=transform)

trainset_size = int(len(svhn_dataset) * 0.9)

trainset, validset = dataset.random_split(svhn_dataset,
                                          [trainset_size, 
                                           len(svhn_dataset) - trainset_size])

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=2)

validloader = torch.utils.data.DataLoader(validset,
                                          batch_size=batch_size,)

####################### Generator definition
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img

####################### Discriminator definition
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, img):
        img_output = img.view(img.shape[0], -1)
        output = self.model(img_output)
        return output

####################### Gradient penalty
def gradient_penalty(D, real, fake):
    
    r = Tensor(np.random.random((real.size(0), 1, 1, 1)))
    interp = (r * real + ((1 - r) * fake)).requires_grad_(True)
    d_interp = D(interp)
    gradients = autograd.grad(
        outputs=d_interp,
        inputs=interp,
        grad_outputs=Variable(Tensor(real.shape[0], 1).fill_(1.0), requires_grad=False),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

#######################
    
# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()


####################### Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))


#####################################################################
#######################  Training ###################################
#####################################################################

if __name__ == "__main__":
    iteration = 0
    
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(trainloader):
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            
            #  Train Discriminator          
            optimizer_D.zero_grad()
            
            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))
            
            # Generate a batch of images
            fake_imgs = generator(z)
            
            # Predict real images
            real_predict = discriminator(real_imgs)
            
            # Predict fake images
            fake_predict = discriminator(fake_imgs)
            
            # Compute gradient penalty
            GP = gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
            
            # Calculare the WD loss
            disc_loss = -torch.mean(  real_predict) + torch.mean(fake_predict) + lambda_gp * GP
            
            disc_loss.backward()
            optimizer_D.step()
            
            optimizer_G.zero_grad()
            
            #  Train Generator
            
            if i % critic == 0:
                # Generate a batch of images
                fake_imgs = generator(z)
                fake_predict = discriminator(fake_imgs)
                gen_loss = -torch.mean(fake_predict)
                gen_loss.backward()
                optimizer_G.step()
                
                if iteration%100==0 :
                    print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                          % (epoch, n_epochs, i, len(trainloader), disc_loss.item(), gen_loss.item())
                          )
                iteration += critic

################## Save weights
#    torch.save(generator.state_dict(), data_path+"weights/GAN_weights.h5")

################# Sampling 1000 images
    sample_size=1000
    z = Variable(Tensor(np.random.normal(0, 1, (sample_size, latent_dim))))
    final_sample = generator(z)
    for i in range(z.shape[0]):
        if (i)%100==0:
            print("%d th image saved"%i)
        save_image(final_sample.data[i], data_path+"final_sample/%s.png" % str(i+1), nrow=1, normalize=True)






