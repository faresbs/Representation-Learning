#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:20:15 2019

@author: chin-weihuang
"""

from __future__ import print_function
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from samplers import *

# plot p0 and p1
plt.figure()

# empirical
xx = torch.randn(10000)
f = lambda x: torch.tanh(x*2+1) + x*0.75
d = lambda x: (1-torch.tanh(x*2+1)**2)*2+0.75
plt.hist(f(xx), 100, alpha=0.5, density=1)
plt.hist(xx, 100, alpha=0.5, density=1)
plt.xlim(-5,5)


# exact
xx = np.linspace(-5,5,1000)
N = lambda x: np.exp(-x**2/2.)/((2*np.pi)**0.5)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.plot(xx, N(xx))


############### import the sampler ``samplers.distribution4''
############### train a discriminator on distribution4 and standard gaussian
############### estimate the density of distribution4

#######--- INSERT YOUR CODE BELOW ---#######

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class Net(nn.Module):
    def __init__(self, n_in=1, critic=False):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 128)
        self.out = nn.Linear(128, 1)
        if critic==True:
            self.out_act = Identity()
        else:
            self.out_act = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        x = self.out_act(x)
        return x


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

device = torch.device('cpu')

net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

n_iter = 5000
batch_size = 512

# P1.1 Testing loss on two gaussians
# mu1 = 100
# mu2 = 5
# sigma1 = 3
# sigma2 = 3
# net.train()
# for it in range(n_iter):
#     optimizer.zero_grad()
#     x = mu1 + sigma1*torch.randn(batch_size,2)
#     x = x.unsqueeze(1)
#     y = mu2 + sigma2*torch.randn(batch_size,2)
#     y = y.unsqueeze(1)
#     Dx = net(x)
#     Dy = net(y)
#     loss = -(math.log(2.) + (1/2.)*torch.mean(torch.log(Dx)) + (1/2.)*torch.mean(torch.log(1-Dy)))
#     loss.backward()
#     optimizer.step()
#     if it%10==0:
#         print('Iteration ' + str(it) + ' loss = ' + str(-loss.item()))

# Other option for implementing the loss
# loss_1 = nn.BCELoss()
#     # error_p = loss_1(Dx, torch.ones(batch_size,1))
#     # error_q = loss_1(Dy, torch.zeros(batch_size,1))
#     # loss = -math.log(2.) + (1/2.)*error_p + (1/2.)*error_q


# P1.3 JSD
n_phi_values = 21
p1 = torch.zeros(batch_size,1)
phi_values = np.linspace(-1.,1.,num=n_phi_values)
jsd_est = np.zeros(n_phi_values)
for i,phi in enumerate(phi_values):
    net = Net(n_in=2,critic=False).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
    net.train()
    for it in range(n_iter):
        optimizer.zero_grad()
        p = torch.cat((p1,torch.rand(batch_size,1)),1).to(device)
        q = torch.cat((phi*torch.ones(batch_size,1),torch.rand(batch_size,1)),1).to(device)
        Dp = net(p)
        Dq = net(q)
        loss = -(math.log(2.) + (1/2.)*torch.mean(torch.log(Dp)) + (1/2.)*torch.mean(torch.log(1-Dq)))
        loss.backward()
        optimizer.step()
        if it%100==0:
            print('Model ' + str(i+1) + ' Iteration ' + str(it) + ' loss = ' + str(-loss.item()))

    net.eval()
    p = torch.cat((p1,torch.rand(batch_size,1)),1).to(device)
    q = torch.cat((phi*torch.ones(batch_size,1),torch.rand(batch_size,1)),1).to(device)
    Dp = net(p)
    Dq = net(q)
    jsd_est[i]  = math.log(2.) + (1/2.)*torch.mean(torch.log(Dp)) + (1/2.)*torch.mean(torch.log(1-Dq))
    del net

plt.figure()
plt.plot(phi_values,jsd_est,'*')
plt.xlabel('Phi values')
plt.ylabel('JSD estimate')

# P1.3 WD
gp_coeff = 10
n_phi_values = 21
p1 = torch.zeros(batch_size,1)
phi_values = np.linspace(-1.,1.,num=n_phi_values)
wd_est = np.zeros(n_phi_values)
for i,phi in enumerate(phi_values):
    net = Net(n_in=2,critic=True).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=1e-3)
    net.train()
    for it in range(n_iter):
        optimizer.zero_grad()
        p = torch.cat((p1,torch.rand(batch_size,1)),1).to(device)
        q = torch.cat((phi*torch.ones(batch_size,1),torch.rand(batch_size,1)),1).to(device)
        Dp = net(p)
        Dq = net(q)
        #  gradient penalty
        a = torch.rand(batch_size, 1).expand(batch_size,2).to(device)
        r = a*p + (1-a)*q
        r.requires_grad = True
        Dr = net(r)
        gradients = torch.autograd.grad(outputs=Dr, inputs=r,
                                    grad_outputs=torch.ones(batch_size,1).to(device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
        loss = -(torch.mean(Dp) - torch.mean(Dq) - gp_coeff*torch.mean((gradients.norm(2, dim=1) - 1) ** 2))
        loss.backward()
        optimizer.step()
        if it%100==0:
            print('Model ' + str(i+1) + ' Iteration ' + str(it) + ' loss = ' + str(-loss.item()))

    net.eval()
    p = torch.cat((p1,torch.rand(batch_size,1)),1).to(device)
    q = torch.cat((phi*torch.ones(batch_size,1),torch.rand(batch_size,1)),1).to(device)
    Dp = net(p)
    Dq = net(q)
    #  gradient penalty
    a = torch.rand(batch_size, 1).expand(batch_size,2).to(device)
    r = a*p + (1-a)*q
    r.requires_grad = True
    Dr = net(r)
    gradients = torch.autograd.grad(outputs=Dr, inputs=r,
                                grad_outputs=torch.ones(batch_size,1).to(device),
                                create_graph=False, retain_graph=False, only_inputs=True)[0]
    wd_est[i] = torch.mean(Dp) - torch.mean(Dq) - gp_coeff*torch.mean((gradients.norm(2, dim=1) - 1) ** 2)
    del net

plt.figure()
plt.plot(phi_values,wd_est,'*')
plt.xlabel('Phi values')
plt.ylabel('WD estimate')




#  P1.4
#n_iter = 0
mu = 0
sigma = 1
f1 = distribution4(batch_size=1024)
f0 = distribution3(batch_size=1024)
net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)
net.train()
for it in range(n_iter):
    optimizer.zero_grad()
    y = torch.Tensor(next(f0)).to(device)
    x = torch.Tensor(next(f1)).to(device)
    # y = mu + sigma*torch.randn(batch_size,1).to(device)
    Dx = net(x)
    Dy = net(y)
    # loss = -(math.log(2.) + (1/2.)*torch.mean(torch.log(Dx)) + (1/2.)*torch.mean(torch.log(1-Dy)))
    loss = -(torch.mean(torch.log(Dx)) + torch.mean(torch.log(1-Dy)))
    loss.backward()
    optimizer.step()
    if it%100==0:
        print('Iteration ' + str(it) + ' loss = ' + str(-loss.item()))



############### plotting things
############### (1) plot the output of your trained discriminator
############### (2) plot the estimated density contrasted with the true density

# (1)
# xx = next(f0)
# r = net(torch.Tensor(xx).to(device)).data.numpy() # evaluate xx using your discriminator; replace xx with the output
r = net(torch.Tensor(xx).unsqueeze(1).to(device)).squeeze().data.numpy() # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')

# (2)
estimate = N(xx)*r/(1-r) # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')


plt.show()


# provided
r = xx # evaluate xx using your discriminator; replace xx with the output
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.plot(xx,r)
plt.title(r'$D(x)$')


estimate = np.ones_like(xx)*0.2 # estimate the density of distribution4 (on xx) using the discriminator;
                                # replace "np.ones_like(xx)*0." with your estimate
plt.subplot(1,2,2)
plt.plot(xx,estimate)
plt.plot(f(torch.from_numpy(xx)).numpy(), d(torch.from_numpy(xx)).numpy()**(-1)*N(xx))
plt.legend(['Estimated','True'])
plt.title('Estimated vs True')


plt.show()
