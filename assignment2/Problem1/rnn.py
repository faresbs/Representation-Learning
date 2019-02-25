"""
script for problem 1 : building an rnn from scratch using Pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

#For consistent random values
torch.manual_seed(0)

class RNN(nn.Module):
    def __init__(self, n_inputs, n_neurons):
        super(RNN, self).__init__()
        
        #Weights
        self.Wx = torch.randn(n_inputs, n_neurons)
        self.Wy = torch.randn(n_neurons, n_neurons)
        self.Wh = torch.randn(n_neurons, n_neurons)
        
        #Bias
        self.by = torch.zeros(1, n_neurons)
    	self.bh = torch.zeros(1, n_neurons)

    def forward(self, X, h0):
        self.y = torch.tanh(torch.mm(h0, self.Wy) + self.by)
        
        self.h = torch.tanh(torch.mm(X, self.Wx) +
                            torch.mm(h0, self.Wh) + self.bh)

        return self.y



N_INPUT = 3 # number of features in input
N_NEURONS = 5 # number of units in layer

X0_batch = torch.tensor([[0,1,2], [3,4,5], 
                         [6,7,8], [9,0,1]],
                        dtype = torch.float) #t=0 => 4 X 3

model = RNN(N_INPUT, N_NEURONS)

Y0_val = model(X0_batch, 0)

print Y0_val

print '-' * 100

rnn = nn.RNNCell(3, 5) # n_input X n_neurons

X_batch = torch.tensor([[0,1,2], [3,4,5], 
                         [6,7,8], [9,0,1]], dtype = torch.float) # X0 and X1

hx = torch.randn(4, 5) # m X n_neurons
output = []

# for each time step
for i in range(2):
    hx = rnn(X_batch[i], hx)
    output.append(hx)

print output



