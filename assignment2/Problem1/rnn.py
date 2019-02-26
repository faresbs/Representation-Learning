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
	def __init__(self, n_inputs, n_class, n_neurons):
		super(RNN, self).__init__()
		
		#Weights
		self.Wx = torch.randn(n_neurons, n_inputs)
		self.Wy = torch.randn(n_class, n_neurons)
		self.Wh = torch.randn(n_neurons, n_neurons)
		
		#Bias
		self.by = torch.zeros(n_class, 1)
		self.bh = torch.zeros(n_neurons, 1)

	def forward(self, X, h):

		self.h = torch.tanh(torch.mm(self.Wx, X) +
							torch.mm(self.Wh, h) + self.bh)

		self.y = torch.nn.Softmax(torch.mm(self.Wy, self.h) + self.by)
		

		return self.y



N_INPUT = 3 # number of features in input
N_NEURONS = 5 # number of units in layer
N_CLASS = 5 
N_BATCH = 3

X0_batch = torch.randn(N_INPUT, N_BATCH)

model = RNN(N_INPUT, N_CLASS, N_NEURONS)

h = np.zeros((N_NEURONS, N_BATCH))
h = torch.from_numpy(h).float()

Y0_val = model(X0_batch, h)

print (Y0_val)

print ('-' * 100)

sd

rnn = nn.RNNCell(3, 5) # n_input X n_neurons

hx = torch.randn(4, 5) # m X n_neurons
output = []

# for each time step
hx = rnn(X0_batch, hx)
output.append(hx)

print (output)



