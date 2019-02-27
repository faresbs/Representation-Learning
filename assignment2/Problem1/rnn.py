"""
script for problem 1 : building an rnn from scratch using Pytorch
"""

import torch 
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt

#For consistent random values
torch.manual_seed(0)

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is 
# what the main script expects. If you modify the contract, 
# you must justify that choice, note it in your report, and notify the TAs 
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which 
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention. 


def clones(module, N):
	"A helper function for producing N identical layers (each with their own parameters)."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
	"""
	emb_size:     The number of units in the input embeddings
	hidden_size:  The number of hidden units per layer
	seq_len:      The length of the input sequences
	vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
	num_layers:   The depth of the stack (i.e. the number of hidden layers at 
				  each time-step)
	dp_keep_prob: The probability of *not* dropping out units in the 
				  non-recurrent connections.
				  Do not apply dropout on recurrent connections.
	"""
	super(RNN, self).__init__()

	# TODO ========================
	# Initialization of the parameters of the recurrent and fc layers. 
	# Your implementation should support any number of stacked hidden layers 
	# (specified by num_layers), use an input embedding layer, and include fully
	# connected layers with dropout after each recurrent layer.
	# Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding 
	# modules, but not recurrent modules.
	#
	# To create a variable number of parameter tensors and/or nn.Modules 
	# (for the stacked hidden layer), you may need to use nn.ModuleList or the 
	# provided clones function (as opposed to a regular python list), in order 
	# for Pytorch to recognize these parameters as belonging to this nn.Module 
	# and compute their gradients automatically. You're not obligated to use the
	# provided clones function.

	#TO CHECK:
	#is this what he mean by Initialization of the parameters of the recurrent and fc layers ??

	self.emb_size = emb_size   
	self.hidden_size = hidden_size
	self.seq_len = seq_len
	self.vocab_size = vocab_size
	self.num_layers = num_layers
	self.dp_keep_prob = dp_keep_prob

  def init_weights_uniform(self):
	# TODO ========================
	# Initialize all the weights uniformly in the range [-0.1, 0.1]
	# and all the biases to 0 (in place)

	#Weights
	#TO CHECK: 
	#Wx = (hidden_size,emb_size) if k = 0 and (hidden_size,hidden_size) otherwise ?
	#DO we only take h0 in the other layers?
	self.Wx = (-0.1 - 0.1) * torch.rand(self.hidden_size, self.emb_size) + 0.1
	self.Wh = (-0.1 - 0.1) * torch.rand(self.hidden_size,self.hidden_size) + 0.1
	self.Wy = (-0.1 - 0.1) * torch.rand(self.Wy.shape) + 0.1

	#Bias
	self.by = torch.zeros(vocab_size, 1)
	self.bh = torch.zeros(hidden_size, 1)
	self.bx = torch.zeros(hidden_size, 1)

	#TO CHECK:
	#how do we first init the weights and then add that to modulesList?
	#difference between doing moduleList and nn.sequential ??
	
	#TO CHECK:
	#can we and how do we use nn.linear after weights init and how do we add them to modulesList?

	module = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
	self.network = self.clones(module, self.num_layers)

	network.apply(init_weights)
	
	
  def init_hidden(self):
	# TODO ========================
	# initialize the hidden states to zero
	"""
	This is used for the first mini-batch in an epoch, only.
	"""
	h0 = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)

	return h0 # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)

  def forward(self, inputs, hidden):
	# TODO ========================
	# Compute the forward pass, using a nested python for loops.
	# The outer for loop should iterate over timesteps, and the 
	# inner for loop should iterate over hidden layers of the stack. 
	# 
	# Within these for loops, use the parameter tensors and/or nn.modules you 
	# created in __init__ to compute the recurrent updates according to the 
	# equations provided in the .tex of the assignment.
	#

#TO CHECK/:
#the hidden states of the l-th layer are used as inputs to to the {l+1} ?

	# Note that those equations are for a single hidden-layer RNN, not a stacked
	# RNN. For a stacked RNN, the hidden states of the l-th layer are used as 
	# inputs to to the {l+1}-st layer (taking the place of the input sequence).

	"""
	Arguments:
		- inputs: A mini-batch of input sequences, composed of integers that 
					represent the index of the current token(s) in the vocabulary.
						shape: (seq_len, batch_size)
		- hidden: The initial hidden states for every layer of the stacked RNN.
						shape: (num_layers, batch_size, hidden_size)
	
	Returns:
		- Logits for the softmax over output tokens at every time-step.
			  **Do NOT apply softmax to the outputs!**
			  Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does 
			  this computation implicitly.
					shape: (seq_len, batch_size, vocab_size)
		- The final hidden states for every layer of the stacked RNN.
			  These will be used as the initial hidden states for all the 
			  mini-batches in an epoch, except for the first, where the return 
			  value of self.init_hidden will be used.
			  See the repackage_hiddens function in ptb-lm.py for more details, 
			  if you are curious.
					shape: (num_layers, batch_size, hidden_size)
	"""
	return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden

  def generate(self, input, hidden, generated_seq_len):
	# TODO ========================
	# Compute the forward pass, as in the self.forward method (above).
	# You'll probably want to copy substantial portions of that code here.
	# 
	# We "seed" the generation by providing the first inputs.
	# Subsequent inputs are generated by sampling from the output distribution, 
	# as described in the tex (Problem 5.3)
	# Unlike for self.forward, you WILL need to apply the softmax activation 
	# function here in order to compute the parameters of the categorical 
	# distributions to be sampled from at each time-step.

	"""
	Arguments:
		- input: A mini-batch of input tokens (NOT sequences!)
						shape: (batch_size)
		- hidden: The initial hidden states for every layer of the stacked RNN.
						shape: (num_layers, batch_size, hidden_size)
		- generated_seq_len: The length of the sequence to generate.
					   Note that this can be different than the length used 
					   for training (self.seq_len)
	Returns:
		- Sampled sequences of tokens
					shape: (generated_seq_len, batch_size)
	"""
   
	return samples






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



