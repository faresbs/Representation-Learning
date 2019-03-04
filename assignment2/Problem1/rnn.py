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


#A helper function for producing N identical layers (each with their own parameters).
def clones(module, N):
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
		self.batch_size = batch_size
		self.hidden_size = hidden_size
		self.seq_len = seq_len
		self.vocab_size = vocab_size
		self.num_layers = num_layers
		self.dp_keep_prob = dp_keep_prob

		###Create modules###

		#embeddings layer
		self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.emb_size)

		#(hidden_size * emb_size + hidden_size) for first layer
		self.f_layer = nn.Linear(self.emb_size + self.hidden_size, self.hidden_size, bias=True)

		#(hidden_size * hidden_size + hidden_size) for hidden layers
		#rec_layer = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size, bias=True)

		#Softmax to calculate probabilities after every output layer
		#self.softmax = nn.Softmax()

		#TO CHECK: fc after each recurrent layer?

		#Create module
		module = nn.Sequential(
							#(hidden_size * hidden_size + hidden_size) for hidden layers
							nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size, bias=True), 
							nn.Tanh(),
							nn.Dropout(dp_keep_prob),
							nn.Linear(self.hidden_size, self.hidden_size, bias=True),
							nn.Tanh(),
							)

		
		#Create k hidden layers for 1 time step
		self.network = clones(module, self.num_layers-1)

		#Output layer (logits)
		self.logit = nn.Linear(self.hidden_size, self.vocab_size, bias=True)


		"""
		#Create the recurrent layers
		rec_module = nn.Sequential(self.rec_layer, nn.Tanh())
		self.network = clones(rec_module, self.num_layers)

		#Add fc layers
		fc_module = nn.Sequential(
							nn.Dropout(dp_keep_prob),
							nn.Linear(self.hidden_size, self.hidden_size, bias=True),
							nn.Tanh(),
							)

		fc_modules = clones(fc_module, self.num_layers)

		self.network.append(fc_modules)
		"""
		#Add out layer in the end
		#self.network.append(self.out)
		
		#Add input recurrent layer at first
		#self.network = nn.ModuleList([self.i_rec, *self.network])
		#print(self.network)
			

	def init_weights_uniform(self):
		# TODO ========================
		# Initialize all the weights uniformly in the range [-0.1, 0.1]
		# and all the biases to 0 (in place)

		#Weights
		#TO CHECK: 
		#Wx = (hidden_size, emb_size) if k = 0 and (hidden_size,hidden_size) otherwise ?
		#DO we only take h0 in the other layers?
		
		#Wx = (hidden_size, emb_size) for the first layer
		#self.Wx = (-0.1 - 0.1) * torch.rand(self.hidden_size, self.emb_size) + 0.1
		self.Wx = torch.Tensor(self.hidden_size, self.emb_size).uniform_(-0.1, 0.1)
		
		#Whh = (hidden_size, hidden_size) for the other layers
		self.Whh = torch.Tensor(self.hidden_size, self.hidden_size).uniform_(-0.1, 0.1)

		## NOTE: linear will already have randomly init weights then if you init another time with init.uniform you get different random values
		#linear_X = nn.Linear(self.emb_size, self.hidden_size, bias=False)
		#torch.nn.init.uniform_(linear_X.weight, -0.1, 0.1)

		#Wh = (hidden_size, hidden_size)
		#self.Wh = (-0.1 - 0.1) * torch.rand(self.hidden_size,self.hidden_size) + 0.1
		self.Wh = torch.Tensor(self.hidden_size, self.hidden_size).uniform_(-0.1, 0.1)

		#Wy = (vocab_size, hidden_size)	
		#self.Wy = (-0.1 - 0.1) * torch.rand(self.vocab_size,self.hidden_size) + 0.1
		self.Wy = torch.Tensor(self.vocab_size,self.hidden_size).uniform_(-0.1, 0.1)

		#Combine the two weights Wx and Wh in combined = [Wh, Wx] in the case of the first layer
		self.i_combined = torch.cat((self.Wh, self.Wx), 1)

		#Combine the two weights Whh and Wh in combined = [Wh, Whh] in the case of the other layers
		self.h_combined = torch.cat((self.Wh, self.Whh), 1)

		#Bias
		self.by = torch.zeros(self.vocab_size)
		self.bh = torch.zeros(self.hidden_size)
		#self.bx = torch.zeros(self.hidden_size, 1)

		#TO CHECK:
		#how do we first init the weights and then add that to modulesList?
		#difference between doing moduleList and nn.sequential ??
		
		#TO CHECK:
		#can we and how do we use nn.linear after weights init and how do we add them to modulesList?


		#Fill the linear layers with init weights and biases
		#To avoid problems of grads
		with torch.no_grad():

			#print(self.combined.shape)
			#print(self.recurrent.weight.shape)

			self.f_layer.weight.copy_(self.i_combined)
			self.f_layer.bias.copy_(self.bh)

			#loop to copy all weights?
			#If we have many layers 
			if(self.num_layers > 1):

				for layer in self.network:
					layer[0].weight.copy_(self.h_combined)
					layer[0].bias.copy_(self.bh)

			self.logit.weight.copy_(self.Wy)
			self.logit.bias.copy_(self.by)


		#TO CHECK: sequential vs modules
		#module = linear
		#module.add_module("softmax", linear_out)
		
	
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

		#Init weights and biases
		self.init_weights_uniform()

		#print(self.network)

		#TO CHECK: embedding all X (seqlen, batch) vs embeddings X each time step (batch) is the same? 

		#Save logits
		logits = torch.zeros(self.seq_len, self.batch_size, self.vocab_size)

		#TO DO: take the hidden state for previous step


		#Loop over the timesteps
		for step in range(0, self.seq_len):

			##Go through the first layer first

			#take embeddings of current timestep
			#(batch_size)
			inp = inputs[step]

			#Take initial hidden state for first layer
			h = hidden[0]

			# Lookup word embeddings
			#(batch_size, enb_size)
			emb = self.embeddings(inp)

			#Combine embeddings and last hidden state
			combined = torch.cat((h, emb), 1)

			out = self.f_layer(combined)

			#save final hidden state for next layer
			hidden[0] = out

			#Loop over the rest of the layers
			for layer in range(self.num_layers-1):
				
				#Take the hidden state of the current layer
				h = hidden[layer]

				#Combine hidden state of l-th layer and current hidden state
				combined = torch.cat((h, out), 1)

				out = self.network[layer](combined)

				#save final hidden state for next layer
				hidden[layer] = out


			#last layer to calculate the logits
			#(seq_len, batch_size, vocab_size)
			logits[step] = self.logit(out)
		#print(logits.shape)
		return logits.view(self.seq_len, self.batch_size, self.vocab_size)


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


rnn = RNN(emb_size=200, hidden_size=100, seq_len=35, batch_size=20, vocab_size=10000, num_layers=2, dp_keep_prob=0.35)

#input = (batch, timesteps)
x = torch.Tensor(35, 20).uniform_(1, 10000)
x = x.type(torch.LongTensor)

h0 = rnn.init_hidden()

rnn.forward(x, h0)

"""
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
"""





















