"""
Script contains eval/helper functions for both problems 4 and 5
"""

import torch 
import torch.nn as nn

import numpy as np
import math, copy, time
import matplotlib.pyplot as plt
import models as models

#torch.manual_seed(1111)
np.random.seed(1111)

if __name__ == '__main__':

	##Problem 5
	#Generate samples using RNN
	dir = 'rnn/best_params.pt'

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print ("Running on " + str(device))


	print("RNN model loaded..")

	model = models.RNN(emb_size=200, hidden_size=1500, 
                seq_len=35, batch_size=20,
                vocab_size=10000, num_layers=2, 
                dp_keep_prob=0.35) 


	model.load_state_dict(torch.load(dir))

	#To remove the dropout 
	model.eval()

	#Size of vocabulary
	vocab = 10000

	#Sample of size batch_size from the vocab using a uniform distribution
	inp = np.random.choice(vocab, size=20, replace=True, p=None)
	#inp = torch.zeros(20)
	#print(inp)
	#Transform to tensor
	inp = torch.from_numpy(inp)

	#Initial hidden state is 0 ?
	#(num_layers, batch_size, hidden_size)
	hidden = torch.zeros(2, 20, 1500)

	#Stop sampling util we reach the seq length or generate an <eos> token
	generated_seq_len = 20

	#Generate samples with model
	samples = model.generate(inp, hidden, generated_seq_len)

	print("Sequence samples: ")

	#Map the samples sequences index into words sequences using vocab dictionary
	#TO DO
	d = {}
	with open("vocab.txt") as f:
	    for line in f:
	       (key, val) = line.split()
	       d[int(key)] = val

