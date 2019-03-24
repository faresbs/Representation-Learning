"""
Script to generate samples using RNN and GRU
"""

import torch 
import torch.nn as nn

import numpy as np
import math, copy, time
import matplotlib.pyplot as plt
import models as models

import collections

from gtts import gTTS
import os

#torch.manual_seed(1111)
np.random.seed(1111)


#Build our vocabulary from index values
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()


def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word



if __name__ == '__main__':

	#Google Text-to-Speech
	speak = True

	#Create folder to save audio
	if not os.path.exists('audio'):
		os.mkdir('audio')

	# Device configuration
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	print ("Running on " + str(device))

	model_type = 'GRU'

	if(model_type=='RNN'):

		#Generate samples using RNN
		dir = 'models/rnn/best_params.pt'

		print("RNN model loaded.")

		model = models.RNN(emb_size=200, hidden_size=1500, 
	                seq_len=35, batch_size=20,
	                vocab_size=10000, num_layers=2, 
	                dp_keep_prob=0.35) 

	elif(model_type=='GRU'):

		#Generate samples using RNN
		dir = 'models/gru/best_params.pt'

		print("GRU model loaded.")

		model = models.GRU(emb_size=200, hidden_size=1500, 
	                seq_len=35, batch_size=20,
	                vocab_size=10000, num_layers=2, 
	                dp_keep_prob=0.35) 



	model.load_state_dict(torch.load(dir))

	#To remove the dropout 
	model.eval()

	#Size of vocabulary
	vocab = 10000

	#Sample of size batch_size from the vocab using a uniform distribution
	#Take a random word
	inp = np.random.choice(vocab, size=20, replace=True, p=None)
	#inp = np.zeros(20, dtype=int)
	#print(inp)
	#Transform to tensor
	inp = torch.from_numpy(inp)

	#Initial hidden state is 0 
	#(num_layers, batch_size, hidden_size)
	hidden = torch.zeros(2, 20, 1500)

	#Stop sampling util we reach the seq length or generate an <eos> token
	generated_seq_len = 35

	#in softmax e^(x/temp) for more coherent results
	temp = 0.5

	#Generate samples with model
	# samples = (generated_seq_len, batch_size)
	samples = model.generate(inp, hidden, generated_seq_len, temp)

	print("Creating samples..")

	train_path = 'data/ptb.train.txt'

	#print('Building vocaulary..')
	_, id_to_word = _build_vocab(train_path)

	samples = samples.numpy().astype(int)

	array = []
	a = []

	#Loop over examples
	for s in range(samples.shape[1]):
		#Loop over words
		for w in range(samples.shape[0]):
			array.append(id_to_word[samples[w, s]])


		if(speak):

			#Remove unwanted tokens
			while (array.count('<unk>')): 
				array.remove('<unk>')

			while (array.count('<eos>')): 
				array.remove('<eos>')

			#Transform the last sentence to voice
			text1 = ' '.join(array)
			print (text1)
			tts = gTTS(text=text1, lang='en')	
			tts.save('audio/sentence'+str(s+1)+'.mp3')

		a.append(array)
		array = []
	
		
	if(model_type=='RNN'):
		np.savetxt('samples_rnn.txt', a,  newline='\n', fmt="%s")

	elif(model_type=='GRU'):
		np.savetxt('samples_gru.txt', a,  newline='\n', fmt="%s")
		






