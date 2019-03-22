"""
Script contains eval/helper functions for both problems 4 and 5
"""

import torch
import torch.nn as nn

import numpy as np
import math, copy, time
import matplotlib.pyplot as plt
import models as models

def loss_per_tstep(file_path):
	L = np.load(file_path + '/loss_timestep_minibatch.npy')
	avg_L = np.mean(L,axis = 0)
	plt.figure()
	plt.plot(avg_L)
	plt.ylabel('loss')
	plt.xlabel('time-step (t)')
	plt.savefig(file_path + '/loss_t-step.png')



#Display learning curve
def learning_curve(train_ppl, val_ppl, time, file, save_path):

	epochs = range(len(train_acc))

	plt.figure()

	plt.plot(epochs, train_acc, 'b', label='Training acc')
	plt.plot(epochs, val_acc, 'g', label='Validation acc')
	plt.title('Training and validation accuracy')
	plt.xlabel('epoch')
	plt.legend()
	plt.savefig(path+"accuracy.png")

	plt.figure()

	plt.plot(epochs, train_loss, 'b', label='Training loss')
	plt.plot(epochs, val_loss, 'g', label='Validation loss')
	plt.title('Training and validation loss')
	plt.xlabel('epoch')
	plt.legend()
	plt.savefig(path+"loss.png")




if __name__ == '__main__':

	#filepath = 'models/rnn/learning_curves.npy'
	#file = 'models/gru/learning_curves.npy'
	#file = np.load(file)
	filepath = 'models/rnn/log.txt'
	# file_path = 'RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=1_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0'
	# loss_per_tstep(file_path)

	with open(filepath) as fp:
	   line = fp.readline()
	   cnt = 1
	   while line:
	       print("Line {}: {}".format(cnt, line.strip()))
	       line = fp.readline()
	       cnt += 1
