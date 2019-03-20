"""
Script contains eval/helper functions for both problems 4 and 5
"""

import torch 
import torch.nn as nn

import numpy as np
import math, copy, time
import matplotlib.pyplot as plt
import models as models


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

	with open(filepath) as fp:  
	   line = fp.readline()
	   cnt = 1
	   while line:
	       print("Line {}: {}".format(cnt, line.strip()))
	       line = fp.readline()
	       cnt += 1

	
