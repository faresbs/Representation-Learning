"""
Script contains eval/helper functions for both problems 4 and 5
"""

import torch
import torch.nn as nn

import numpy as np
import math, copy, time
import matplotlib.pyplot as plt
import models as models

def plot_grads(file_path_RNN,file_path_GRU):
	grads_RNN = np.load(file_path_RNN + '/gradsL_wrt_ht.npy')
	grads_RNN = (grads_RNN - np.min(grads_RNN))/(np.max(grads_RNN) - np.min(grads_RNN))
	grads_GRU = np.load(file_path_GRU + '/gradsL_wrt_ht.npy')
	grads_GRU = (grads_GRU - np.min(grads_GRU))/(np.max(grads_GRU) - np.min(grads_GRU))

	plt.figure(figsize=(10, 6))
	plt.plot(grads_RNN,label='RNN')
	plt.plot(grads_GRU,label='GRU')
	plt.title('Norm of the avg. gradient of the loss at the final time-step ht')
	plt.ylabel('Norm')
	plt.xlabel('Time-step')
	plt.legend()
	plt.savefig(file_path_RNN + '/gradsL_wrt_ht.png')
	plt.savefig(file_path_GRU + '/gradsL_wrt_ht.png')

def loss_per_tstep(file_path_RNN,file_path_GRU,file_path_Trans):
	L_RNN = np.load(file_path_RNN + '/loss_timestep_minibatch.npy')
	avg_L_RNN = np.mean(L_RNN,axis = 0)
	L_GRU = np.load(file_path_GRU + '/loss_timestep_minibatch.npy')
	avg_L_GRU = np.mean(L_GRU,axis = 0)
	L_Trans = np.load(file_path_Trans + '/loss_timestep_minibatch.npy')
	avg_L_Trans = np.mean(L_Trans,axis = 0)
	plt.figure(figsize=(10, 6))
	plt.plot(avg_L_RNN,label='RNN')
	plt.plot(avg_L_GRU,label='GRU')
	plt.plot(avg_L_Trans,label='Transformer')
	plt.title('Average loss at each time-step on the validation data')
	plt.ylabel('Loss')
	plt.xlabel('Time-step (t)')
	plt.legend()
	plt.savefig(file_path_RNN + '/loss_t-step.png')
	plt.savefig(file_path_GRU + '/loss_t-step.png')
	plt.savefig(file_path_Trans + '/loss_t-step.png')



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
	# filepath = 'models/rnn/log.txt'
	# file_path_RNN = 'RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=50_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_0'
	# file_path_RNN = 'q5_1_RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0003_batch_size=32_seq_len=35_hidden_size=1500_num_layers=3_dp_keep_prob=0.45_save_best_0'
	# file_path_GRU = 'q5_1_GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=12_batch_size=32_seq_len=35_hidden_size=1024_num_layers=2_dp_keep_prob=0.4_save_best_0'
	# file_path_Trans = 'q5_1_TRANSFORMER_ADAM_model=TRANSFORMER_optimizer=ADAM_initial_lr=0.0001_batch_size=128_seq_len=35_hidden_size=1024_num_layers=6_dp_keep_prob=0.9_save_best_save_dir=q5_1__0'

	file_path_GRU = 'q5_2_GRU_SGD_LR_SCHEDULE_model=GRU_optimizer=SGD_LR_SCHEDULE_initial_lr=12_batch_size=32_seq_len=35_hidden_size=1024_num_layers=2_dp_keep_prob=0.4_save_best_save_dir=q5_2__0'
	file_path_RNN = 'q5_2_RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0003_batch_size=32_seq_len=35_hidden_size=1500_num_layers=3_dp_keep_prob=0.45_save_best_save_dir=q5_2__0'

	# loss_per_tstep(file_path_RNN,file_path_GRU,file_path_Trans)
	plot_grads(file_path_RNN,file_path_GRU)

	# with open(filepath) as fp:
	#    line = fp.readline()
	#    cnt = 1
	#    while line:
	#        print("Line {}: {}".format(cnt, line.strip()))
	#        line = fp.readline()
	#        cnt += 1
