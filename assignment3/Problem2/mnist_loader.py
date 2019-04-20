from torchvision.datasets import utils
import torch.utils.data as data_utils
import torch
import os
import numpy as np
from torch import nn
from torch.nn.modules import upsampling
#from torch.functional import F
from torch.optim import Adam

import matplotlib
import matplotlib.pyplot as plt

import wget
import os  



def get_data_loader(dataset_location, batch_size):
	URL = "http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/"
	# start processing
	def lines_to_np_array(lines):
		return np.array([[int(i) for i in line.split()] for line in lines])
		
	splitdata = []
	for splitname in ["train", "valid", "test"]:
		filename = "binarized_mnist_%s.amat" % splitname
		filepath = os.path.join(dataset_location, filename)
		#utils.download_url(URL + filename, dataset_location)

		#Check if file exists, so that we dont download
		if (os.path.exists(dataset_location+'/'+filename)):
			print('already downloaded !')
		else:
			print('downloading..')	
			wget.download(URL + filename, dataset_location+'/'+filename)

		with open(filepath) as f:
			lines = f.readlines()
		x = lines_to_np_array(lines).astype('float32')
		x = x.reshape(x.shape[0], 1, 28, 28)
		# pytorch data loader
		dataset = data_utils.TensorDataset(torch.from_numpy(x))
		dataset_loader = data_utils.DataLoader(x, batch_size=batch_size, shuffle=splitname == "train")
		splitdata.append(dataset_loader)

	return splitdata

"""
if __name__ == "__main__":

	train, valid, test = get_data_loader("binarized_mnist", 64)

	for x in train:
		print(x.shape)
		#x = x.permute(0,3,1,2)
		#x = x[0,0].unsqueeze(2)
		x = x[0,0]
		#a = torch.Tensor(x.shape[0], x.shape[1], 3)
		
		#for i in range(3):
		#	a[:,:,i] = x

		#print(a.shape)
		plt.imshow(x)
		plt.show()
		break
		
"""

