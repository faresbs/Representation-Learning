import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp

import mnist_loader
import time


##This code was inspired from:
#https://github.com/pytorch/examples/blob/master/vae/main.py


class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		##Encoder
		conv_e = nn.Sequential(
				nn.Conv2d(1, 32, kernel_size=3, bias=True, stride=1),
				nn.ELU(),
				nn.AvgPool2d(kernel_size=2, stride=2),
				nn.Conv2d(32, 64, kernel_size=3, bias=True, stride=1),
				nn.ELU(),
				nn.AvgPool2d(kernel_size=2, stride=2),
				nn.Conv2d(64, 256, kernel_size=5, bias=True, stride=1),
				nn.ELU(),
				
			)

		linear_e = nn.Linear(256, 100*2)

		self.encoder = nn.ModuleList([conv_e, linear_e])

		##Decoder
		conv_d = nn.Sequential(
		#Takes z latent variable of size 100
				
				nn.Conv2d(256, 64, kernel_size=5, bias=True, stride=1, padding=4),
				nn.ELU(),
				nn.UpsamplingBilinear2d(size=None, scale_factor=2),
				nn.Conv2d(64, 32, kernel_size=3, bias=True, stride=1, padding=2),
				nn.ELU(),
				nn.UpsamplingBilinear2d(size=None, scale_factor=2),
				nn.Conv2d(32, 16, kernel_size=3, bias=True, stride=1, padding=2),
				nn.ELU(),
				nn.Conv2d(16, 1, kernel_size=3, bias=True, stride=1, padding=2)

			)

		linear_d = nn.Linear(100, 256)
		relu = nn.ELU()

		self.decoder = nn.ModuleList([linear_d, relu, conv_d])

	#Outputs mean/log-variance
	def encode(self, x):
		z = self.encoder[0](x)
		#Reshape for FC
		z = z.view(z.size(0), -1)

		#Outputs 2 vectors of size 100, mean vector and std vector
		z = self.encoder[1](z)

		#first 100 for mean vector, the other 100 for logvar
		return z[:, :100], z[:, 100:]

	#Outputs reconstructed x
	def decode(self, z):
		z = self.decoder[0](z)
		z = self.decoder[1](z)

		#Get dim of z to know if we are processing batches or 1 example
		dim_z = len(z.shape)

		if(dim_z == 1):
			#Reshape z from 1 dim to 4 dim	
			z = z.view(1, z.shape[0], 1, 1)
		else:
			#Reshape z from 2 dim to 4 dim
			z = z.view(z.shape[0], z.shape[1], 1, 1)
		
		recon_x = self.decoder[2](z)
		#Squish the output values between 0 and 1 for BCE
		recon_x = torch.sigmoid(recon_x)
		return recon_x

	#Sampling by re-perameterization trick
	def reparameterize(self, mu, logvar):
		#Need std for normal distribution
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		z = mu + eps*std
		return z


	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		#z = (batch,latent_space)
		recon_x = self.decode(z)
		
		return recon_x, z, mu, logvar



# Reconstruction + KL divergence losses summed over all elements of batch
def loss_elbo(recon_x, x, mu, logvar):
	
	marginal_likelihood = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')
	#marginal_likelihood = x.view(-1, 784) * torch.log(recon_x.view(-1, 784)) + (1.0-x.view(-1, 784)) * torch.log(1-recon_x.view(-1, 784))
	#print(marginal_likelihood.shape)
	#sd
	#Note: you can compute this using logvar or std
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
	#From paper

	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	loss = marginal_likelihood + KLD

	#ELBO = -loss
	
	return loss



def train(epoch, train_loader):
	#Mode train
	model.train()

	train_loss = 0

	for i, inputs in enumerate(train_loader):

		inputs = inputs.to(device)
		optimizer.zero_grad()
		recon_batch, z, mu, logvar = model(inputs)
		loss = loss_elbo(recon_batch, inputs, mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()

		if i % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, i * len(inputs), len(train_loader.dataset),
				100. * i / len(train_loader),
				loss.item() / len(inputs)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(train_loader.dataset)))


def eval(epoch, dataset, loader, eval_method='elbo'):
	#Mode eval
	model.eval()

	epoch_loss = 0

	with torch.no_grad():

		for i, inputs in enumerate(loader):
			#print(i)
			inputs = inputs.to(device)
			recon_batch, z, mu, logvar = model(inputs)
			
			if(eval_method=='elbo'):
				epoch_loss += loss_elbo(recon_batch, inputs, mu, logvar).item()

			else:
				#Draw K samples from the encoder
				K = 200
				z = torch.zeros([inputs.shape[0], K, 100]).to(device)
				for i in range(K):
					z[:,i,:] = model.reparameterize(mu, logvar)
				
				epoch_loss += np.sum(loss_IS(model, inputs, z))
				
	
	if(dataset=='val'):
		epoch_loss /= len(loader.dataset)
		print('====> Validation: {:.4f}'.format(epoch_loss))

	elif(dataset=='test'):
		epoch_loss /= len(loader.dataset)
		print('====> Test: {:.4f}'.format(epoch_loss))


##Evaluate VAE using importance sampling 
#Q2.1

def loss_IS(model, true_x, z):

	#Loop over the elements i of batch
	M = true_x.shape[0]

	#Save logp(x)
	logp_x = np.zeros([M])

	#Get mean and std from encoder
	#2 Vectors of 100
	mu, logvar = model.encode(true_x.to(device))
	std = torch.exp(0.5*logvar)

	K = 200

	#Loop over tha batch
	for i in range(M):
		#z_ik
		samples = z[i,:,:]

		#Compute the reconstructed x's from sampled z's
		x = model.decode(samples.to(device))	

		#Compute the p(x_i|z_ik) of x sampled from z_ik
		#Bernoulli dist = Apply BCE
		#Output an array of losses

		true_xi = true_x[i,:,:].view(-1, 784)
		x = x.view(-1, 784)
		
		#P(x¦z) is shape (200, 784)
		p_xz = -(true_xi * torch.log(x) + (1.0-true_xi) * torch.log(1.0-x))
		
		#Multiply the independent probabilities (784)	
		#Apply logsumexp to avoid small image prob
		#P(x¦z) is shape (200, 784)
		p_xz = logsumexp(p_xz.cpu().numpy(), axis=1)

	
		##q(z_ik¦x_i) follows a normal dist
		#q_z = mgd(samples, mu, std)
		s = std[i, :].view([std.shape[1]])
		m = mu[i, :].view([std.shape[1]])

		q_z = multivariate_normal.pdf(samples.cpu().numpy(),mean=m.cpu().numpy(), cov=np.diag(s.cpu().numpy()**2))

		
		##p(z_ik) follows a normal dist with mean 0/variance 1
		#Normally distributed with loc=0 and scale=1
		std_1 = torch.ones(samples.shape[1])
		mu_0 = torch.zeros(samples.shape[1])	

		p_z = multivariate_normal.pdf(samples.cpu().numpy(),mean=mu_0.cpu().numpy(), cov=np.diag(std_1.cpu().numpy()**2))

		#Multiply the probablities
		
		#logp_x[i] = np.log((1.0/K) * np.sum(np.exp(np.log(p_xz) + np.log(p_z) - np.log(q_z))))
		#logp_x[i] = np.log((1.0/K) * np.sum(np.exp(p_xz.cpu().numpy() + np.log(p_z) - np.log(q_z))))
		logp_x[i] = np.log((1.0/K) * np.sum((p_xz * p_z)/q_z))

	return logp_x

		

if __name__ == "__main__":

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = VAE().to(device)
	optimizer = optim.Adam(model.parameters(), lr=3e-4)

	###Training

	#n_epochs = 20

	#Load data loaders
	train_loader, val_loader, test_loader = mnist_loader.get_data_loader("binarized_mnist", 64)

	#Train + val
	#for epoch in range(n_epochs):
	#	train(epoch, train_loader)
		#eval(epoch, val_loader)

	#	with torch.no_grad():
			#Generate a batch of images using current parameters 
	#		sample = torch.randn(64, 100).to(device)
	#		sample = model.decode(sample).cpu()
	#		save_image(sample.view(64, 1, 28, 28),
	#				   'results/sample_' + str(epoch) + '.png')


	#Saving the model weights
	#torch.save(model.state_dict(), 'weights/weights.h5')

	###Evaluating (Q2.2)

	path_weights = 'weights/weights.h5'

	model.load_state_dict(torch.load(path_weights))
	print("Model successfully loaded")

	#put the model in eval mode
	model = model.eval()

	eval_method = ['elbo', 'IS']
	dataset = ['val', 'test']

	with torch.no_grad():

		for method in eval_method:
			for data in dataset:

				print('Evaluating using '+method+'...')

				if(data == 'val'):
					eval(0, data, val_loader, eval_method=method)

				if(data == 'test'):
					eval(0, data, test_loader, eval_method=method)



	










