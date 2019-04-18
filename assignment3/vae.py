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
		#z = self.linear_d(z)
		z = self.decoder[0](z)
		#z = self.relu(z)
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

	#Note: you can compute this using logvar or std
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	loss = marginal_likelihood + KLD

	#Normalize
	#D_train = logvar.shape[0]
	#loss /= D_train

	ELBO = -loss
	
	return ELBO



def train(epoch, train_loader):
	#Mode train
	model.train()

	train_loss = 0

	for i, inputs in enumerate(train_loader):

		inputs = inputs.to(device)
		optimizer.zero_grad()
		recon_batch, z, mu, logvar = model(inputs)
		loss = loss_elbo(recon_batch, z, inputs, mu, logvar)
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

		for i, inputs in enumerate(test_loader):
			print(i)
			inputs = inputs.to(device)
			recon_batch, z, mu, logvar = model(inputs)
			
			if(eval_method=='elbo'):
				epoch_loss += loss_elbo(recon_batch, inputs, mu, logvar).item()
			
			else:
				
				#Draw K samples from the encoder
				K = 200
				z = torch.zeros([64, K, 100]).to(device)
				for i in range(K):
					z[:,i,:] = model.reparameterize(mu, logvar)

				#Average log-likelihoods within the batch
				print(np.average(loss_IS(model, inputs, z).numpy()))
				epoch_loss += np.average(loss_IS(model, inputs, z).item())

	
	if(dataset=='val'):
		epoch_loss /= len(test_loader.dataset)
		print('====> Validation: {:.4f}'.format(epoch_loss))

	elif(dataset=='test'):
		epoch_loss /= len(test_loader.dataset)
		print('====> Test: {:.4f}'.format(epoch_loss))

	


#Helper function to compute the pdf of mgd
#Multivariate gaussian distribution
#q(z¦x) = N(z;mu,std) where mu,std are given by the encoder network
#P(z) = N(Z;0,1)
def mgd(z, mean, std):
	#Compute covariance from observations z_ik

	std = std.squeeze()
	mean = mean.squeeze()

	#In case of 1 sample
	if(len(mean.shape)==1):
		#Covariance is the diag()
		cov = np.diag(std.cpu().numpy()**2)
		#Compute pdf
		p = multivariate_normal.pdf(z.cpu().numpy(),mean=mean.cpu().numpy(), cov=cov)

		#p = (1/np.sqrt((2*np.pi).pow(mean.shape[0])*np.linalg.det(cov)))np.exp()

		return p 

	#In case of batch of samples
	else:
		m = mean.shape[0]
		p = np.zeros((m))

		for i in range(m):
			s = std[i, :].view([std.shape[1]])
			m = mean[i, :].view([std.shape[1]])
			z_i = z[i, :].view([std.shape[1]])

			p[i] = multivariate_normal.pdf(z_i.cpu().numpy(),mean=m.cpu().numpy(), cov=np.diag(s.cpu().numpy()**2))

		return p


##Evaluate VAE using importance sampling 
#Q2.1

def loss_IS2(model, true_x, z):

	#Loop over the elements i of batch
	m = true_x.shape[0]

	#Save logp(x)
	logp_x = np.zeros([m])

	for k in range(z.shape[1]):
		print(k)
		#z_ik
		samples = z[:,k,:]

		#Compute the reconstructed x's from sampled z's
		x = model.decode(samples.to(device))

		#Compute the p(x_i|z_ik) of x sampled from z_ik
		#Bernoulli dist = Apply BCE
		#Output an array of losses
		loss = nn.BCELoss(reduction='mean')
		p_x = torch.zeros([m])
		
		#Loop over batch to compute the BCE for each
		for i in range(m) :
			true_xi = true_x[i,:,:].view(1,1,true_x.shape[2],true_x.shape[3])
			xi = x[i,:,:].view(1,1,true_x.shape[2],true_x.shape[3])
			p_x[i] = loss(xi.view(-1, 784), true_xi.view(-1, 784))
		
		#print(p_x.cpu().numpy().shape)	
		
		##q(z_ik¦x_i) follows a normal dist

		#Get mean and std from encoder
		#2 Vectors of 100
		mu, logvar = model.encode(true_x.to(device))
		std = torch.exp(0.5*logvar)
		q_z = mgd(samples, mu, std)
		#print(q_z.shape)

		##p(z_ik) follows a normal dist with mean 0/variance 1
		#(64, 100)	
		#Normally distributed with loc=0 and scale=1
		std = torch.ones(samples.shape)
		mu = torch.zeros(samples.shape)
		p_z = mgd(samples, mu, std)

		#print(p_z.shape)
			
		#Multiply the probablities
		#marginal_likelihood += (p_x * p_z)/q_z
		#Use logsumexp trick to avoid ver small prob
		logp_x += np.exp(np.log(p_x.cpu().numpy()) + np.log(p_z) - np.log(q_z))

		print(logp_x)

	#Divide sum over K and apply log
	logp_x = logp_x * (1/z.shape[1])
	logp_x = np.log(logp_x)

	print(logp_x)

	sd

	return logp_x


def loss_IS(model, true_x, z):

	marginal_likelihood = 0

	#Loop over the elements i of batch
	m = true_x.shape[0]

	#Save logp(x)
	logp_x = torch.zeros([m])

	for i in range(m):
		for k in range(z.shape[1]):
			print(k)
			#z_ik
			samples = z[i,k,:]

			#x_i of current element i of batch
			true_xi = true_x[i,:,:].view(1,1,true_x.shape[2],true_x.shape[3])

			#Compute the reconstructed x's from sampled z's
			x = model.decode(samples.to(device))

			#Compute the p(x_i|z_ik) of x sampled from z_ik
			#Bernoulli dist = Apply BCE
			#Outputs a scalar
			p_x = F.binary_cross_entropy(x.view(-1, 784), true_xi.view(-1, 784), reduction='sum')
			p_x = p_x.item()
			#print(p_x)		

			##q(z_ik¦x_i) follows a normal dist

			#Get mean and std from encoder
			#2 Vectors of 100
			mu, logvar = model.encode(true_xi.to(device))
			std = torch.exp(0.5*logvar)
			q_z = mgd(samples, mu, std)

			#print(q_z)

			##p(z_ik) follows a normal dist with mean 0/variance 1
			#Normally distributed with loc=0 and scale=1
			std = torch.ones(samples.shape)
			mu = torch.zeros(samples.shape)
			p_z = mgd(samples, mu, std)

			#print(q_z)
			
			#Multiply the probablities
			#marginal_likelihood += (p_x * p_z)/q_z
			#Use logsumexp trick to avoid ver small prob
			marginal_likelihood += np.exp(np.log(p_x) + np.log(p_z) - np.log(q_z))

		#Divide sum over K and apply log
		marginal_likelihood = marginal_likelihood * (1/z.shape[1])
		logp_x[i] = np.log(marginal_likelihood)

		#print(logp_x)

	return logp_x
		

if __name__ == "__main__":

	#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	#model = VAE().to(device)
	#optimizer = optim.Adam(model.parameters(), lr=3e-4)


	###Training

	#n_epochs = 20

	#Load data loaders
	train_loader, valid_loader, test_loader = mnist_loader.get_data_loader("binarized_mnist", 64)

	#Train + val
	#for epoch in range(n_epochs):
	#	train(epoch, train_loader)
	#	eval(epoch, valid_loader)

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

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model = VAE().to(device)
	model.load_state_dict(torch.load(path_weights))
	print("Model successfully loaded")

	#put the model in eval mode
	model = model.eval()

	eval_method = ['elbo', 'IS']
	#eval_method = ['elbo']
	dataset = ['val', 'test']

	with torch.no_grad():

		for method in eval_method:
			for data in dataset:

				print('Evaluating using '+method+'...')

				if(data == 'val'):
					eval(0, data, valid_loader, eval_method=method)

				if(data == 'test'):
					eval(0, data, test_loader, eval_method=method)


		#Generate a batch of images using trained model
		print('Generating samples...') 
		sample = torch.randn(64, 100).to(device)
		sample = model.decode(sample).cpu()
		save_image(sample.view(64, 1, 28, 28),
					   'results/sample_' + str(epoch) + '.png')



	










