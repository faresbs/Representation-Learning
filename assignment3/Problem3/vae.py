import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import classify_svhn as svhn

import matplotlib.pyplot as plt


##This code was inspired from:
#https://github.com/pytorch/examples/blob/master/vae/main.py


class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		##Encoder
		conv_e = nn.Sequential(
				nn.Conv2d(3, 32, kernel_size=3, bias=True, stride=1),
				nn.ReLU(),

				nn.AvgPool2d(kernel_size=3, stride=1),
				nn.Conv2d(32, 64, kernel_size=3, bias=True, stride=1),
				nn.ReLU(),

				nn.AvgPool2d(kernel_size=3, stride=1),
				nn.Conv2d(64, 124, kernel_size=3, bias=True, stride=2),
				nn.ReLU(),

				nn.AvgPool2d(kernel_size=3, stride=1),
				nn.Conv2d(124, 256, kernel_size=3, bias=True, stride=2),
				nn.ReLU(),
				
			)

		linear_e = nn.Linear(4*4*256, 100*2)

		self.encoder = nn.ModuleList([conv_e, linear_e])

		##Decoder
		conv_d = nn.Sequential(
		#Takes z latent variable of size 100
				
				nn.Conv2d(256, 64, kernel_size=3, bias=True, stride=1, padding=4),
				nn.ReLU(),
				nn.UpsamplingBilinear2d(size=None, scale_factor=2),
				nn.Conv2d(64, 32, kernel_size=3, bias=True, stride=1, padding=2),
				nn.ReLU(),
				nn.UpsamplingBilinear2d(size=None, scale_factor=2),
				nn.Conv2d(32, 16, kernel_size=3, bias=True, stride=1, padding=1),
				nn.ReLU(),
				nn.Conv2d(16, 3, kernel_size=3, bias=True, stride=1, padding=1)

			)

		linear_d = nn.Linear(100, 256)
		relu = nn.ReLU()

		self.decoder = nn.ModuleList([linear_d, relu, conv_d])

	#Outputs mean/log-variance
	def encode(self, x):

		z = self.encoder[0](x)
		#Reshape for FC
		z = z.view(z.size(0), -1)

		#print(z.shape)

		#Outputs 2 vectors of size 100, mean vector and std vector
		z = self.encoder[1](z)

		#first 100 for mean vector, the other 100 for logvar
		return z[:, :100], z[:, 100:]

	#Outputs reconstructed x
	def decode(self, z):
		#z = self.linear_d(z)
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
		#Different results with sigmoid
		recon_x = torch.tanh(recon_x)
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
		#print(z.shape)
		#z = (batch,latent_space)
		recon_x = self.decode(z)
		return recon_x, mu, logvar



# Reconstruction + KL divergence losses summed over all elements of batch
def loss_elbo(recon_x, x, mu, logvar):

	#Use MSE loss because we are dealing with RGB images
	loss = nn.MSELoss(reduction='sum')
	marginal_likelihood = loss(recon_x.view(recon_x.shape[0], recon_x.shape[1], recon_x.shape[2]**2), x.view(recon_x.shape[0], recon_x.shape[1], x.shape[2]**2))
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	loss = marginal_likelihood + KLD

	return loss



def train(epoch, train_loader):
	#Mode train
	model.train()

	train_loss = 0

	for i, inputs in enumerate(train_loader):

		x = inputs[0]
		y = inputs[1]
		
		x = x.to(device)
		optimizer.zero_grad()

		recon_batch, mu, logvar = model(x)
		loss = loss_elbo(recon_batch, x, mu, logvar)
		loss.backward()
		train_loss += loss.item()
		optimizer.step()

		if i % 100 == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, i * len(x), len(train_loader.dataset),
				100. * i / len(train_loader),
				loss.item() / len(x)))

	print('====> Epoch: {} Average loss: {:.4f}'.format(
		  epoch, train_loss / len(train_loader.dataset)))


def eval(epoch, valid_loader):
	#Mode eval
	model.eval()

	epoch_loss = 0

	with torch.no_grad():

		for i, inputs in enumerate(test_loader):

			x = inputs[0]
			y = inputs[1]

			x = x.to(device)
			recon_batch, mu, logvar = model(x)
			epoch_loss += loss_elbo(recon_batch, x, mu, logvar).item()
				
	epoch_loss /= len(test_loader.dataset)
	print('====> Test Average loss: {:.4f}'.format(epoch_loss))


#Disentangled representation Q3.2
#epsilon is the small perturbation
#Accepts one sample z (latent_space)
#Saves 100 (latent_space dimension) images
def disentangled(z, epsilon=5):

	latent_space = z.shape[0]
	#Loop over the dimensions of latent space
	new_z = z.clone()

	for i in range(latent_space):
		
		new_z[i] = z[i] + epsilon 
		sample = model.decode(new_z)

		save_image(sample.view(1, 3, 32, 32),
					   'Disentangled representation/sample_' + str(i) + '.png', normalize=True)

		new_z = z.clone()

#Interpolating in the data space Q3.3
#Accepts one sample z (latent_space)
#Saves 2+n images (from the two z samples and their (number of alpha) interpolations)
def interpolating(z1, z2, method):
	alpha = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5]

	#(a)Interpolate latent space
	if(method=='latent'):
		for i in range(len(alpha)):
			#new_z = 

			save_image(sample.view(1, 3, 32, 32),
					   'Interpolation/sample_' + str(alpha[i]) + '.png', normalize=True)

	#(b)Interpolate image space 






if __name__ == "__main__":

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = VAE().to(device)
	optimizer = optim.Adam(model.parameters(), lr=3e-4)

	###Training

	#n_epochs = 1000

	#Load data
	train_loader, valid_loader, test_loader = svhn.get_data_loader("svhn", 32)

	#Train + val
	#for epoch in range(n_epochs):
	#	train(epoch, train_loader)
	#	eval(epoch, valid_loader)

	#	with torch.no_grad():
			#Generate a batch of images using current parameters 
			#Sample z from prior p(z) = N(0,1)
	#		sample = torch.randn(16, 100).to(device)
	#		sample = model.decode(sample)
	#		save_image(sample.view(16, 3, 32, 32),
	#				   'results/sample_' + str(epoch) + '.png', normalize=True)


	#Saving the model weights
	#torch.save(model.state_dict(), 'weights/weights.h5')


	###Qualitative Evaluation
	path_weights = 'weights/weights.h5'

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model.load_state_dict(torch.load(path_weights))
	print("Model successfully loaded")

	#put the model in eval mode
	model = model.eval()

	#Q3.2
	#Sample z from prior p(z)=N(0,1)
	z = torch.randn(100).to(device)
	disentangled(z)

	#Q3.3
	#Sample two z from prior p(z)=N(0,1)
	#z1 = torch.randn(100).to(device)
	#z2 = torch.randn(100).to(device)
	#interpolating(z1, z2, 'latent')
