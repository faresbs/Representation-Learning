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
		relu = nn.ELU()

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
		#Reshape z from 2 dim to 4 dim
		z = z.view(z.shape[0], z.shape[1], 1, 1)
		recon_x = self.decoder[2](z)
		#Different with sigmoid whyy ??
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
			inputs = inputs.to(device)
			recon_batch, mu, logvar = model(inputs)
			epoch_loss += loss_elbo(recon_batch, inputs, mu, logvar).item()
				
	epoch_loss /= len(test_loader.dataset)
	print('====> Test Average loss: {:.4f}'.format(epoch_loss))




if __name__ == "__main__":

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	model = VAE().to(device)
	optimizer = optim.Adam(model.parameters(), lr=3e-4)


	###Training

	n_epochs = 100

	#Load data
	train_loader, valid_loader, test_loader = svhn.get_data_loader("svhn", 32)

	#Train + val
	for epoch in range(n_epochs):
		train(epoch, train_loader)
		#eval(epoch, valid_loader)

		with torch.no_grad():
			#Generate a batch of images using current parameters 
			sample = torch.randn(32, 100).to(device)
			sample = model.decode(sample)
			#print(sample.shape)
			save_image(sample.view(32, 3, 32, 32),
					   'results/sample_' + str(epoch) + '.png', normalize=True)


	#Saving the model weights
	torch.save(model.state_dict(), 'weights/weights.h5')

	###Evaluating

	#path_weights = 'weights/weights.h5'

	#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	#model = VAE()
	#model.load_state_dict(torch.load(path_weights))
	#print("Model successfully loaded")

	#put the model in eval mode
	#model = model.eval()

	#model = model.to(device)

	#samples = torch.randn(64, 200, 100).to(device)
	#data = torch.randn(64, 784).to(device)

	#data = torch.sigmoid(data)

	#loss_IS(model, data, samples)