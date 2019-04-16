import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

import mnist_loader

##This code was inspired from:
#https://github.com/pytorch/examples/blob/master/vae/main.py


class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()
		##Encoder
		conv_e = nn.Sequential(
				nn.Conv2d(1, 32, kernel_size=3, bias=True, stride=1),
				nn.ReLU(True),
				nn.AvgPool2d(kernel_size=2, stride=2),
				nn.Conv2d(32, 64, kernel_size=3, bias=True, stride=1),
				nn.ReLU(True),
				nn.AvgPool2d(kernel_size=2, stride=2),
				nn.Conv2d(64, 256, kernel_size=5, bias=True, stride=1),
				nn.ReLU(True),
				
			)

		linear_e = nn.Linear(256, 100*2)

		self.encoder = nn.ModuleList([conv_e, linear_e])

		##Decoder
		conv_d = nn.Sequential(
		#Takes z latent variable of size 100
				
				nn.Conv2d(256, 64, kernel_size=5, bias=True, stride=1, padding=4),
				nn.ReLU(True),
				nn.UpsamplingBilinear2d(size=None, scale_factor=2),
				nn.Conv2d(64, 32, kernel_size=3, bias=True, stride=1, padding=2),
				nn.ReLU(True),
				nn.UpsamplingBilinear2d(size=None, scale_factor=2),
				nn.Conv2d(32, 16, kernel_size=3, bias=True, stride=1, padding=2),
				nn.ReLU(True),
				nn.Conv2d(16, 1, kernel_size=3, bias=True, stride=1, padding=2)

			)

		linear_d = nn.Linear(100, 256)
		relu = nn.ReLU(True)

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
		#Reshape z from 2 dim to 4 dim
		z = z.view(z.shape[0], z.shape[1], 1, 1)
		recon_x = self.decoder[2](z)
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
		return recon_x, mu, logvar



# Reconstruction + KL divergence losses summed over all elements of batch
def loss_elbo(recon_x, x, mu, logvar):
	
	marginal_likelihood = F.binary_cross_entropy(torch.sigmoid(recon_x.view(-1, 784)), x.view(-1, 784), reduction='sum')

	#print(marginal_likelihood)

	#Note: you can compute this using logvar or std
	# 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

	#KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
	#KLD = 0.5 * torch.sum(mu.pow(2) - logvar + logvar.exp() - 1)
	KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

	loss = marginal_likelihood + KLD

	#Normalize
	#D_train = logvar.shape[0]
	#ELBO /= D_train

	#loss = -ELBO

	return loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-4)


def train(epoch, train_loader):
	#Mode train
	model.train()

	train_loss = 0

	for i, inputs in enumerate(train_loader):

		inputs = inputs.to(device)
		optimizer.zero_grad()
		recon_batch, mu, logvar = model(inputs)
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



##Evaluate VAE using importance sampling 
def loss_IS(model, x, z):

	marginal_likelihood = 0

	M = x.shape[0]

	#x_i = model.decode(samples)
	#x_i = x_i.view(-1, 784)

	##p(x_i¦z_ik) follows a bernoulli dist 
	#(64, 784)
	m =  torch.distributions.Bernoulli(x)
	p_x = m.sample()

	for k in range(z.shape[1]):
		#z_ik
		samples = z[:,k,:]

		##p(z_ik) follows a normal dist with mean 0/variance 1
		#(64, 100)	
		p_z = torch.randn(64, 100)	
		
		##q(z_ik¦x_i) follows a normal dist
		
		#Get mean and std from encoder
		mu, logvar = model.encode(x.view(64,1,28,28))
		std = torch.exp(0.5*logvar)

		m = torch.distributions.Normal(mu, std)
		#Or we can do this instead : p_z = mu + samples*std
		q_z = m.sample()
		
		#Multiply the probablities
		#print(p_z.shape) #(64, 100)
		#print(q_z.shape) #(64, 100)
		#print(p_x.shape) #(64, 784)

		marginal_likelihood += (p_x * p_z)/q_z

	#Divide sum over K and apply log
	marginal_likelihood = marginal_likelihood * (1/z.shape[1])
	marginal_likelihood = torch.log(marginal_likelihood)

	print(marginal_likelihhod.shape)
	sd
		

if __name__ == "__main__":


	###Training

	#n_epochs = 20

	#Load data loaders
	#train_loader, valid_loader, test_loader = mnist_loader.get_data_loader("binarized_mnist", 64)

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

	###Evaluating

	path_weights = 'weights/weights.h5'

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

	model = VAE()
	model.load_state_dict(torch.load(path_weights))
	print("Model successfully loaded")

	#put the model in eval mode
	model = model.eval()

	model = model.to(device)

	samples = torch.randn(64, 200, 100).to(device)
	data = torch.randn(64, 784).to(device)

	data = torch.sigmoid(data)

	loss_IS(model, data, samples)