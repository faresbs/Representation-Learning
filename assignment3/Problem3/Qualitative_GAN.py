import torch
from torch import nn, optim
from torch.nn import functional as F
from torchvision.utils import save_image
import numpy as np

from WGANGP import Generator
from vae import VAE

img_size=32
channels=3
latent_dim=100

img_shape = (channels, img_size, img_size)
data_path='./'

#Disentangled representation Q3.2
#epsilon is the small perturbation
#Accepts one sample z (latent_space)
#Saves 100 (latent_space dimension) images
def disentangled(z, model, epsilon=3):
    latent_space = z.shape[1]
    print("latent_space=",latent_space)
    model_name = model.__class__.__name__
    dim1=21
    dim2=65
    size=5
	#Loop over the dimensions of latent space
    new_z = z.clone()
    #sample = torch.empty((1,*img_shape)).to(device)

    print(model_name)
    for i in range(size):
      if i==0:
        if model_name == 'VAE':
            sample = model.decode(z).view(1,3,32,32)
        else:
            sample = model(z)
        for i in range(size-1):
          new_z[0][dim2] += epsilon/2.
          if model_name =='VAE':
            sample0 = model.decode(new_z).view(1, 3, 32, 32)
            sample= torch.cat([sample,sample0])
          else:
            sample0 = model(new_z)
            sample= torch.cat([sample,sample0])
        new_z[0][dim2] = z[0][dim2]
      else:
        new_z[0][dim1] += epsilon/2.
        if model_name=='VAE':
          sample0 = model.decode(new_z).view(1, 3, 32, 32)
          sample = torch.cat([sample,sample0])
        else:
          sample0 = model(new_z)
          sample = torch.cat([sample,sample0])
        for i in range(size-1):
          new_z[0][dim2] += epsilon/2.
          if model_name=='VAE':
            sample0 = model.decode(new_z).view(1, 3, 32, 32)
            sample= torch.cat([sample,sample0])
          else:
            sample0 = model(new_z)
            sample= torch.cat([sample,sample0])

      new_z[0][dim2] = z[0][dim2]


    print(sample.shape)
    save_image(sample.data[0:25],data_path+'Tests/sample_' + model_name + str(dim1)+'-'+str(dim2) +'_epsilon_' +str(epsilon) +'.png', nrow=5, normalize=True)
    #new_z = z.clone()

#Q3.3
#Accepts one sample z (latent_space)
#Saves 2+n images (from the two z samples and their (number of alpha) interpolations)
def interpolating(z0, z1, method, model):
	alpha = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

	sample = torch.zeros([len(alpha),3, 32, 32])

	#(a)Interpolate latent space
	if(method=='latent'):
		i = 0
		for a in alpha:
			new_z = a * z0 + (1 - a)*z1

			aux = model(new_z)
			sample[i, :, :, :] = aux.view(1, 3, 32, 32)
			i += 1

		save_image(sample.view(len(alpha), 3, 32, 32),
					   'Interpolation/latent space/GANsample.png', normalize=True)

	#(b)Interpolate image space

	elif(method=='image'):

		sample0 = model(z0)

		save_image(sample0.view(1, 3, 32, 32),
					   'Interpolation/image space/GANsample0.png', normalize=True)

		sample1 = model(z1)

		save_image(sample1.view(1, 3, 32, 32),
					   'Interpolation/image space/GANsample1.png', normalize=True)


		i = 0
		for a in alpha:

			aux = a * sample0 + (1 - a) * sample1
			sample[i, :, :, :] = aux.view(1, 3, 32, 32)
			i += 1

		save_image(sample.view(len(alpha), 3, 32, 32),
					   'Interpolation/image space/GANsample.png', normalize=True)


if __name__ == "__main__":
    ###Qualitative Evaluation
    model_eval = 'GAN'

    if model_eval=='VAE':
      path_weights = data_path+'weights/weights.h5'
      model = VAE()
    if model_eval=='GAN':
      path_weights = data_path+'weights/GAN_weights.h5'
      model = Generator()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model.load_state_dict(torch.load(path_weights))
        print("Model successfully loaded")
    else:
        device = torch.device("cpu")
        model.load_state_dict(torch.load(path_weights,map_location='cpu'))
        print("Model successfully loaded")


    model = model.to(device)

    #put the model in eval mode
    model = model.eval()

    #Q3.2
    #Sample z from prior p(z)=N(0,1)
    # z = torch.FloatTensor(np.random.normal(0, 1, (1, latent_dim))).to(device)
    # disentangled(z, model, epsilon=90)

    #Q3.3
    #Sample two z from prior p(z)=N(0,1)
    z1 = torch.randn(1,100).to(device)
    z2 = torch.randn(1,100).to(device)
    interpolating(z1, z2, 'image', model)
