# Create a Neural Network (MLP) with 2 hidden layers


# Package imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1) # set a seed so that the results are consistent

class NN(object):

    def __init__(self, hidden_dims=(1024, 2048), n_hidden, n_class=10, input_dim=784, mode='train', datapath=None, model_path=None):
    	
    	self.hidden_dims = hidden_dims
    	self.n_hidden = n_hidden
    	self.input_dim = input_dim

    	#all dims of the network
    	self.dims = [input_dim]
    	self.dims.extend(hidden_dims)
    	self.dims.append(n_class)

    	#load the npy data
    	self.data = np.load(datapath)

    	#Divide data to train, val and test
    	self.D_train = self.data[0]
    	self.D_val = self.data[1]
    	self.D_test = self.data[2]

    	#dim of datasets
    	self.dim_data = (self.D_train[1].shape[0], self.D_val[1].shape[0], self.D_test[1].shape[0])

    	self.model_path = model_path


    #def initialize_weights(self, n_hidden, dims, init_method):
    def initialize_weights(self, init_method):

    	#init dic to save weights and biases
    	parameters = {}

    	#if the dim of the network doesn't respect the n_hidden layers
    	print len(self.dims)
    	print self.n_hidden+2
    	assert len(self.dims) == (self.n_hidden + 2),"network dimensions are incoherent!"

    	#Random init of weights
    	if('random'):
    		for i in range(self.n_hidden+1):

		    	W = np.random.randn(self.dims[i+1], self.dims[i]) * 0.01
		    	b = np.random.randn(1, self.dims[i+1]) * 0.01

		    	#Save weights
		    	parameters.update({"W"+str(i+1):W, "b"+str(i+1):b})

		#weights init to zeros
		if('zeros'):
			for i in range(self.n_hidden+1):
				W = np.zeros(self.dims[i+1], self.dims[i])
				b = np.zeros(1, self.dims[i+1])

				#Save weights
				parameters.update({"W"+str(i+1):W, "b"+str(i+1)})
			

		return parameters 

    def forward(self,input,labels):
    	pass

    def activation(self,input):
    	pass

    def loss(self,prediction):
    	pass

    def softmax(self,input):
    	pass

    def backward(self,cache,labels):
    	pass

    def update(self,grads):
    	pass

    def train(self):
    	pass

    def test(self):
    	pass



if __name__ == '__main__':
	nn = NN(hidden_dims=(20, 15), datapath='datasets/mnist.pkl.npy')
	print(nn.dim_data)

	parameters = nn.initialize_weights('zeros')
	
	for key, value in parameters.iteritems() :
		print key, value.shape