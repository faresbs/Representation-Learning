# Create a Neural Network (MLP) with 2 hidden layers


# Package imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1) # set a seed so that the results are consistent

class NN(object):

	def __init__(self, hidden_dims=(1024, 2048), n_hidden=2, n_class=10, input_dim=784, mode='train', datapath=None, model_path=None):
		
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

		#Reshape to (Mx, m) from (m, Mx) to simplify calculations
		self.D_train[0] = self.D_train[0].T
		self.D_val[0] = self.D_val[0].T
		self.D_test[0] = self.D_test[0].T

		#dim of datasets
		self.dim_data = (self.D_train[1].shape[0], self.D_val[1].shape[0], self.D_test[1].shape[0])

		self.model_path = model_path

		self.n_class = n_class


	#def initialize_weights(self, n_hidden, dims, init_method):
	def initialize_weights(self, init_method):

		#init dic to save weights and biases
		parameters = {}

		#if the dim of the network doesn't respect the n_hidden layers
		assert len(self.dims) == (self.n_hidden + 2),"network dimensions are incoherent!"

		#Random init of weights
		if(init_method=='random'):
			for i in range(self.n_hidden+1):

				W = np.random.randn(self.dims[i+1], self.dims[i]) * 0.01
				b = np.random.randn(self.dims[i+1], 1) * 0.01

				#Save weights
				parameters.update({"W"+str(i+1):W, "b"+str(i+1):b})

		#weights init to zeros
		if(init_method=='zeros'):
			for i in range(self.n_hidden+1):
				W = np.zeros((self.dims[i+1], self.dims[i]))
				b = np.zeros((self.dims[i+1], 1))

				#Save weights
				parameters.update({"W"+str(i+1):W, "b"+str(i+1):b})
			
		return parameters 


	def forward(self, X, parameters):

		#init dic to save cache of the forward 
		cache = {}

		#output of the last layer 
		A = X
		
		for i in range(self.n_hidden):

			# Retrieve parameters
			W = parameters["W"+str(i+1)]
			b = parameters["b"+str(i+1)]

			# Forward pass in hidden layer
			Z = np.dot(W, A) + b
			A = self.activation(Z)
			
			#Save cache
			cache.update({"Z"+str(i+1):Z, "A"+str(i+1):A}) 

		#Apply softmax at the last layer
		#Retrieve parameters
		W = parameters["W"+str(self.n_hidden+1)]
		b = parameters["b"+str(self.n_hidden+1)]

		#logits
		Z = np.dot(W, A) + b
		#prob after softmax
		A = self.softmax(Z)
			
		#Save cache
		#We don't need to cache the Z ?
		cache.update({"Z"+str(self.n_hidden+1):Z, "A"+str(self.n_hidden+1):A})

		#returns the last output as prediction + cache
		return A, cache


	def activation(self, z):
		#Relu
		a = np.maximum(0, z)
		return a

		# Sigmoid function
		#a = 1 / (1 + np.exp(-z))
		#return a



    #Compute the cross entropy cost
	def loss(self, y_hat, y):
		#y_hat is the probability after softmax
		log_likelihood = -np.log(y_hat[range(m),y])
		loss = np.sum(log_likelihood) / m
		return loss

	#Measure prob with softmax
	def softmax(self,inp):
		exps = np.exp(inp)
		return exps / np.sum(exps)

    	#Stable softmax
    	#exps = np.exp(inp - np.max(inp))
    	#return exps / np.sum(exps)


	def backward(self, parameters, cache, labels, X):

		#One hot encoding
		labels = np.eye(self.n_class)[labels]

		print labels.shape
		print X.shape

		m = len(labels)
		
		dZ3 = cache['A3'] - labels.T
		dW3 = (1./m) * np.dot(dZ3, cache['A2'].T)
		db3 = (1./m) * np.sum(dZ3, axis=0, keepdims=True)

		# Derivation of relu
		if (cache['Z2'].all >= 0):
			drelu = 1
		else:
			drelu = 0

		dZ2 = np.dot(dZ3.T, parameters['W3']) * drelu
		dW2 = (1./m) * np.dot(dZ2.T, cache['A1'].T)
		db2 = (1./m) * np.sum(dZ2, axis=0, keepdims=True)
		
		# Derivation of relu
		if (cache['Z1'].all >= 0):
			drelu = 1
		else:
			drelu = 0

		dZ1 = np.dot(dZ2, parameters['W2']) * drelu
		dW1 = (1./m) * np.dot(dZ1.T, X.T)
		db1 = (1./m) * np.sum(dZ1, axis=0, keepdims=True)

		# gradients must have same dimension as the parameters
		assert(dW2.shape == parameters['W2'].shape)
		assert(db2.shape == parameters['b2'].shape)
		assert(dW1.shape == parameters['W1'].shape)
		assert(db1.shape == parameters['b1'].shape)

		grads = {"dW2":dW2, "db2":db2, "dW1":dW1, "db1":db1}

		return grads


	def update(self,grads):
		pass

	def train(self):
		pass

	def test(self):
		pass



if __name__ == '__main__':
	nn = NN(hidden_dims=(20, 15), datapath='../datasets/mnist.pkl.npy')
	print("train/val/test: "+str(nn.dim_data))

	parameters = nn.initialize_weights(init_method='zeros')

	#for key, value in parameters.iteritems() :
	#	print key, value.shape

	out, cache = nn.forward(nn.D_train[0], parameters)

	#for key, value in cache.iteritems() :
	#	print key, value.shape


	grads = nn.backward(parameters, cache, nn.D_train[1], nn.D_train[0])