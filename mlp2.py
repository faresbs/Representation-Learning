# Create a Neural Network (MLP) with 2 hidden layers


# Package imports
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1) # set a seed so that the results are consistent

#TO FIX

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
		#number of examples
		m = y.shape[1]

		#y_hat is the probability after softmax
		#y is one hot vector
		log_likelihood = -np.log(y_hat)*y
		loss = np.sum(log_likelihood) / m
		return loss


	#Measure prob with softmax
	def softmax(self,inp):
		#exps = np.exp(inp)
		#return exps / np.sum(exps)

		#Stable softmax
		exps = np.exp(inp - np.max(inp, axis=0))
		return exps / np.sum(exps, axis=0)


	def backward(self, parameters, cache, y, X):

		#init dic for gradients
		grads = {}

		#Number of examples
		m = len(y)

		dZ = cache["A"+str(self.n_hidden+1)] - y.T 
		dW = (1./m) * np.dot(dZ, cache["A"+str(self.n_hidden)].T)
		db = (1./m) * np.sum(dZ, axis=1, keepdims=True)

		# gradients must have same dimension as the parameters
		assert(dW.shape == parameters["W"+str(self.n_hidden+1)].shape)
		assert(db.shape == parameters["b"+str(self.n_hidden+1)].shape)

		# Save updated grads
		grads.update({"dW"+str(self.n_hidden+1):dW, "db"+str(self.n_hidden+1):db})


		for i in range(self.n_hidden, 0, -1):

			# Derivation of relu
			drelu = cache["Z"+str(i)]
			drelu[drelu<=0] = 0
			drelu[drelu>0] = 1

			#parameters["W"+str(i+1)]
			dZ = np.dot(dW.T, dZ) * drelu

			if(i == 1):
				A = X
			else:
				A = cache["A"+str(i-1)]

			dW = (1./m) * np.dot(dZ, A.T)
			db = (1./m) * np.sum(dZ, axis=1, keepdims=True)

			# gradients must have same dimension as the parameters
			assert(dW.shape == parameters["W"+str(i)].shape)
			assert(db.shape == parameters["b"+str(i)].shape)

			# Save updated grads
			grads.update({"dW"+str(i):dW, "db"+str(i):db})

		return grads

		"""			
		dZ3 = cache['A3'] - y.T
		dW3 = (1./m) * np.dot(dZ3, cache['A2'].T)
		db3 = (1./m) * np.sum(dZ3, axis=1, keepdims=True)
		
		# Derivation of relu
		drelu = cache['Z2']
		drelu[drelu<=0] = 0
		drelu[drelu>0] = 1

		dZ2 = np.dot(parameters['W3'].T, dZ3) * drelu

		dW2 = (1./m) * np.dot(dZ2, cache['A1'].T)
		db2 = (1./m) * np.sum(dZ2, axis=1, keepdims=True)

		# Derivation of relu
		drelu = cache['Z1']
		drelu[drelu<=0] = 0
		drelu[drelu>0] = 1

		dZ1 = np.dot(parameters['W2'].T, dZ2) * drelu
		dW1 = (1./m) * np.dot(dZ1, X.T)
		db1 = (1./m) * np.sum(dZ1, axis=1, keepdims=True)


		# gradients must have same dimension as the parameters
		assert(dW3.shape == parameters['W3'].shape)
		assert(db3.shape == parameters['b3'].shape)
		assert(dW2.shape == parameters['W2'].shape)
		assert(db2.shape == parameters['b2'].shape)
		assert(dW1.shape == parameters['W1'].shape)
		assert(db1.shape == parameters['b1'].shape)

		grads = {"dW3":dW3, "db3":db3, "dW2":dW2, "db2":db2, "dW1":dW1, "db1":db1}
		
		return grads
		"""

	#Update parameters using stochastic gradient descent
	def update(self, grads, parameters, learning_rate):


		for i in range(self.n_hidden+1):

			# Retrieve parameters
			W = parameters["W"+str(i+1)]
			b = parameters["b"+str(i+1)]

			# Retrieve grads
			dW = grads["dW"+str(i+1)]
			db = grads["db"+str(i+1)]

			# Update parameters using gradient descent
			W = W - learning_rate * grads["dW"+str(i+1)]
			b = b - learning_rate * grads["db"+str(i+1)]

			# Save updated parameters
			parameters.update({"W"+str(i+1):W, "b"+str(i+1):b})

		return parameters

	def train(self, iterations, init_method, learning_rate, X, labels):

		#One hot encoding of labels
		y = np.eye(self.n_class)[labels]

		parameters = self.initialize_weights(init_method)

		# Loop (gradient descent)
		for i in range(iterations):

			print str(i)+"/"+str(iterations)

			#Forward pass
			out, cache = self.forward(X, parameters)

			#Backward pass
			grads = self.backward(parameters, cache, y, X)

			#Update
			parameters = self.update(grads, parameters, learning_rate)

			#print out.shape
			#print labels.shape
			print self.loss(out, y.T)

		return parameters

	def test(self):
		pass



if __name__ == '__main__':
	nn = NN(hidden_dims=(20, 15), datapath='./datasets/mnist.pkl.npy')
	print("train/val/test: "+str(nn.dim_data))

	#parameters = nn.initialize_weights(init_method='zeros')

	#for key, value in parameters.iteritems() :
	#	print key, value.shape

	#out, cache = nn.forward(nn.D_train[0], parameters)

	#for key, value in cache.iteritems() :
	#	print key, value.shape

	#grads = nn.backward(parameters, cache, nn.D_train[1], nn.D_train[0])

	parameters = nn.train(100, 'random', 0.5, nn.D_train[0], nn.D_train[1])
	out, cache = nn.forward(nn.D_train[0], parameters)

	#print nn.D_train[1][0]
	#print out[:, 0]
