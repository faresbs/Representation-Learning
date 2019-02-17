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
		#counter for number of parameters
		n_params = 0
		#if the dim of the network doesn't respect the n_hidden layers
		assert len(self.dims) == (self.n_hidden + 2),"network dimensions are incoherent!"

		#Glarot distribution init of weights
		if(init_method=='glorot'):
			for i in range(self.n_hidden+1):

				d = np.sqrt(6.0/(self.dims[i]+self.dims[i+1]))

				W = 2*d*np.random.rand(self.dims[i+1], self.dims[i]) - d
				b = np.zeros((self.dims[i+1], 1))

				n_params = n_params + W.size + b.size
				#Save weights
				parameters.update({"W"+str(i+1):W, "b"+str(i+1):b})


		#Normal distribution init of weights
		#To break symmetry
		if(init_method=='normal'):
			for i in range(self.n_hidden+1):

				W = np.random.randn(self.dims[i+1], self.dims[i])
				b = np.zeros((self.dims[i+1], 1))

				n_params = n_params + W.size + b.size

				#Save weights
				parameters.update({"W"+str(i+1):W, "b"+str(i+1):b})

		#weights init to zeros
		if(init_method=='zero'):
			for i in range(self.n_hidden+1):

				W = np.zeros((self.dims[i+1], self.dims[i]))
				b = np.zeros((self.dims[i+1], 1))

				n_params = n_params + W.size + b.size
				#Save weights
				parameters.update({"W"+str(i+1):W, "b"+str(i+1):b})


		print('Number of parameters = ' + str(n_params))

		#plt.figure()
		#plt.plot(parameters['W'+str(i+1)].flatten(), '.')
		#plt.title("W"+str(i)+" using "+str(init_method)+" initializaton method")
		#plt.xlabel('parameter')
		#plt.ylabel('initial value')

		# plt.figure()
		# plt.plot(parameters['W'+str(i+1)].flatten(), '.')
		# plt.title("W"+str(i)+" using "+str(init_method)+" initializaton method")
		# plt.xlabel('parameter')
		# plt.ylabel('initial value')
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
			# print('size at ' + str(i) + 'layer' + str(Z.shape))
			A = self.activation(Z)

			#Save cache
			cache.update({"Z"+str(i+1):Z, "A"+str(i+1):A})

		#Apply softmax at the last layer
		#Retrieve parameters
		W = parameters["W"+str(self.n_hidden+1)]
		b = parameters["b"+str(self.n_hidden+1)]

		#logits
		Z = np.dot(W, A) + b
		# print('size at last ' + 'layer' + str(Z.shape))
		#prob after softmax
		A = self.softmax(Z)

		#Save cache
		cache.update({"Z"+str(self.n_hidden+1):Z, "A"+str(self.n_hidden+1):A})

		#returns the last output as prediction + cache
		return A, cache


	def activation(self, z, function="relu"):
		#Relu
		if (function=="relu"):
			a = np.maximum(0, z)
			return a

		#Sigmoid function
		if(function=="sigmoid"):
			a = 1 / (1 + np.exp(-z))
			return a


		#tanh function
		if(function=="tanh"):
			a = (2 / (1 + np.exp(-(2*z)))) - 1
			return a


    #Compute the cross entropy cost
	def loss(self, y_hat, y):
		#number of examples
		m = y.shape[1]

		#y_hat is the probability after softmax
		#y is one hot vector
		log_likelihood = -np.log(y_hat)*y
		#print(type(log_likelihood))

		#Transform nan values to 0 (nan = infinity * 0), this is optional
		a = np.isnan(log_likelihood)
		log_likelihood[a] = 0
		#print(y_hat[0, :])
		#print(-np.log(y_hat)[0, :])
		#print(log_likelihood[0, :])

		#print(y[0, :])
	
		#print(np.sum(log_likelihood))
		loss = np.sum(log_likelihood) / m
		return loss


	#Measure prob with softmax
	def softmax(self,inp):
		# exps = np.exp(inp)
		# return exps / np.sum(exps)

		#Stable softmax
		exps = np.exp(inp - np.max(inp, axis=0))
		return exps / np.sum(exps, axis=0)


	def backward(self, parameters, cache, y, X, act_function="relu"):

		#init dic for gradients
		grads = {}

		#Number of examples
		m = len(y)
		# print('in backward')
		# print(cache["A"+str(self.n_hidden+1)].shape)
		# print(cache["A"+str(self.n_hidden+1)][:,0])
		# print(y.shape)
		# print(y.T[:,0])
		#Derivative of cross entropy with respect to softmax
		dZ = cache["A"+str(self.n_hidden+1)] - y.T
		dW = (1./m) * np.dot(dZ, cache["A"+str(self.n_hidden)].T)
		db = (1./m) * np.sum(dZ, axis=1, keepdims=True)

		# gradients must have same dimension as the parameters
		assert(dW.shape == parameters["W"+str(self.n_hidden+1)].shape)
		assert(db.shape == parameters["b"+str(self.n_hidden+1)].shape)

		# Save updated grads
		grads.update({"dW"+str(self.n_hidden+1):dW, "db"+str(self.n_hidden+1):db})


		for i in range(self.n_hidden, 0, -1):

			#Derivation of relu
			if(act_function=="relu"):
				d_activation = cache["Z"+str(i)]
				d_activation[d_activation<=0] = 0
				d_activation[d_activation>0] = 1

			#Derivation of sigmoid
			if(act_function=="sigmoid"):
				A = self.activation(cache["Z"+str(i)], function="sigmoid")
				d_activation = A * (1 - A)

			#Derivation of tanh
			if(act_function=="tanh"):
				A = self.activation(cache["Z"+str(i)], function="tanh")
				d_activation = 1 - np.power(A, 2)

			dZ = np.dot(parameters["W"+str(i+1)].T, dZ) * d_activation

			#if we have A0 then take X instead
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

		#################Explicitly code the derivation equations for every layer##################

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

	def train(self, iterations, init_method, learning_rate, X, labels, mini_batch=512, act_function="relu"):

		#One hot encoding of labels
		y = np.eye(self.n_class)[labels]

		parameters = self.initialize_weights(init_method)

		#Save avg loss in each epoch
		avg_loss = []

		#Save train and val acc in each epoch
		train_acc = []
		val_acc = []

		#Get size of mini batches
		nb_batchs = int(np.ceil(float(len(y)) / mini_batch))

		#init mini batch with shapes similar to X and y
		#X = (Mx, m) and y = (m, n_classes)
		batch_X = np.zeros((X.shape[0], mini_batch))
		batch_y = np.zeros((mini_batch, y.shape[1]))

		# Loop (gradient descent)
		for i in range(iterations):

			print "epoch: "+str(i)+"/"+str(iterations)

			start = 0

			#init mini batch with shapes similar to X and y
			#X = (Mx, m) and y = (m, n_classes)
			batch_X = np.zeros((X.shape[0], mini_batch))
			batch_y = np.zeros((mini_batch, y.shape[1]))

			#losses
			losses = []

			for batch in range(nb_batchs):
				end = start + mini_batch

				#If it exceeds the nb of examples
				if(end > X.shape[1]):
					end = X.shape[1]

				batch_X = X[:, start:end]
				batch_y = y[start:end, :]

				start = end

				#Forward pass
				out, cache = self.forward(batch_X, parameters)

				#Backward pass
				grads = self.backward(parameters, cache, batch_y, batch_X, act_function)

				#Update
				parameters = self.update(grads, parameters, learning_rate)

				#Loss for each batch
				losses.append(self.loss(out, batch_y.T))

				#print self.loss(out, batch_y.T)

			#Avg loss over batches
			avg_loss.append(np.sum(losses)/len(losses))
			print "Avg Train loss: "+str(np.sum(losses)/len(losses))

			#Training accuracy
			acc = self.test(self.D_train[0], self.D_train[1], parameters)
			print('Training Acc : %.3f ' % acc)
			train_acc.append(acc)

			#Validation accuracy
			acc = self.test(self.D_val[0], self.D_val[1], parameters)
			print('Validation Acc : %.3f ' % acc)
			val_acc.append(acc)

		#Plot loss curve
		self.visualize_loss(avg_loss, init_method, 'Training loss')

		#Plot training & validation accuracy curve
		self.visualize_acc(train_acc, val_acc, 'Training', 'Validation')

		return parameters


	def visualize_loss(self, x, title, label):
		epochs = range(len(x))

		plt.figure()

		plt.plot(epochs, x, 'b', label=label)
		plt.title(str(title)+" initializaton method")
		plt.xlabel('epoch')
		plt.ylabel('loss')
		plt.legend()
		plt.show()
		#plt.savefig(path+"accuracy.png")


	def visualize_acc(self, x1, x2, label1, label2):
		epochs = range(len(x1))

		plt.figure()

		plt.plot(epochs, x1, 'b', label=label1)
		plt.plot(epochs, x2, 'r', label=label2)
		plt.xlabel('epoch')
		plt.ylabel('accuracy')
		plt.legend()
		plt.show()
		#plt.savefig(path+"accuracy.png")


	#Evaluate model
	def test(self, X, y, parameters):

		out, _ = self.forward(X, parameters)
		predicted = np.argmax(out,axis=0)
		total = X.shape[1]
		correct = np.sum(predicted==y)
		acc = 100.*correct/total

		return acc


	def grad_check(self, epsilon, parameters, key, p, X, y, visualize=1):
		#One hot encoding of labels
		y = np.eye(self.n_class)[y]
		n,m = parameters[key].shape
		p = np.minimum(p,n*m)
		grad = np.zeros(p)
		grad_approx = np.zeros(p)
		for p_i in range(p):
			# indices
			j = np.int(np.floor(p_i/n))
			i = p_i-j*n

				#analytic gradient
			_, cache = self.forward(X, parameters)
			grads = self.backward(parameters, cache, y, X)
			grad[p_i] = grads['d'+key][i,j]
			# print('d%s[%d,%d] gradient = %.9f' % (key,i,j,grad))

			#finite difference approximation
			parameters[key][i,j] = parameters[key][i,j] + epsilon
			out, _ = nn.forward(X, parameters)
			loss_plus_e = nn.loss(out, y.T)
			parameters[key][i,j] = parameters[key][i,j] - 2*epsilon
			out, _ = nn.forward(X, parameters)
			loss_minus_e = nn.loss(out, y.T)
			grad_approx[p_i] = (loss_plus_e - loss_minus_e) / (2*epsilon)
			# print('d%s[%d,%d] gradient approximation = %.9f' %(key,i,j,grad_approx))

			#back to initial point
			parameters[key][i,j] = parameters[key][i,j] + epsilon

		if visualize == 1:
			plt.figure()
			plt.plot(grad, 'o', label='Analytic gradient')
			plt.plot(grad_approx,'rx', label='Gradient approximation')
			plt.title("Gradient checking for %s" %(key))
			plt.xlabel('parameter')
			plt.ylabel('gradient')
			plt.legend()
			plt.show()
		return np.abs(grad-grad_approx).max()


if __name__ == '__main__':

	nn = NN(hidden_dims=(512, 256), n_hidden=2, datapath='./datasets/mnist.pkl.npy')
	print("train/val/test: "+str(nn.dim_data))

	parameters = nn.train(10,'glorot', 20, nn.D_train[0], nn.D_train[1], mini_batch=64, act_function="relu")
	sd

	# parameters = nn.train(50,'glorot', 0.01, nn.D_train[0], nn.D_train[1], mini_batch=64, act_function="tanh")
	#print('Test Acc : %.3f ' % nn.test(nn.D_train[0], nn.D_train[1], parameters))
	#parameters = nn.train(20, 'glarot', 0.01, nn.D_train[0], nn.D_train[1])
	#print('-----training')
	#nn.test(nn.D_train[0],nn.D_train[1],parameters)
	#print('-----validation')
	#nn.test(nn.D_val[0],nn.D_val[1],parameters)

	#gd = nn.grad_check(0.000001,parameters,'b1', nn.D_train[0][:,2:3], nn.D_train[1][2:3])

	#parameters = nn.train(100, 'normal', 0.1, nn.D_train[0], nn.D_train[1])
	parameters = nn.train(100, 'glorot', 0.01, nn.D_train[0], nn.D_train[1], mini_batch=105)
	print('-----training')
	print(nn.test(nn.D_train[0],nn.D_train[1],parameters))
	print('-----validation')
	print(nn.test(nn.D_val[0],nn.D_val[1],parameters))


	K = 5
	I = 5
	N_array = []
	diff_array = []
	for i in np.arange(0,I+1):
		for k in np.arange(1,K+1):
			N = k*(10**i)
			N_array = np.append(N_array, N)
			epsilon = 1.0/N
			print(N)
			diff = nn.grad_check(epsilon,parameters,'W2', 10, nn.D_train[0][:,0:1], nn.D_train[1][0:1], visualize=0)
			diff_array = np.append(diff_array,diff)
	plt.figure()
	plt.plot(N_array, diff_array,  '*')
	plt.xscale('log')
	plt.title("Gradient checking for W2")
	plt.xlabel('N')
	plt.ylabel('Max. difference')
	plt.show()
