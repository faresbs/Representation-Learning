# Create a Neural Network (MLP) with 2 hidden layers
# Package imports

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets as datasets
import sklearn.linear_model

np.random.seed(1) # set a seed so that the results are consistent

# The breast cancer dataset
data = datasets.load_breast_cancer()

# Define The NN structure
def structure(X, n_y, n_h):
	
	n_X = X.shape[1]										# Number of input neurones
	n_y = n_y												# Number of output neurones 
	n_H = n_h 												# Number of hidden neurones

	return (n_X, n_H, n_y)


# Initialize parameters with random values to break symmetry
def initialize_parameters(n_X, n_H, n_y):
	W1 = np.random.randn(n_H, n_X) * 0.01 
	b1 = np.random.randn(1, n_H) * 0.01
	W2 = np.random.randn(n_y, n_H) * 0.01
	b2 = np.random.randn(n_y, 1) * 0.01

	parameters = {"W1":W1, "b1":b1, "W2":W2, "b2":b2}

	return parameters 


# Sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# Forward Prop
def Forward_propagation(X, parameters):
	
	# Retrieve parameters
	W1 = parameters["W1"]
	b1 = parameters["b1"]
	W2 = parameters["W2"]
	b2 = parameters["b2"]

	# Forward pass in hidden layer
	Z1 = np.dot(X, W1.T) + b1
	A1 = np.tanh(Z1)
	# Forward pass in the output layer
	Z2 = np.dot(A1, W2.T) + b2
	A2 = sigmoid(Z2)

	#cache = {Z1:"Z1", Z2:"Z2", A1:"A1", A2:"A2"} 

	return A2, A1


# Compute the cross entropy cost
def cost_function(A2, y):

	# Reshape y from (m, ) to (m, 1)
	y = y.reshape(y.shape[0], 1) 
	m = np.shape(y)[0]
	
	J = 0
	J = J + ((-1./m) * (np.dot(y.T, np.log(A2)) + np.dot((1 - y).T, np.log(1 - A2))))

	J = np.squeeze(J)
	return J


# Backpropgation
def backprop(parameters, A2, A1, X, y):
	
	# Reshape y from (m, ) to (m, 1)
	y = y.reshape(y.shape[0], 1) 
	
	m = y.shape[0]

	#Retrive parameters
	W2 = parameters["W2"]
	W1 = parameters["W1"]
	b2 = parameters["b2"]
	b1 = parameters["b1"]

	dZ2 = A2 - y
	dW2 = (1./m) * np.dot(dZ2.T, A1)
	db2 = (1./m) * np.sum(dZ2, axis=0, keepdims=True)

	# Derivation of tanh is 1 - np.power(A1, 2)
	dZ1 = (1./m) * (np.dot(dZ2, dW2) * (1 - np.power(A1, 2)))
	dW1 = np.dot(dZ1.T, X)
	db1 = (1./m) * np.sum(dZ1, axis=0, keepdims=True)

	# gradients must have same dimension as the parameters
	assert(dW2.shape == W2.shape)
	assert(db2.shape == b2.shape)
	assert(dW1.shape == W1.shape)
	assert(db1.shape == b1.shape)

	grads = {"dW2":dW2, "db2":db2, "dW1":dW1, "db1":db1}

	return grads


# Update 
def Update(parameters, grads, learning_rate):
	
	# Retrieve parameters
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W1 = parameters["W1"]
	b1 = parameters["b1"]

	# Retrieve grads
	dW2 = grads["dW2"]
	db2 = grads["db2"]
	dW1 = grads["dW1"]
	db1 = grads["db1"]

	# Update parameters using gradient descent
	W2 = W2 - learning_rate * dW2
	W1 = W1 - learning_rate * dW1
	b2 = b2 - learning_rate * db2
	b1 = b1 - learning_rate * db1

	# Save updated parameters
	parameters = {"W2":W2, "W1":W1, "b2":b2, "b1":b1}
	
	return parameters


# Put all together
def model(X, y, n_y, n_h, iterations, learning_rate):
	(n_X, n_H, n_y) = structure(X, n_y, n_h)
	parameters = initialize_parameters(n_X, n_H, n_y)

	# Retrieve parameters
	W2 = parameters["W2"]
	b2 = parameters["b2"]
	W1 = parameters["W1"]
	b1 = parameters["b1"]

	# Loop (gradient descent)
	for i in range(iterations):
		#Forward pass
		A2, A1 = Forward_propagation(X, parameters)
		#Backward pass
		grads = backprop(parameters, A2, A1, X, y)
		#Update
		parameters = Update(parameters, grads, learning_rate) 

		# FIX ME : why cost is not decreasing

		print cost_function(A2, y)

	return parameters	

# Make predictions wil new examples
def predict(X, parameters):
	A2, A1 = Forward_propagation(X, parameters)
	if(A2 >= 0.5):
		p = 1
	else :
		p = 0
	return p


X = data.data
y = data.target

#print X[2]
#print y[2] 

param = model(X, y, 1, 4, 1000, 0.1)
#print predict(X[20], param)