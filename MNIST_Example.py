# Run with exec(open('MNIST_Example.py').read())

# Do this in Mac Terminal with Python3 to save time
# import tensorflow as tf
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# import numpy as np
# np.save('x_train.npy', x_train)
# np.save('x_test.npy', x_test)
# np.save('y_train.npy', y_train)
# np.save('y_test.npy', y_test)

import numpy as np
import time

# Import data
# Shape is 60,000 by 28 by 28 or 60,000 by 1
x_train = np.load('x_train.npy')
x_test = np.load('x_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Shape inputs to 60,000 by 784
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train/255 - 0.5; # Subtracting 0.5 to give 0 mean found to improve results
x_test = x_test/255 - 0.5;
y_test_one_hot = np.zeros([10000, 10])
y_train_one_hot = np.zeros([60000, 10])

# Convert to one-hot vectors - shape 60,000 by 10. NB numbering goes from 0 to 9
count = 0
while count < 10000:
	y_test_one_hot[count, y_test[count]] = 1
	count = count + 1
y_test = y_test_one_hot

count = 0
while count < 60000:
	y_train_one_hot[count, y_train[count]] = 1
	count = count + 1
y_train = y_train_one_hot

# Weight matrices - NB 'W1' links from layer 0 to layer 1
# Note - no bias nodes are implemented here
W1 = np.random.randn(128, 784)
W2 = np.random.randn(64, 128)
W3 = np.random.randn(10, 64)
dW1 = np.zeros([128, 784])
dW2 = np.zeros([64, 128])
dW3 = np.zeros([10, 64])

# Activation and output matrices - NB 'A1' is the activation of layer 1
A0 = np.zeros([784, 1])
A1 = np.zeros([128, 1])
A2 = np.zeros([64, 1])
A3 = np.zeros([10, 1])
Z1 = np.zeros([128, 1])
Z2 = np.zeros([64, 1])
Z3 = np.zeros([10, 1])

# Sigmoid activation function chosen, but others could be trialled
def sigmoid(x):
	return 1/(1 + np.exp(-x))

def sigmoid_derivative(x):
	return (np.exp(-x))/((np.exp(-x) + 1)**2)

def forward_pass(x_train_single):
	global A0, A1, A2, A3, Z1, Z2, Z3
	A0 = x_train_single
	Z1 = np.dot(W1, A0)
	A1 = sigmoid(Z1)
	Z2 = np.dot(W2, A1)
	A2 = sigmoid(Z2)
	Z3 = np.dot(W3, A2)
	A3 = sigmoid(Z3)

def backward_pass(y_train_single):
	global dW1, dW2, dW3, W1, W2, W3, A0, A1, A2, A3, Z1, Z2, Z3

	# Go back one layer
	# lambda1 is 10 by 1
	lambda1 = 2*(A3 - y_train_single)*sigmoid_derivative(Z3)
	# dW3 is 10 by 64
	dW3 = np.outer(lambda1, A2)

	# Go back another layer
	# lambda2 is 64 by 1
	lambda2 = np.dot(np.transpose(W3), lambda1)
	lambda2 = lambda2*sigmoid_derivative(Z2)
	# dW2 is 64 by 128
	dW2 = np.outer(lambda2, A1)

	# Go back another layer
	# lambda3 is 128 by 1
	lambda3 = np.dot(np.transpose(W2), lambda2)
	lambda3 = lambda3*sigmoid_derivative(Z1)
	# dW1 is 128 by 784
	dW1 = np.outer(lambda3, A0)

# Finds the cost function C for a single traning example
def cost_function(y_train_single):
	global A3
	return np.sum(np.square(np.transpose(A3) - y_train_single))

# After 10 epochs the percentage accuracy was:
# 96.1% for the learning set of 60,000 data points
# 94.8% for the training set of 10,000 data points
def percentage_accuracy(x, y_one_hot, setSize):
	global A3
	noCorrect = 0
	iteration = 0
	while iteration < setSize:
		forward_pass(x[iteration,:])
		if(y_one_hot[iteration, np.argmax(A3)] == 1):
			noCorrect = noCorrect + 1
		iteration = iteration + 1
	return(100*noCorrect/setSize)

epochs = 0
while epochs < 10:
	count = 0
	while count < 60000:
		forward_pass(x_train[count,:])
		backward_pass(y_train[count,:])

		# Correct the weights with the derivative multiplied by learning rate
		# 0.05 found to be optimal learning rate - anything larger gives unstable learning
		W1 = W1 - 0.05*dW1
		W2 = W2 - 0.05*dW2
		W3 = W3 - 0.05*dW3
		
		count = count + 1
		if((count % 5000) == 0):
			print(epochs+1)
			print(count)
			forward_pass(x_train[0,:])
			print(cost_function(y_train[0,:]))
			# Pause to allow Mac heat management
			time.sleep(5)

	# Pause for Mac heat management
	time.sleep(20)
	epochs = epochs + 1







