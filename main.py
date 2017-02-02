from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import math
import random
import csv


"""
Hyperparameters
	nHid: number of neurons in the hidden layer
	nIn: number of neurons in the input layer
	nOut: number of neurons in the output layer
	nEpochs: number of iterations through training data for learning the weights
	alpha: learning rate, begins decaying with 4th epoch
"""
nHid = 200
nIn = 784
nOut = 10
nEpochs = 6
alpha = 0.005

#loads and returns training labels, training data, testing data
def load_dataset():
    mndata = MNIST('./data/')
    X_train, labels_train = map(np.array, mndata.load_training())
    X_test, _ = map(np.array, mndata.load_testing())
    return X_train, labels_train, X_test

#activation function for the output layer
def softmax(z):
	r = [0]*10
	s = 0.0
	b = z.max()
	for j in range(10):
		s += np.exp(z[j] - b)
	for j in range(10):
		r[j] = np.exp(z[j] - b) / s
	return r

#compares an array of predictions with an array of labels and returns the accuracy as a decimal
def score(x, y):
	total = np.shape(x)[0]
	right = 0.0
	for i in range(np.shape(x)[0]):
		if(list(x[i]) == list(y[i])):
			right += 1.0
	return right / total

#predicts the labels given the weights, data, and size of the data
def predict(W, V, data, size):
	X = np.hstack((data[0:size],np.array([[1]*size]).T))
	S1 = np.vstack((np.dot(V, X.T), np.array([[1]*size]))).T
	H = S1 * (S1 > 0)
	S2 = np.dot(W, H.T)
	Z = np.apply_along_axis(softmax, 0, S2)
	Y_predicted = np.apply_along_axis(unhot, 1, Z.T)
	return Y_predicted

#calculates the loss function given the weights, labels, data, and size of the data
def loss(W, V, Y, data, size):
	X = np.hstack((data[0:size],np.array([[1]*size]).T))
	S1 = np.vstack((np.dot(V, X.T), np.array([[1]*size]))).T
	H = S1 * (S1 > 0)
	S2 = np.dot(W, H.T)
	Z = np.apply_along_axis(softmax, 0, S2)
	loss = np.array([0.0])

	"""
	Debug print statemnts to check for bad values in Z (output)

	if(0 in Z):
		print("warning2: cannot divide by zero")
		Z[Z==0] = [1]
	test = Z < 0
	if(True in test):
		print("warning3: should not have negative values in Z")
	"""

	for i in range(size):
		loss += -1.0 * np.dot(Y[i], np.apply_along_axis(np.log, 1, np.array([Z[:,i]])).T)
	return (np.sum(loss) / size)

#one-hot encodes a row of label probabilities
def unhot(row):
	row = list(row)
	z = row.index(max(row))
	return [0]*z + [1] + [0]*(10-1-z)

#Load data
X_train, labels_train, X_test = load_dataset()

#Initialize weight vectors W and V
V = np.random.normal(0, 0.01, (nHid, nIn+1))
W = np.random.normal(0, 0.01, (nOut, nHid+1))

#One-hot encode the labels
Y_train = np.zeros((60000, 10))
for x in range(60000):
	Y_train[x,labels_train[x]] = 1

#Regularize and center each feature
def preprocess(a):
	v = np.var(a)
	if(v==0):
		return (a - np.mean(a))
	return ((a - np.mean(a)) / v**0.5)
X_train = np.apply_along_axis(preprocess, 0, X_train)
X_test = np.apply_along_axis(preprocess, 0, X_test)

#Shuffle training data
indices = np.arange(60000)
random.shuffle(indices)
X_train = X_train[indices]
Y_train = Y_train[indices]

#Split training data
X_validation = X_train[0:10000]
Y_validation = Y_train[0:10000]
X_train = X_train[10000:]
Y_train = Y_train[10000:]

"""
Initialize values for debugging (see line 153)

counter=0
iterations = []
losses = []
accuracies = []
"""

#Learn the weights for neural net using training data
for e in range(nEpochs):
	indices = np.arange(50000)
	random.shuffle(indices)
	#decay learning rate each epoch beginning with the 4th epoch
	if(e > 3):
		alpha = alpha*0.5
	for s in indices:
		
		#Forward propogation
		x = np.hstack((np.array([X_train[s]]),np.array([[1]])))
		y = np.array([Y_train[s]])
		S1 = np.vstack((np.dot(V, x.T), np.array([[1]])))
		height = np.shape(S1)[0]
		width = np.shape(S1)[1]
		H = S1 * (S1 > 0)
		S2 = np.dot(W, H)
		Z = np.apply_along_axis(softmax, 0, S2)

		"""
		Debug print statements to check loss, training accuracy, and validation accuracy as the network learns the weights
		Requires initializations (see line 125)

		counter+=1
		if(counter%10000==0):
			los = loss(W, V, Y_train, X_train, 50000)
			Y_train_predicted = predict(W, V, X_train, 50000)
			sco = score(Y_train_predicted, Y_train[0:50000])
			Y_validation_predicted = predict(W, V, X_validation, 10000)
			print("loss: ")
			print(los)
			print("training accuracy: ")
			print(sco)
			print("validation accuracy: ")
			iterations += [counter]
			losses += [los]
			accuracies += [sco]
			sco2 = score(Y_validation, Y_validation_predicted)
			print(sco2)
		"""	

		#Backward Propogation
		D = Z.T - y
		gradW = np.dot(D.T, H.T)
		gradV = np.dot((np.dot(D, W[:,0:200]).T), x)
		height = np.shape(S1)[0] - 1
		for i in range(height):
			if(S1[i][0]) < 0:
				gradV[i] = [0]*785

		#Stochastic Gradient Descent updates
		W = W - alpha*gradW
		V = V - alpha*gradV

#Print the final validation accuracy which tests the neural net on 10000 data points held out from the training set whose labels are known
print("Final Validation Accuracy: ")
Y_v_p = predict(W, V, X_validation, 10000)
print(score(Y_v_p, Y_validation))

#Print the final training accuracy which tests the neural net on 50000 data points from the training set whose labels are known
print("Final Training Accuracy: ")
Y_t_p = predict(W, V, X_train, 50000)
print(score(Y_t_p, Y_train))

#Get the prediction vector for the testing data
Y_k_p = predict(W, V, X_test, 10000)

#Reshapes the prediction vector for easier outputting to csv file
def flats(vec):
	r = np.zeros((np.shape(vec)[0], 1))
	for i in range(np.shape(vec)[0]):
		row = list(vec[i])
		x = row.index(max(row))
		r[i][0] = int(x)
	return r

#Reshape prediction vector
Y_k_p = flats(Y_k_p)

#Save predictions to predictions.csv
with open('predictions.csv', 'wb') as csvfile:
    newwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    newwriter.writerow(['Id','Category'])
    for i in xrange(Y_k_p.shape[0]):
        first = i+1
        second = int(Y_k_p[i])
        newwriter.writerow([first, second])

"""
Saves debug info to a csv file (see lines 125 and 153)

with open('output.csv', 'wb') as csvfile:
    newwriter = csv.writer(csvfile, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
    newwriter.writerow(['Iterations','Loss', 'Train Acc'])
    for i in xrange(len(iterations)):
        first = iterations[i]
        second = losses[i]
        third = accuracies[i]
        newwriter.writerow([first, second, third])
"""