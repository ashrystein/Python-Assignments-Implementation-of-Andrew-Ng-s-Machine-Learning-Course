import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from featureNormalize import featureNormalize
from computeCostMulti import computeCostMulti,computeCostMulti2
from gradientDescentMulti import gradientDescentMulti

#3
data = pd.read_csv('ex1data2.txt', header = None) #read from dataset
X = data.iloc[:,0:2] # read first 2column
y = data.iloc[:,2] # read third column
m = len(y) # number of training example
data_top = data.head() # view first few rows of the data
print(data_top)

#3.1
X = featureNormalize(X)

#3.2
ones = np.ones((m,1))
y = y[:,np.newaxis]
X = np.hstack((ones, X)) # adding the intercept term
theta = np.zeros([3,1])
iterations = 400
alpha = 0.01

J = computeCostMulti(X,y,theta)
print('CostFunction: ',J)
J1 = computeCostMulti2(X,y,theta)
print('CostFunction2: ',J1)

theta = gradientDescentMulti(X, y, theta, alpha, iterations)
print('Theta from gradientDescent: ',theta)




