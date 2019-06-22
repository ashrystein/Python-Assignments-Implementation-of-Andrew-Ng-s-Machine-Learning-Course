import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from computeCost import computeCost
from gradientDescent import gradientDescent


#2
data = pd.read_csv('ex1data1.txt', header = None) #read from dataset
X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
data_top = data.head() # view first few rows of the data
print(data_top)


#2.1
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X,y,'rx')
plt.show()


#2.2
ones = np.ones((m,1))
y = y[:,np.newaxis]
X = X[:,np.newaxis]
X = np.hstack((ones, X)) # adding the intercept term
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01

J = computeCost(X,y,theta)
print('CostFunction: ',J)
theta = gradientDescent(X, y, theta, alpha, iterations)
print('Theta from gradientDescent: ',theta)

plt.plot(X[:,1], y,'rx',label='Training Data')
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta),'b',label='Linear regression')
plt.legend()
plt.show()