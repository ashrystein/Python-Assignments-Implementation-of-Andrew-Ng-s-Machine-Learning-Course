import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from displayData import displayData
from costFunctionReg import costFunctionReg
from oneVsAll import oneVsAll
from predictNN import predictNN

#1.1
data = loadmat('ex3data1.mat')
X = data['X']
y = data['y']

#1.2
displayData(X)


#1.4
m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X))
(m,n) = X.shape

theta_optimized = oneVsAll(X,y)
#print(theta_optimized.shape)

pred = np.argmax(X @ theta_optimized.T, axis = 1)
pred = [e if e else 10 for e in pred]
acc = np.mean(pred == y.flatten()) * 100
print(acc , '%')


#2.1
weights = loadmat('ex3weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']
Theta2 = np.roll(Theta2, 1, axis=0)
input_layer_size  = 400
hidden_layer_size = 25


#2.2
predNN = predictNN(Theta1,Theta2,X)
#print(predNN)
acc = np.mean(predNN == y.flatten()) * 100
print(acc , '%')



