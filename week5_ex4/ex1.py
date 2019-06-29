import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from displayData import displayData
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from backPropagation import backPropagation
from computeNumericalGradient import computeNumericalGradient
from fmincg import fmincg
from predictNN import predictNN
from displayData import displayData

#1.1,1.2
data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']
#displayData(X)

#1.3,1.4
weights = loadmat('ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']

input_layer_size = 400
hidden_layer_size = 25
num_labels = 10
lam = 1
nn_params = np.hstack((Theta1.ravel(), Theta2.ravel()))

J = nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lam)
print(J)

#2
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

initial_nn_params = np.hstack((initial_Theta1.ravel(), initial_Theta2.ravel()))

D = backPropagation(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lam)
#print(D)

#Stop Numerical grd after checking
#num_grad = computeNumericalGradient(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lam , D)
#print(num_grad)

nn_theta_opt = fmincg(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lam)
#print(nn_theta_opt)

theta1_opt = np.reshape(nn_theta_opt[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
theta2_opt = np.reshape(nn_theta_opt[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1))
#print(theta1_opt)
#print(theta2_opt)


theta2_opt = np.roll(theta2_opt, 1, axis=0)
predNN = predictNN(theta1_opt,theta2_opt,X)
print('>>' , '\n' , predNN)
acc = np.mean(predNN == y.flatten()) * 100
print(acc , '%')

displayData(theta1_opt)