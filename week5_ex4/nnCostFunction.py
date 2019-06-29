import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sigmoid import sigmoid


def nnCostFunction(nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lam):

    Theta1 = np.reshape(nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1))

    m = len(y)
    ones = np.ones((m,1))
    a1 = np.hstack((ones, X))

    z2 =(a1 @ Theta1.T)
    a2 = sigmoid(z2)
    a2 = np.hstack((ones, a2))

    z3 = (a2 @ Theta2.T)
    h = sigmoid(z3)

    y_d = pd.get_dummies(y.flatten())

    temp1 = np.multiply(y_d, np.log(h))
    temp2 = np.multiply(1-y_d, np.log(1-h))
    temp3 = np.sum(temp1 + temp2)

    sum1 = np.sum(np.sum(np.power(Theta1[:,1:],2), axis = 1))
    sum2 = np.sum(np.sum(np.power(Theta2[:,1:],2), axis = 1))

    J = np.sum(temp3 / (-m)) + (sum1 + sum2) * lam / (2*m)

    return J