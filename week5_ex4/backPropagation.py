import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from sigmoid import sigmoid
from sigmoidGradient import sigmoidGradient

def backPropagation(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lam):
    Theta1 = np.reshape(initial_nn_params[:hidden_layer_size*(input_layer_size+1)], (hidden_layer_size, input_layer_size+1))
    Theta2 = np.reshape(initial_nn_params[hidden_layer_size*(input_layer_size+1):], (num_labels, hidden_layer_size+1))
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)
    m = len(y)
    ones = np.ones((m,1))
    X = np.hstack((ones, X))
    y_d = pd.get_dummies(y.flatten())


    for t in range(m):
        at = X[t]
        z2 =(at @ Theta1.T)
        a2 = sigmoid(z2)
        a2 = np.hstack((np.ones(1), a2))
        z3 = (a2 @ Theta2.T)
        a3 = sigmoid(z3)
        hypothesis = a3

        delta_layer3 = hypothesis - y_d.iloc[t,:][np.newaxis,:]
        z2 = np.hstack((np.ones(1), z2))
        delta_layer2  = np.multiply(Theta2.T @ delta_layer3.T , sigmoidGradient(z2).T[:,np.newaxis])
        delta_layer2 = delta_layer2[1:,:]

        Theta2_grad = Theta2_grad + (delta_layer3.T @ (a2[np.newaxis,:]))
        Theta1_grad = Theta1_grad + (delta_layer2 @ (at[np.newaxis,:]))

        delta1 = Theta1_grad / m
        delta2 = Theta2_grad / m
        
        delta1[:,1:] = delta1[:,1:] + Theta1[:,1:] * lam / m
        delta2[:,1:] = delta2[:,1:] + Theta2[:,1:] * lam / m
        
        Dvec = np.hstack((delta1.ravel(), delta2.ravel()))

    return Dvec