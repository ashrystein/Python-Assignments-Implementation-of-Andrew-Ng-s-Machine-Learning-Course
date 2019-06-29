import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy.optimize as opt
from nnCostFunction import nnCostFunction


def computeNumericalGradient(initial_nn_params,input_layer_size,hidden_layer_size,num_labels,X, y,lam , D):
    eps = 1e-4
    sz = len(initial_nn_params)
    perturb = np.zeros((sz,1))
    numgrad = np.zeros(sz)
    for i in range(sz):#run this loop for 10 iterations and uncomment last line to see numerical and Backprob gradient
        perturb[i] = eps
        numgrad[i] = (nnCostFunction(initial_nn_params+perturb.flatten(),input_layer_size,hidden_layer_size,num_labels,X, y,lam)-nnCostFunction(initial_nn_params-perturb.flatten(),input_layer_size,hidden_layer_size,num_labels,X, y,lam))/float(2*eps)
        perturb[i] = 0
        print("Element: {0}. Numerical Gradient = {1:.9f}. BackProp Gradient = {2:.9f}.".format(i,numgrad[i],D[i]))
    return numgrad